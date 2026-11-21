"""Execution lifecycle events — Redis pub/sub for the SSE stream.

The worker publishes one small JSON event per status transition
(``QUEUED`` is implicit in the 202 response; ``RUNNING``, then the
terminal status — or ``INTERRUPTED`` when a human approval waits, see
docs/interrupt.md). ``GET /executions/{id}/stream`` subscribes and
relays them as Server-Sent Events; token-by-token model output is
deliberately out of scope for v1.

Best-effort by design, on both sides:

* Publishing never raises and never blocks a run — a missing or
  unreachable Redis means the event is simply lost. The broker URL
  is not always Redis (tests use ``memory://``), in which case the
  publisher is a no-op.
* Subscribing returns ``None`` when Redis is unavailable; the SSE
  endpoint then falls back to polling the executions table, so the
  stream still works — just with 1s granularity instead of instant.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

CHANNEL_PREFIX = "exec_events:"

TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})

#: A stream closes after this long even if the run is still going — the
#: client reconnects (or polls) from there. SSE is a notification
#: channel, not a lock on a worker thread.
MAX_STREAM_SECONDS = 600.0

#: How often the stream sends a ``: ping`` comment so proxies and
#: clients know the connection is alive.
HEARTBEAT_INTERVAL_SECONDS = 15.0

_client: Any = None
_client_tried = False
_NO_REDIS = object()


def channel_name(execution_id: str) -> str:
    return f"{CHANNEL_PREFIX}{execution_id}"


def build_event(
    event_type: str, execution_id: str, status: str | None = None, **extra: Any
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": event_type,
        "execution_id": execution_id,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    if status is not None:
        event["status"] = status
    event.update(extra)
    return event


def _redis_client() -> Any | None:
    """The shared Redis client, or None when the broker is not Redis.

    The client is built lazily from the Celery broker URL (the same
    Redis the rate limiter uses). A non-Redis URL (``memory://`` in
    tests) means events are unavailable, not an error.
    """
    global _client, _client_tried
    if _client_tried:
        return None if _client is _NO_REDIS else _client
    _client_tried = True
    try:
        from app.config import settings

        url = settings.celery_broker_url
        if not url.startswith(("redis://", "rediss://")):
            _client = _NO_REDIS
            return None
        import redis

        _client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        return _client
    except Exception as exc:  # noqa: BLE001 — events never break a run
        logger.debug("Execution events unavailable: %s", exc)
        _client = _NO_REDIS
        return None


def publish_execution_event(
    execution_id: str, event_type: str, status: str | None = None, **extra: Any
) -> bool:
    """Publish one lifecycle event. Returns False when events are
    unavailable — the caller never checks this; a lost event is fine
    because the SSE endpoint also polls the executions table."""
    client = _redis_client()
    if client is None:
        return False
    try:
        client.publish(
            channel_name(execution_id),
            json.dumps(build_event(event_type, execution_id, status, **extra)),
        )
        return True
    except Exception as exc:  # noqa: BLE001 — events never break a run
        logger.debug("Could not publish execution event: %s", exc)
        return False


def subscribe_execution_events(execution_id: str) -> Any | None:
    """A pubsub subscribed to the execution's channel, or None when
    Redis is unavailable. The caller falls back to DB polling."""
    client = _redis_client()
    if client is None:
        return None
    try:
        pubsub = client.pubsub()
        pubsub.subscribe(channel_name(execution_id))
        return pubsub
    except Exception as exc:  # noqa: BLE001 — fall back to DB polling
        logger.debug("Could not subscribe to execution events: %s", exc)
        return None
