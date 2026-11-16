"""Redis-backed sliding-window rate limit.

Shared store (the Celery broker Redis, ``CELERY_BROKER_URL``): the
limit is global across API processes, not per-process — the old
in-memory version let each replica keep its own counters. One sorted
set per key, scored by hit time; a hit older than the window is
dropped before counting.

On Redis errors the check fails OPEN and logs a warning: the limit is
a guard, not auth, and a Redis outage must not take guest chat down
with it (dispatch would fail anyway while the broker is down). Two
concurrent requests at the boundary can both pass — the check-then-add
is not atomic — an overshoot of at most the concurrency level, which
is acceptable for a guard.

The guest chat endpoint is the only platform caller.
"""

from __future__ import annotations

import itertools
import logging
import time

import redis
from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

# Guest chat: 20 messages per minute per user. The limit is a module
# constant so tests can lower it with monkeypatch.
GUEST_CHAT_LIMIT = 20
GUEST_CHAT_WINDOW_SECONDS = 60

_KEY_PREFIX = "ratelimit:guest_chat:"

_client: redis.Redis | None = None
_counter = itertools.count()


def _redis_client() -> redis.Redis:
    """The shared Redis client (cached; tests replace ``_client``)."""
    global _client
    if _client is None:
        _client = redis.Redis.from_url(
            settings.celery_broker_url, decode_responses=True
        )
    return _client


def _check(client, key: str, limit: int, window_seconds: int) -> bool:
    """True when the call is within the limit (and records the hit)."""
    now = time.monotonic()
    client.zremrangebyscore(key, 0, now - window_seconds)
    if client.zcard(key) >= limit:
        return False
    member = f"{time.monotonic_ns()}-{next(_counter)}"
    client.zadd(key, {member: now})
    client.expire(key, window_seconds)
    return True


def check_guest_rate_limit(user_id: str) -> None:
    """Raise 429 when the user sent too many guest-chat messages."""
    try:
        allowed = _check(
            _redis_client(),
            f"{_KEY_PREFIX}{user_id}",
            GUEST_CHAT_LIMIT,
            GUEST_CHAT_WINDOW_SECONDS,
        )
    except Exception:
        logger.warning("guest rate limit unavailable — failing open", exc_info=True)
        return
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many requests — slow down")


def reset_guest_rate_limit() -> None:
    """Clear all guest counters (tests). Best-effort on Redis errors."""
    try:
        client = _redis_client()
        for key in client.keys(f"{_KEY_PREFIX}*"):
            client.delete(key)
    except Exception:
        logger.warning("guest rate limit reset failed", exc_info=True)
