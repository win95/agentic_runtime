"""Redis-backed sliding-window rate limit with bounded in-process fallback.

Shared store (the Celery broker Redis, ``CELERY_BROKER_URL``): the
limit is global across API processes when Redis is healthy. When Redis
is unavailable the check uses a bounded in-process window (fail closed
per process — stricter than fail-open, but guests can still chat on
other replicas). Two concurrent requests at the boundary can both pass.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections import defaultdict, deque

import redis
from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

GUEST_CHAT_LIMIT = 20
GUEST_CHAT_WINDOW_SECONDS = 60
_FALLBACK_MAX_KEYS = 4096

_KEY_PREFIX = "ratelimit:guest_chat:"

_client: redis.Redis | None = None
_counter = itertools.count()
_fallback: dict[str, deque[float]] = defaultdict(deque)


def _redis_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(
            settings.celery_broker_url, decode_responses=True
        )
    return _client


def _check_redis(client, key: str, limit: int, window_seconds: int) -> bool:
    now = time.monotonic()
    client.zremrangebyscore(key, 0, now - window_seconds)
    if client.zcard(key) >= limit:
        return False
    member = f"{time.monotonic_ns()}-{next(_counter)}"
    client.zadd(key, {member: now})
    client.expire(key, window_seconds)
    return True


def _check_fallback(user_id: str, limit: int, window_seconds: int) -> bool:
    """Bounded in-process sliding window when Redis is down."""
    now = time.monotonic()
    if len(_fallback) > _FALLBACK_MAX_KEYS and user_id not in _fallback:
        return False
    window = _fallback[user_id]
    while window and window[0] <= now - window_seconds:
        window.popleft()
    if len(window) >= limit:
        return False
    window.append(now)
    return True


def check_guest_rate_limit(user_id: str) -> None:
    """Raise 429 when the user sent too many guest-chat messages."""
    try:
        allowed = _check_redis(
            _redis_client(),
            f"{_KEY_PREFIX}{user_id}",
            GUEST_CHAT_LIMIT,
            GUEST_CHAT_WINDOW_SECONDS,
        )
    except Exception:
        logger.warning(
            "guest rate limit unavailable — using bounded in-process fallback",
            exc_info=True,
        )
        if not _check_fallback(user_id, GUEST_CHAT_LIMIT, GUEST_CHAT_WINDOW_SECONDS):
            raise HTTPException(status_code=429, detail="Too many requests — slow down")
        return
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many requests — slow down")


def reset_guest_rate_limit() -> None:
    """Clear all guest counters (tests). Best-effort on Redis errors."""
    _fallback.clear()
    try:
        client = _redis_client()
        for key in client.keys(f"{_KEY_PREFIX}*"):
            client.delete(key)
    except Exception:
        logger.warning("guest rate limit reset failed", exc_info=True)