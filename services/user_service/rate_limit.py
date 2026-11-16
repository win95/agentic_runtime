"""Redis-backed sliding-window rate limit (self-contained copy).

Same shape as the platform's ``app/core/rate_limit.py`` — duplicated,
not imported, because this service never imports the platform app.
One sorted set per key, scored by hit time; a hit older than the
window is dropped before counting.

On Redis errors the check fails OPEN and logs a warning: the limit is
a guard, not auth, and a Redis outage must not lock every user out of
login. Two concurrent requests at the boundary can both pass — the
check-then-add is not atomic — an overshoot of at most the
concurrency level, which is acceptable for a guard.
"""

from __future__ import annotations

import itertools
import logging
import time

import redis
from fastapi import HTTPException

from services.user_service.config import settings

logger = logging.getLogger(__name__)

# Login: 5 attempts per minute per (email, IP). Keyed on both so one
# attacker's failures cannot lock the victim out from another IP.
LOGIN_LIMIT = 5
LOGIN_WINDOW_SECONDS = 60
# Register: 5 accounts per minute per IP (account-creation spam).
REGISTER_LIMIT = 5
REGISTER_WINDOW_SECONDS = 60

_KEY_PREFIX = "ratelimit:user_service:"

_client: redis.Redis | None = None
_counter = itertools.count()


def _redis_client() -> redis.Redis:
    """The shared Redis client (cached; tests replace ``_client``)."""
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return _client


def check_rate_limit(key: str, limit: int, window_seconds: int) -> None:
    """Raise 429 when the key sent too many requests in the window."""
    full_key = f"{_KEY_PREFIX}{key}"
    try:
        now = time.monotonic()
        client = _redis_client()
        client.zremrangebyscore(full_key, 0, now - window_seconds)
        if client.zcard(full_key) >= limit:
            raise HTTPException(status_code=429, detail="Too many requests — slow down")
        member = f"{time.monotonic_ns()}-{next(_counter)}"
        client.zadd(full_key, {member: now})
        client.expire(full_key, window_seconds)
    except HTTPException:
        raise
    except Exception:
        logger.warning("rate limit unavailable — failing open", exc_info=True)


def reset_rate_limit() -> None:
    """Clear all counters (tests). Best-effort on Redis errors."""
    try:
        client = _redis_client()
        for key in client.keys(f"{_KEY_PREFIX}*"):
            client.delete(key)
    except Exception:
        logger.warning("rate limit reset failed", exc_info=True)
