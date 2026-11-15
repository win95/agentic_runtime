"""In-memory sliding-window rate limit (demo).

Single-process only: each API process keeps its own counters, so the
limit is per-process, not global. Production needs a shared store
(Redis). The guest chat endpoint is the only caller.
"""

from __future__ import annotations

import threading
import time

from fastapi import HTTPException

# Guest chat: 20 messages per minute per user. The limit is a module
# constant so tests can lower it with monkeypatch.
GUEST_CHAT_LIMIT = 20
GUEST_CHAT_WINDOW_SECONDS = 60

_hits: dict[str, list[float]] = {}
_lock = threading.Lock()


def check_guest_rate_limit(user_id: str) -> None:
    """Raise 429 when the user sent too many guest-chat messages."""
    now = time.monotonic()
    with _lock:
        hits = [t for t in _hits.get(user_id, []) if now - t < GUEST_CHAT_WINDOW_SECONDS]
        if len(hits) >= GUEST_CHAT_LIMIT:
            raise HTTPException(
                status_code=429, detail="Too many requests — slow down"
            )
        hits.append(now)
        _hits[user_id] = hits


def reset_guest_rate_limit() -> None:
    """Clear all counters (tests)."""
    with _lock:
        _hits.clear()
