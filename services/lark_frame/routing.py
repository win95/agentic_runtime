"""Routing for the Lark frame (agent_lark_me, Step 2).

Pure logic — no I/O, no Lark SDK, no platform calls — so both
transports (webhook + WS channel) share exactly one routing layer.
Transport-specific code normalizes its event into a
``NormalizedMessage`` and calls :func:`route_message`.

Rules, in order (a rule that fails means "no enqueue"):
1. Only text messages are routed (cards, files, ... are dropped).
2. The bot's own messages are skipped (sender_type == "app").
3. Group/topic messages route only when the bot is @-mentioned
   (mentions carry the bot's open_id). DMs route directly. This
   check runs BEFORE the allowlist and the rate limit, so chatter
   that never calls the bot costs nobody a refusal or rate budget
   (P1-3). A card action (``via_card_action=True``) skips it — a
   button press on the bot's own card addresses the bot by itself.
4. Only allowlisted senders route (D-LARK-06) — outsiders get one
   refusal from the frame (rate-limited to 1/hour/user) and no run.
5. Per-sender rate limit (D-LARK-17): 10 messages/minute/open_id.
   The first drop in a window gets one "too fast" notice; the rest
   are silent.
6. ``session_id`` is per (chat, user): ``"lark-" +
   sha1(chat_id + "|" + open_id)[:16]`` (21 chars, fits the platform's
   64-char session pattern; deterministic, so the frame recomputes it
   every time — no mapping table).

Dedupe is NOT here: the caller checks the seen-set BEFORE routing
(a duplicate must not consume rate-limit budget) and marks it ONLY
after a successful enqueue (D-LARK-19) — so a platform failure lets
Lark retry the event instead of losing the message.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

# Routing decisions (D-LARK-06/17).
RATE_LIMIT_PER_MINUTE = 10
REFUSAL_PER_HOUR = 1
RATE_NOTICE_PER_WINDOW = 1
PLATFORM_DOWN_NOTICE_PER_5MIN = 1

REFUSAL_TEXT = (
    "Xin lỗi, bot này chỉ trả lời người trong danh sách cho phép. "
    "Liên hệ admin để được thêm vào."
)
RATE_LIMIT_TEXT = "Bạn nhắn quá nhanh, thử lại sau ít phút."
PLATFORM_DOWN_TEXT = "Hệ thống đang bận, thử lại sau ít phút."


@dataclass
class NormalizedMessage:
    """One inbound Lark message, transport-neutral."""

    message_id: str
    root_id: str | None
    chat_id: str
    chat_type: str  # "p2p" | "group" | "topic"
    sender_type: str  # "user" | "app"
    sender_id: dict  # {union_id, user_id, open_id}
    text: str
    mentions: list = field(default_factory=list)  # open_ids mentioned
    msg_type: str = "text"


@dataclass
class RouteDecision:
    action: str  # "route" | "drop" | "refuse" | "rate_limited"
    reason: str
    session_id: str | None = None
    message: NormalizedMessage | None = None


def session_id_for(chat_id: str, open_id: str) -> str:
    """Per-(chat, user) session id (D-LARK-11)."""
    digest = hashlib.sha1(f"{chat_id}|{open_id}".encode("utf-8")).hexdigest()
    return f"lark-{digest[:16]}"


class RateLimiter:
    """Fixed-window per-sender limits, thread-safe (D-LARK-17).

    Three counters share one lock: message rate (10/min), refusal
    rate (1/hour) and rate-limit notices (1/window). Windows are
    fixed (aligned to the clock) — simple and good enough for a
    pre-enqueue guard; the platform's Redis limits stay the second
    line of defence.
    """

    def __init__(
        self,
        per_minute: int = RATE_LIMIT_PER_MINUTE,
        refusal_per_hour: int = REFUSAL_PER_HOUR,
    ):
        self.per_minute = per_minute
        self.refusal_per_hour = refusal_per_hour
        self._lock = threading.Lock()
        self._msg_windows: dict[str, list] = {}  # open_id -> [window, count]
        self._refusal_windows: dict[str, list] = {}
        self._notice_windows: dict[str, list] = {}

    def _check(self, table: dict, key: str, window_seconds: int, limit: int) -> bool:
        now = int(time.time())
        window = now // window_seconds
        with self._lock:
            entry = table.get(key)
            if entry is None or entry[0] != window:
                table[key] = [window, 1]
                return True
            if entry[1] >= limit:
                return False
            entry[1] += 1
            return True

    def allow_sender(self, open_id: str) -> bool:
        """True when this sender may send one more message this minute."""
        return self._check(self._msg_windows, open_id, 60, self.per_minute)

    def allow_refusal(self, open_id: str) -> bool:
        """True when a refusal may be sent to this sender this hour."""
        return self._check(self._refusal_windows, open_id, 3600, self.refusal_per_hour)

    def allow_rate_notice(self, open_id: str) -> bool:
        """True only for the first rate-limited drop in this minute."""
        return self._check(
            self._notice_windows, open_id, 60, RATE_NOTICE_PER_WINDOW
        )

    def allow_platform_down_notice(self, open_id: str) -> bool:
        """True only for the first platform-down notice in 5 minutes."""
        return self._check(
            self._notice_windows, f"down:{open_id}", 300,
            PLATFORM_DOWN_NOTICE_PER_5MIN,
        )


class SeenSet:
    """Dedupe by message_id (D-LARK-19).

    Redis when reachable (``REDIS_URL`` — the same Redis the platform
    uses), in-memory fallback otherwise. Redis is the durable path:
    a frame restart must not re-enqueue messages Lark already saw
    answered. The in-memory fallback logs a warning and accepts the
    restart risk. ``mark`` is called ONLY after a successful enqueue.
    """

    def __init__(self, redis_url: str | None = None, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._memory: dict[str, float] = {}
        self._redis = None
        self._redis_error_logged = False
        if redis_url:
            try:
                import redis as redis_lib

                self._redis = redis_lib.Redis.from_url(
                    redis_url, socket_connect_timeout=2, socket_timeout=2
                )
                self._redis.ping()
            except Exception as exc:  # noqa: BLE001 — Redis is optional
                self._redis = None
                import logging

                logging.getLogger(__name__).warning(
                    "SeenSet: Redis unreachable (%s) — in-memory dedupe only", exc
                )

    def contains(self, key: str) -> bool:
        if self._redis is not None:
            try:
                return bool(self._redis.exists(f"lark:seen:{key}"))
            except Exception:  # noqa: BLE001 — fall through to memory
                if not self._redis_error_logged:
                    self._redis_error_logged = True
                    import logging

                    logging.getLogger(__name__).warning(
                        "SeenSet: Redis read failed — in-memory dedupe only"
                    )
        with self._lock:
            expiry = self._memory.get(key)
            return expiry is not None and expiry > time.time()

    def mark(self, key: str) -> None:
        if self._redis is not None:
            try:
                self._redis.set(
                    f"lark:seen:{key}", "1", nx=True, ex=self.ttl_seconds
                )
                return
            except Exception:  # noqa: BLE001 — fall through to memory
                pass
        with self._lock:
            self._memory[key] = time.time() + self.ttl_seconds
            if len(self._memory) > 10000:
                now = time.time()
                for k in [k for k, v in self._memory.items() if v <= now]:
                    del self._memory[k]


def route_message(
    msg: NormalizedMessage,
    *,
    bot_open_id: str | None,
    allowlist: set[str],
    rate_limiter: RateLimiter,
    via_card_action: bool = False,
) -> RouteDecision:
    """Decide what the frame does with one inbound message.

    ``via_card_action``: the operator pressed a button on the bot's
    own card — that addresses the bot by itself, so the @mention
    check is skipped (the allowlist + rate limit still apply).
    """
    if msg.msg_type != "text":
        return RouteDecision("drop", "not_text", message=msg)
    if msg.sender_type == "app":
        return RouteDecision("drop", "own_message", message=msg)
    open_id = (msg.sender_id or {}).get("open_id") or ""
    if not open_id:
        return RouteDecision("drop", "no_sender_open_id", message=msg)
    if not via_card_action and msg.chat_type in ("group", "topic"):
        if bot_open_id:
            if bot_open_id not in (msg.mentions or []):
                return RouteDecision("drop", "not_mentioned", message=msg)
        elif not msg.mentions:
            # Loose mode (bot open_id unknown — see README): route
            # only when somebody was @-mentioned at all.
            return RouteDecision("drop", "not_mentioned", message=msg)
    if open_id not in allowlist:
        if rate_limiter.allow_refusal(open_id):
            return RouteDecision("refuse", "not_allowlisted", message=msg)
        return RouteDecision("drop", "refusal_rate_limited", message=msg)
    if not rate_limiter.allow_sender(open_id):
        return RouteDecision("rate_limited", "rate_limited", message=msg)
    return RouteDecision(
        "route",
        "ok",
        session_id=session_id_for(msg.chat_id, open_id),
        message=msg,
    )
