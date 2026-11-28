"""Webhook frame for agent_lark_me (Step 2, D-LARK-04/19).

``ThreadingHTTPServer`` (stdlib) + urllib — the webhook frame needs no
Lark SDK. Two public URLs (one server, two paths):

- ``POST /lark/events`` — event subscription (Request URL in the
  Lark console). Answers the ``url_verification`` challenge, verifies
  the SHA256 signature + ``app_id``, routes the message and enqueues
  the run, then returns 200. A platform failure returns 500 so Lark
  retries the event (the message is marked seen only after a
  successful enqueue — D-LARK-19).
- ``POST /lark/card`` — card callback (Callback Configuration in the
  console). Answers its own ``url_verification`` challenge, verifies
  the signature, parses ``action.value`` (``{action, chat_id,
  incident_ref}`` — D-LARK-21) and enqueues the resolve run. The
  response goes back immediately (a toast) and the heavy work runs in
  a background thread — the card UI waits on the response, so the
  message path's "enqueue then 200" does not apply here.

Health (D-LARK-18): ``GET /healthz`` (liveness) and ``GET /readyz``
(platform reachable). In webhook mode they run on the same server as
the event routes; in websocket mode they run on a separate health-only
server (``start_health_server``, ``FRAME_HEALTH_PORT``, default 8002),
because the WS channel opens no HTTP server of its own.

Verification, in order, on every request: SHA256 signature
(``X-Lark-Signature`` = sha256(timestamp + nonce + encrypt_key +
body) — only Lark, holding the encrypt key, can produce one — plus
a freshness check: the ``X-Lark-Request-Timestamp`` must be within
5 minutes of now, so a captured request cannot be replayed later),
then the verification token and ``app_id`` in the event header
(``event_callback``), or the top-level token (``card.action.trigger``
— card callbacks carry no ``app_id``; the signature + token are the
authentication there). Webhook mode refuses to start without
``LARK_ENCRYPT_KEY`` and ``LARK_VERIFICATION_TOKEN`` (fail-fast in
``load_config``). Bodies are capped at 1 MB. Any failure is 401
(fail-closed); nothing is logged that contains a secret, a token
or message content beyond the routing decision.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from services.lark_frame import platform as platform_io
from services.lark_frame.routing import (
    PLATFORM_DOWN_TEXT,
    RATE_LIMIT_TEXT,
    REFUSAL_TEXT,
    NormalizedMessage,
    RateLimiter,
    RouteDecision,
    SeenSet,
    route_message,
    session_id_for,
)

logger = logging.getLogger(__name__)

MISSION_ID = "agent_lark_me_mission"

# A signed request older than this is a replay, not a live event.
SIGNATURE_MAX_AGE_SECONDS = 300
# Lark event bodies are small JSON; anything larger is not a Lark event.
MAX_CONTENT_LENGTH = 1_000_000


@dataclass
class FrameConfig:
    app_id: str
    app_secret: str
    verification_token: str = ""
    encrypt_key: str = ""
    platform_base_url: str = "http://localhost:8000"
    platform_api_key: str = ""
    allowlist: set = field(default_factory=set)
    bot_open_id: str | None = None  # None -> fetch at startup via bot/v3/info
    host: str = "0.0.0.0"
    port: int = 8000
    # Health-only server in websocket mode (D-LARK-18): FRAME_PORT
    # defaults to 8000, which the platform already occupies, so the
    # health server needs its own port.
    health_port: int = 8002
    redis_url: str | None = None
    transport: str = "webhook"
    # Mission to enqueue (env LARK_MISSION_ID; default is the
    # agent_lark_me mission).
    mission_id: str = MISSION_ID


class FrameState:
    """Shared frame state (thread-safe by construction)."""

    def __init__(self, config: FrameConfig):
        self.config = config
        self.rate_limiter = RateLimiter()
        self.seen = SeenSet(config.redis_url)
        self.started_at = time.time()
        self.bot_open_id = config.bot_open_id

    def log_event(self, **fields) -> None:
        """One structured log line per event (for 'bot im lặng' debug)."""
        logger.info("lark event %s", " ".join(f"{k}={v}" for k, v in fields.items()))

    def ensure_bot_open_id(self) -> None:
        """Learn the bot's own open_id once (for @bot detection)."""
        if self.bot_open_id:
            return
        self.bot_open_id = platform_io.fetch_bot_open_id(
            self.config.app_id, self.config.app_secret
        )
        if self.bot_open_id:
            logger.info("Lark bot open_id resolved")
        else:
            logger.warning(
                "Lark bot open_id unknown — group routing falls back to "
                "loose mode (any mention routes); set LARK_BOT_OPEN_ID "
                "or check the app credentials"
            )


def verify_signature(headers, body: bytes, encrypt_key: str) -> bool:
    """SHA256 request signature (the SDK's exact algorithm) + freshness.

    signature = sha256(timestamp + nonce + encrypt_key + body).
    Fail-closed: without an encrypt key there is nothing to verify
    against, so verification fails (webhook mode refuses to start
    without a key — see ``load_config``). The timestamp must parse
    and be within ``SIGNATURE_MAX_AGE_SECONDS`` of now, so a captured
    request cannot be replayed later.
    """
    if not encrypt_key:
        return False
    timestamp = headers.get("X-Lark-Request-Timestamp") or ""
    nonce = headers.get("X-Lark-Request-Nonce") or ""
    signature = headers.get("X-Lark-Signature") or ""
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    if abs(time.time() - ts) > SIGNATURE_MAX_AGE_SECONDS:
        return False
    expected = hashlib.sha256(
        (timestamp + nonce + encrypt_key).encode("utf-8") + body
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


def _token_matches(token: str, verification_token: str) -> bool:
    """A verification token matches only when one is configured."""
    if not verification_token:
        return False
    return hmac.compare_digest(token or "", verification_token)


def decrypt_payload(body: dict, encrypt_key: str) -> dict:
    """Decrypt an encrypted event body (``{"encrypt": ...}``).

    Same algorithm as the SDK's AESCipher: key = sha256(encrypt_key),
    IV = first 16 bytes of the base64-decoded ciphertext, AES-CBC +
    PKCS7. Raises ValueError when decryption fails.
    """
    encrypt = body.get("encrypt")
    if not encrypt:
        return body
    if not encrypt_key:
        raise ValueError("encrypted event but no LARK_ENCRYPT_KEY configured")
    try:
        from Crypto.Cipher import AES
    except ImportError as exc:
        raise ValueError("pycryptodome is not installed") from exc
    key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    raw = base64.b64decode(encrypt)
    if len(raw) < AES.block_size * 2 or len(raw) % AES.block_size != 0:
        raise ValueError("invalid ciphertext length")
    cipher = AES.new(key, AES.MODE_CBC, raw[: AES.block_size])
    plain = cipher.decrypt(raw[AES.block_size:])
    pad = plain[-1]
    if pad < 1 or pad > AES.block_size or plain[-pad:] != bytes([pad]) * pad:
        raise ValueError("invalid PKCS7 padding")
    return json.loads(plain[:-pad].decode("utf-8"))


def normalize_message_event(event: dict) -> NormalizedMessage | None:
    """Schema 2.0 ``im.message.receive_v1`` event -> NormalizedMessage."""
    message = event.get("message") or {}
    sender = event.get("sender") or {}
    sender_id = sender.get("sender_id") or {}
    content_raw = message.get("content") or "{}"
    try:
        content = json.loads(content_raw)
    except (json.JSONDecodeError, TypeError):
        content = {}
    text = content.get("text", "") if isinstance(content, dict) else ""
    mentions = []
    for m in message.get("mentions") or []:
        mid = (m or {}).get("id") or {}
        open_id = mid.get("open_id")
        if open_id:
            mentions.append(open_id)
    message_id = message.get("message_id") or ""
    if not message_id or not message.get("chat_id"):
        return None
    return NormalizedMessage(
        message_id=message_id,
        root_id=message.get("root_id"),
        chat_id=message.get("chat_id"),
        chat_type=message.get("chat_type") or "",
        sender_type=sender.get("sender_type") or "",
        sender_id={
            "union_id": sender_id.get("union_id") or "",
            "user_id": sender_id.get("user_id") or "",
            "open_id": sender_id.get("open_id") or "",
        },
        text=text,
        mentions=mentions,
        msg_type=message.get("message_type") or "",
    )


def handle_routed_message(
    state: FrameState, msg: NormalizedMessage, decision: RouteDecision
) -> int:
    """Sync the sender + enqueue the run. Returns the HTTP status for Lark.

    200 on success (and after a successful mark-seen); 500 on a
    platform failure so Lark retries the event.
    """
    cfg = state.config
    open_id = (msg.sender_id or {}).get("open_id") or ""
    union_id = (msg.sender_id or {}).get("union_id") or ""
    try:
        synced = platform_io.sync_user(
            cfg.platform_base_url,
            cfg.platform_api_key,
            open_id=open_id,
            union_id=union_id,
        )
        user_id = synced.get("user_id")
        platform_io.enqueue(
            cfg.platform_base_url,
            cfg.platform_api_key,
            mission_id=cfg.mission_id,
            input_data={
                "message": msg.text,
                "chat_id": msg.chat_id,
                "chat_type": msg.chat_type,
                "sender_id": msg.sender_id,
                "message_id": msg.message_id,
                "root_id": msg.root_id or "",
            },
            session_id=decision.session_id or session_id_for(msg.chat_id, open_id),
            user_id=user_id,
        )
    except platform_io.PlatformError as exc:
        state.log_event(
            event_id=msg.message_id, decision="platform_error", error=str(exc)[:200]
        )
        if state.rate_limiter.allow_platform_down_notice(open_id):
            platform_io.send_lark_text(
                cfg.app_id, cfg.app_secret, chat_id=msg.chat_id, text=PLATFORM_DOWN_TEXT
            )
        return 500
    except Exception as exc:  # noqa: BLE001 — the frame must never die here
        # A bug in the platform callers (not a platform failure) still
        # means "not enqueued" — 500 so Lark retries, instead of a
        # dropped connection.
        state.log_event(
            event_id=msg.message_id, decision="error", error=str(exc)[:200]
        )
        return 500
    state.seen.mark(msg.message_id)
    state.log_event(
        event_id=msg.message_id,
        decision="routed",
        chat_id=msg.chat_id,
        user_id=user_id,
        session_id=decision.session_id,
    )
    return 200


def handle_message_event(state: FrameState, event: dict) -> int:
    """One message event -> HTTP status. Never raises."""
    msg = normalize_message_event(event)
    if msg is None:
        state.log_event(decision="drop", reason="bad_event")
        return 200
    if state.seen.contains(msg.message_id):
        state.log_event(event_id=msg.message_id, decision="drop", reason="duplicate")
        return 200
    decision = route_message(
        msg,
        bot_open_id=state.bot_open_id,
        allowlist=state.config.allowlist,
        rate_limiter=state.rate_limiter,
    )
    if decision.action == "route":
        return handle_routed_message(state, msg, decision)
    if decision.action == "refuse":
        platform_io.send_lark_text(
            state.config.app_id,
            state.config.app_secret,
            chat_id=msg.chat_id,
            text=REFUSAL_TEXT,
        )
        state.log_event(event_id=msg.message_id, decision="refuse", reason=decision.reason)
        return 200
    if decision.action == "rate_limited":
        open_id = (msg.sender_id or {}).get("open_id") or ""
        if state.rate_limiter.allow_rate_notice(open_id):
            platform_io.send_lark_text(
                state.config.app_id,
                state.config.app_secret,
                chat_id=msg.chat_id,
                text=RATE_LIMIT_TEXT,
            )
        state.log_event(event_id=msg.message_id, decision="rate_limited")
        return 200
    state.log_event(
        event_id=msg.message_id, decision="drop", reason=decision.reason
    )
    return 200


def handle_card_action(state: FrameState, payload: dict) -> None:
    """One card action -> enqueue the resolve run (background thread).

    The HTTP response (a toast) is already on its way — this runs
    after it, so a platform failure here cannot ask Lark to retry;
    the user gets a 'system busy' message in the chat instead.

    The operator goes through the same routing as a message
    (allowlist + 10/min rate limit — a button press is not a free
    pass into the platform). The chat id comes from the event
    context (``context.open_chat_id`` — Lark puts the card's own chat
    there, in both transports), cross-checked against the ``chat_id``
    in the button value: a mismatch is refused, because the value is
    attacker-controlled text on a card anyone in the group can copy.
    When the context carries no chat id (older payloads), the value
    is trusted as a fallback — the allowlist on the operator is then
    the real gate — and the fallback is logged.
    """
    cfg = state.config
    action = payload.get("action") or {}
    value_raw = action.get("value") or ""
    try:
        value = json.loads(value_raw) if isinstance(value_raw, str) else value_raw
    except json.JSONDecodeError:
        logger.warning("Card action value is not JSON: %r", value_raw[:200])
        return
    if not isinstance(value, dict):
        logger.warning("Card action value is not an object: %r", value_raw[:200])
        return
    action_name = value.get("action") or ""
    value_chat_id = value.get("chat_id") or ""
    incident_ref = value.get("incident_ref") or ""
    if not action_name or not value_chat_id:
        logger.warning("Card action missing action/chat_id: %r", value_raw[:200])
        return
    context = payload.get("context") or {}
    open_message_id = context.get("open_message_id") or ""
    context_chat_id = context.get("open_chat_id") or ""
    if context_chat_id and context_chat_id != value_chat_id:
        logger.warning(
            "Card action chat_id mismatch: context=%r value=%r",
            context_chat_id,
            value_chat_id,
        )
        return
    chat_id = context_chat_id or value_chat_id
    if not context_chat_id:
        logger.warning(
            "Card action without context open_chat_id — trusting "
            "value.chat_id; allowlist on the operator is the gate"
        )
    operator = payload.get("operator") or {}
    operator_id = operator.get("operator_id") or {}
    open_id = operator_id.get("open_id") or ""
    union_id = operator_id.get("union_id") or ""
    if not open_id:
        logger.warning("Card action has no operator open_id")
        return
    # Same gate as a message: allowlist + rate limit. A button press
    # on the bot's own card addresses the bot, so the @mention check
    # is skipped (via_card_action=True).
    card_msg = NormalizedMessage(
        message_id=open_message_id,
        root_id=None,
        chat_id=chat_id,
        chat_type="group",
        sender_type="user",
        sender_id=operator_id,
        text=f"[card action {action_name} for {incident_ref}]",
        mentions=[],
        msg_type="text",
    )
    decision = route_message(
        card_msg,
        bot_open_id=state.bot_open_id,
        allowlist=cfg.allowlist,
        rate_limiter=state.rate_limiter,
        via_card_action=True,
    )
    if decision.action == "refuse":
        platform_io.send_lark_text(
            cfg.app_id, cfg.app_secret, chat_id=chat_id, text=REFUSAL_TEXT
        )
        state.log_event(
            decision="card_refuse", action=action_name, chat_id=chat_id
        )
        return
    if decision.action == "rate_limited":
        if state.rate_limiter.allow_rate_notice(open_id):
            platform_io.send_lark_text(
                cfg.app_id, cfg.app_secret, chat_id=chat_id, text=RATE_LIMIT_TEXT
            )
        state.log_event(
            decision="card_rate_limited", action=action_name, chat_id=chat_id
        )
        return
    if decision.action != "route":
        state.log_event(
            decision="card_drop",
            action=action_name,
            chat_id=chat_id,
            reason=decision.reason,
        )
        return
    try:
        synced = platform_io.sync_user(
            cfg.platform_base_url,
            cfg.platform_api_key,
            open_id=open_id,
            union_id=union_id,
        )
        user_id = synced.get("user_id")
        if action_name == "resolve":
            text = (
                f"Vụ việc {incident_ref} đã được xử lý xong. "
                "Làm retrospective từ history rồi đổi tên group."
            )
        else:
            text = f"Hành động {action_name} cho vụ việc {incident_ref}."
        platform_io.enqueue(
            cfg.platform_base_url,
            cfg.platform_api_key,
            mission_id=cfg.mission_id,
            input_data={
                "message": text,
                "chat_id": chat_id,
                "chat_type": "group",
                "sender_id": operator_id,
                "message_id": open_message_id,
                "root_id": "",
                "action": action_name,
                "incident_ref": incident_ref,
            },
            session_id=decision.session_id or session_id_for(chat_id, open_id),
            user_id=user_id,
        )
        state.log_event(
            decision="card_routed", action=action_name, chat_id=chat_id,
            incident_ref=incident_ref,
        )
    except platform_io.PlatformError as exc:
        state.log_event(
            decision="card_platform_error", action=action_name,
            chat_id=chat_id, error=str(exc)[:200],
        )
        if state.rate_limiter.allow_platform_down_notice(open_id):
            platform_io.send_lark_text(
                cfg.app_id, cfg.app_secret, chat_id=chat_id, text=PLATFORM_DOWN_TEXT
            )


def dispatch_card_action(state: FrameState, payload: dict) -> str:
    """Dedupe + dispatch one card action. Returns "accepted" or "duplicate".

    Shared by the webhook handler (which answers the HTTP request
    first) and the WS channel handler. The heavy work
    (``handle_card_action``) runs in a background thread — the card UI
    waits on the response, so it cannot wait for the platform.
    """
    context = payload.get("context") or {}
    action = payload.get("action") or {}
    value_raw = action.get("value") or ""
    dedupe_key = f"card:{context.get('open_message_id') or ''}:{value_raw}"
    if state.seen.contains(dedupe_key):
        state.log_event(decision="drop", reason="duplicate_card")
        return "duplicate"
    state.seen.mark(dedupe_key)
    threading.Thread(
        target=handle_card_action, args=(state, payload), daemon=True
    ).start()
    return "accepted"


def _serve_health(handler, state: FrameState) -> bool:
    """Serve /healthz and /readyz. Returns True when the path was handled.

    Shared by the webhook handler and the health-only server (WS mode),
    so both transports answer health the same way.
    """
    path = urlparse(handler.path).path
    if path == "/healthz":
        uptime = int(time.time() - state.started_at)
        handler._send_json(
            200,
            {
                "status": "ok",
                "transport": state.config.transport,
                "uptime_seconds": uptime,
            },
        )
        return True
    if path == "/readyz":
        ok = platform_io.platform_health(state.config.platform_base_url)
        handler._send_json(
            200 if ok else 503,
            {"status": "ok" if ok else "platform_unreachable"},
        )
        return True
    return False


def make_handler(state: FrameState):
    """Build the request handler bound to one FrameState."""

    class LarkFrameHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: A002 - stdlib name
            logger.debug("lark frame http: %s", fmt % args)

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> tuple[dict | None, bytes]:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None, b""
            if length > MAX_CONTENT_LENGTH:
                logger.warning(
                    "lark frame http: body too large (%s bytes)", length
                )
                return None, b""
            raw = self.rfile.read(length) if length > 0 else b""
            if not raw:
                return None, raw
            try:
                return json.loads(raw.decode("utf-8")), raw
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None, raw

        def do_GET(self):  # noqa: N802 - stdlib name
            if not _serve_health(self, state):
                self._send_json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802 - stdlib name
            path = urlparse(self.path).path
            if path not in ("/lark/events", "/lark/card"):
                self._send_json(404, {"error": "not found"})
                return
            body, raw = self._read_json()
            if body is None:
                self._send_json(400, {"error": "invalid json"})
                return
            if not verify_signature(self.headers, raw, state.config.encrypt_key):
                state.log_event(decision="drop", reason="bad_signature", path=path)
                self._send_json(401, {"error": "bad signature"})
                return
            try:
                body = decrypt_payload(body, state.config.encrypt_key)
            except ValueError as exc:
                state.log_event(decision="drop", reason="decrypt_failed", path=path)
                self._send_json(400, {"error": str(exc)})
                return
            if body.get("type") == "url_verification":
                if not _token_matches(
                    body.get("token") or "", state.config.verification_token
                ):
                    self._send_json(401, {"error": "bad verification token"})
                    return
                # The challenge must be returned verbatim, within 1s.
                self._send_json(200, {"challenge": body.get("challenge")})
                return
            # Token + app_id are checked on EVERY event, not only on
            # url_verification (they used to be skipped here). The
            # event header carries both; a card callback carries the
            # token at the top level and no app_id at all — for cards
            # the signature + token are the authentication.
            header = body.get("header") or {}
            if body.get("type") == "event_callback":
                if not _token_matches(
                    header.get("token") or "", state.config.verification_token
                ):
                    state.log_event(decision="drop", reason="bad_token", path=path)
                    self._send_json(401, {"error": "bad verification token"})
                    return
                app_id = header.get("app_id") or ""
                if not app_id or app_id != state.config.app_id:
                    state.log_event(decision="drop", reason="bad_app_id", path=path)
                    self._send_json(401, {"error": "bad app_id"})
                    return
            elif body.get("type") == "card.action.trigger":
                if not _token_matches(
                    body.get("token") or "", state.config.verification_token
                ):
                    state.log_event(decision="drop", reason="bad_token", path=path)
                    self._send_json(401, {"error": "bad verification token"})
                    return
            else:
                self._send_json(200, {})
                return
            if path == "/lark/events":
                if body.get("type") != "event_callback":
                    self._send_json(200, {})
                    return
                if header.get("event_type") != "im.message.receive_v1":
                    state.log_event(
                        decision="drop",
                        reason="ignored_event",
                        event_type=header.get("event_type"),
                    )
                    self._send_json(200, {})
                    return
                status = handle_message_event(state, body.get("event") or {})
                self._send_json(status, {})
                return
            # /lark/card
            if body.get("type") != "card.action.trigger":
                self._send_json(200, {})
                return
            # Respond first (the card UI waits on it); heavy work runs
            # in a background thread — see dispatch_card_action.
            result = dispatch_card_action(state, body)
            toast = (
                "Đã xử lý rồi."
                if result == "duplicate"
                else "Đã nhận, bot đang xử lý."
            )
            self._send_json(
                200, {"toast": {"type": "info", "content": toast}}
            )

    return LarkFrameHandler


def make_health_handler(state: FrameState):
    """Handler serving only /healthz + /readyz (WS mode, D-LARK-18).

    No POST routes exist here on purpose: in websocket mode no Request
    URL is registered, so nothing would ever call /lark/events.
    """

    class LarkHealthHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: A002 - stdlib name
            logger.debug("lark frame health: %s", fmt % args)

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - stdlib name
            if not _serve_health(self, state):
                self._send_json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802 - stdlib name
            self._send_json(404, {"error": "not found"})

    return LarkHealthHandler


def start_health_server(
    state: FrameState, host: str | None = None, port: int | None = None
) -> ThreadingHTTPServer:
    """Start the health-only server on a daemon thread; returns the server.

    Used by the WS transport, whose channel opens no HTTP server of its
    own — without this, liveness is visible only via the process and
    the reconnect logs. A bind failure raises (fail fast at startup)
    so a supervisor restarts with a clear error instead of running a
    bot with no health endpoint.
    """
    host = host if host is not None else state.config.host
    port = port if port is not None else state.config.health_port
    server = ThreadingHTTPServer((host, port), make_health_handler(state))
    thread = threading.Thread(
        target=server.serve_forever, daemon=True, name="lark-health"
    )
    thread.start()
    logger.info(
        "Lark health server listening on %s:%s (transport=%s)",
        host,
        server.server_address[1],
        state.config.transport,
    )
    return server


def run_webhook(state: FrameState) -> None:
    """Serve the webhook frame forever (ThreadingHTTPServer).

    Webhook mode refuses to start without an encrypt key and a
    verification token — without them the signature/token checks
    have nothing to verify against and the frame would be
    fail-open (P0-1). WS mode is unaffected (no HTTP event routes).
    """
    if state.config.transport == "webhook" and (
        not state.config.encrypt_key or not state.config.verification_token
    ):
        raise SystemExit(
            "LARK_ENCRYPT_KEY / LARK_VERIFICATION_TOKEN are required in "
            "webhook mode — without them event verification is fail-open"
        )
    state.ensure_bot_open_id()
    server = ThreadingHTTPServer(
        (state.config.host, state.config.port), make_handler(state)
    )
    logger.info(
        "Lark webhook frame listening on %s:%s (transport=%s)",
        state.config.host, state.config.port, state.config.transport,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
