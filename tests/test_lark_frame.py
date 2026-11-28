"""Lark frame — routing rules, signature/crypto, and the webhook server.

Routing is transport-agnostic (shared by webhook + WS). The server
tests run a real ThreadingHTTPServer on an ephemeral port with the
platform stubbed — no credentials, no network.
"""

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from services.lark_frame.__main__ import load_config, load_env_file
from services.lark_frame.routing import (
    REFUSAL_TEXT,
    NormalizedMessage,
    RateLimiter,
    SeenSet,
    route_message,
    session_id_for,
)
from services.lark_frame.webhook import (
    FrameConfig,
    FrameState,
    decrypt_payload,
    make_handler,
    start_health_server,
    verify_signature,
)

BOT = "ou_bot"
ALLOWLIST = {"ou_1"}


def _msg(**over):
    base = {
        "message_id": "om_1",
        "root_id": "",
        "chat_id": "oc_dm",
        "chat_type": "p2p",
        "sender_type": "user",
        "sender_id": {"union_id": "on_1", "user_id": "u_1", "open_id": "ou_1"},
        "text": "hello",
        "mentions": [],
        "msg_type": "text",
    }
    base.update(over)
    return NormalizedMessage(**base)


@pytest.fixture
def limiter():
    return RateLimiter()


def test_session_id_is_deterministic_and_within_pattern():
    a = session_id_for("oc_dm", "ou_1")
    b = session_id_for("oc_dm", "ou_1")
    assert a == b and a.startswith("lark-") and len(a) == 21
    assert session_id_for("oc_dm", "ou_2") != a
    assert session_id_for("oc_other", "ou_1") != a


def test_dm_from_allowlisted_user_routes(limiter):
    d = route_message(_msg(), bot_open_id=BOT, allowlist=ALLOWLIST, rate_limiter=limiter)
    assert d.action == "route"
    assert d.session_id == session_id_for("oc_dm", "ou_1")


def test_bot_own_message_dropped(limiter):
    d = route_message(
        _msg(sender_type="app"), bot_open_id=BOT, allowlist=ALLOWLIST,
        rate_limiter=limiter,
    )
    assert d.action == "drop" and d.reason == "own_message"


def test_non_text_dropped(limiter):
    d = route_message(
        _msg(msg_type="image"), bot_open_id=BOT, allowlist=ALLOWLIST,
        rate_limiter=limiter,
    )
    assert d.action == "drop" and d.reason == "not_text"


def test_outsider_refused_once_per_hour(limiter):
    msg = _msg(sender_id={"open_id": "ou_out"})
    d1 = route_message(msg, bot_open_id=BOT, allowlist=ALLOWLIST, rate_limiter=limiter)
    assert d1.action == "refuse" and d1.reason == "not_allowlisted"
    d2 = route_message(msg, bot_open_id=BOT, allowlist=ALLOWLIST, rate_limiter=limiter)
    assert d2.action == "drop" and d2.reason == "refusal_rate_limited"


def test_group_requires_bot_mention(limiter):
    msg = _msg(chat_id="oc_grp", chat_type="group", mentions=[])
    d = route_message(msg, bot_open_id=BOT, allowlist=ALLOWLIST, rate_limiter=limiter)
    assert d.action == "drop" and d.reason == "not_mentioned"
    msg = _msg(chat_id="oc_grp", chat_type="group", mentions=[BOT])
    d = route_message(msg, bot_open_id=BOT, allowlist=ALLOWLIST, rate_limiter=limiter)
    assert d.action == "route"


def test_rate_limit_after_ten_per_minute(limiter):
    for _ in range(10):
        d = route_message(
            _msg(), bot_open_id=BOT, allowlist=ALLOWLIST, rate_limiter=limiter
        )
        assert d.action == "route"
    d = route_message(
        _msg(), bot_open_id=BOT, allowlist=ALLOWLIST, rate_limiter=limiter
    )
    assert d.action == "rate_limited" and d.reason == "rate_limited"
    d = route_message(
        _msg(), bot_open_id=BOT, allowlist=ALLOWLIST, rate_limiter=limiter
    )
    assert d.action == "rate_limited" and d.reason == "rate_limited"


def test_seen_set_memory_mode():
    seen = SeenSet(redis_url=None)
    assert not seen.contains("k1")
    seen.mark("k1")
    assert seen.contains("k1")


def test_verify_signature_matches_sdk_algorithm():
    encrypt_key = "ek_test"
    body = b'{"type":"event_callback"}'
    ts, nonce = "1700000000", "abc"
    sig = hashlib.sha256(
        (ts + nonce + encrypt_key).encode("utf-8") + body
    ).hexdigest()
    headers = {
        "X-Lark-Request-Timestamp": ts,
        "X-Lark-Request-Nonce": nonce,
        "X-Lark-Signature": sig,
    }
    assert verify_signature(headers, body, encrypt_key) is True
    assert verify_signature(headers, body, "wrong") is False
    # No encrypt key configured -> signature not enforced (unencrypted mode).
    assert verify_signature({}, body, "") is True


def test_decrypt_payload_round_trip():
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad

    encrypt_key = "ek_test"
    plaintext = {"type": "event_callback", "token": "vt"}
    raw_plain = json.dumps(plaintext).encode()
    key = hashlib.sha256(encrypt_key.encode()).digest()
    iv = get_random_bytes(16)
    ciphertext = iv + AES.new(key, AES.MODE_CBC, iv).encrypt(pad(raw_plain, 16))
    import base64

    encrypted = base64.b64encode(ciphertext).decode()
    assert decrypt_payload({"encrypt": encrypted}, encrypt_key) == plaintext
    # Unencrypted body passes through unchanged.
    assert decrypt_payload(plaintext, encrypt_key) == plaintext
    with pytest.raises(ValueError):
        decrypt_payload({"encrypt": "!!!not-base64!!!"}, encrypt_key)


# --- Webhook server end-to-end -------------------------------------------

@pytest.fixture
def server(monkeypatch):
    def fake_sync_user(*a, **k):
        return {"user_id": "u-1", "email": "lark-ou_1@lark.local", "created": True}

    monkeypatch.setattr("services.lark_frame.platform.sync_user", fake_sync_user)
    monkeypatch.setattr("services.lark_frame.platform.enqueue", lambda *a, **k: None)
    monkeypatch.setattr("services.lark_frame.platform.send_lark_text", lambda *a, **k: None)
    state = FrameState(
        FrameConfig(
            app_id="cli_smoke",
            app_secret="s",
            verification_token="vt",
            platform_base_url="http://127.0.0.1:9",
            platform_api_key="k",
            allowlist={"ou_1"},
            bot_open_id=BOT,
            host="127.0.0.1",
            port=0,
        )
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    yield port, state
    httpd.shutdown()


def _req(port, method, path, body=None, headers=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _event(message_id="om_1", open_id="ou_1", text="ping", mentions=None, chat_type="p2p"):
    return {
        "type": "event_callback",
        "header": {
            "event_id": "e1",
            "event_type": "im.message.receive_v1",
            "app_id": "cli_smoke",
        },
        "event": {
            "sender": {
                "sender_type": "user",
                "sender_id": {"open_id": open_id, "user_id": "u_1", "union_id": "on_1"},
            },
            "message": {
                "message_id": message_id,
                "chat_id": "oc_dm",
                "chat_type": chat_type,
                "message_type": "text",
                "content": json.dumps({"text": text}),
                "root_id": "",
                "mentions": mentions or [],
            },
        },
    }


def test_healthz(server):
    port, _ = server
    status, body = _req(port, "GET", "/healthz")
    assert status == 200 and body["status"] == "ok" and body["transport"] == "webhook"


# --- Health-only server (WS mode, D-LARK-18) -----------------------------

@pytest.fixture
def health_server():
    state = FrameState(
        FrameConfig(
            app_id="cli_smoke",
            app_secret="s",
            platform_base_url="http://127.0.0.1:9",
            allowlist={"ou_1"},
            bot_open_id=BOT,
            host="127.0.0.1",
            health_port=0,
            transport="websocket",
        )
    )
    httpd = start_health_server(state, "127.0.0.1", 0)
    port = httpd.server_address[1]
    yield port, state
    httpd.shutdown()


def test_health_server_healthz_reports_websocket(health_server):
    port, _ = health_server
    status, body = _req(port, "GET", "/healthz")
    assert status == 200
    assert body["status"] == "ok" and body["transport"] == "websocket"
    assert isinstance(body["uptime_seconds"], int) and body["uptime_seconds"] >= 0


def test_health_server_readyz_platform_down(health_server):
    port, _ = health_server
    status, body = _req(port, "GET", "/readyz")
    assert status == 503 and body["status"] == "platform_unreachable"


def test_health_server_has_no_event_routes(health_server):
    port, _ = health_server
    status, _ = _req(port, "POST", "/lark/events", body={"type": "event_callback"})
    assert status == 404
    status, _ = _req(port, "GET", "/lark/events")
    assert status == 404
    status, _ = _req(port, "GET", "/nope")
    assert status == 404


# --- .env loading ---------------------------------------------------------

def test_load_env_file_loads_values_then_load_config_reads_them(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LARK_APP_ID=cli_from_dotenv\n"
        "LARK_APP_SECRET=s_from_dotenv\n"
        "LARK_TRANSPORT=websocket\n"
    )
    for name in ("LARK_APP_ID", "LARK_APP_SECRET", "LARK_TRANSPORT"):
        monkeypatch.delenv(name, raising=False)
    assert load_env_file(env_file) == env_file
    assert os.environ["LARK_APP_ID"] == "cli_from_dotenv"
    config = load_config()
    assert config.app_id == "cli_from_dotenv"
    assert config.app_secret == "s_from_dotenv"
    assert config.transport == "websocket"


def test_load_env_file_shell_env_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LARK_APP_ID=cli_from_dotenv\nLARK_APP_SECRET=s_from_dotenv\n")
    monkeypatch.setenv("LARK_APP_ID", "cli_from_shell")
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)
    load_env_file(env_file)
    assert os.environ["LARK_APP_ID"] == "cli_from_shell"
    assert os.environ["LARK_APP_SECRET"] == "s_from_dotenv"


def test_load_env_file_missing_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    assert load_env_file(tmp_path / ".env") is None
    assert "LARK_APP_ID" not in os.environ


def test_url_verification(server):
    port, _ = server
    status, body = _req(
        port, "POST", "/lark/events",
        {"type": "url_verification", "token": "vt", "challenge": "c123"},
    )
    assert status == 200 and body == {"challenge": "c123"}
    status, _ = _req(
        port, "POST", "/lark/events",
        {"type": "url_verification", "token": "wrong", "challenge": "c123"},
    )
    assert status == 401


def test_message_routes_and_dedupe(server):
    port, state = server
    status, _ = _req(port, "POST", "/lark/events", _event())
    assert status == 200
    # Same message_id again -> duplicate, still 200, no second enqueue.
    status, _ = _req(port, "POST", "/lark/events", _event())
    assert status == 200


def test_card_action_toast_and_duplicate(server):
    port, _ = server
    card_value = {"action": "resolve", "chat_id": "oc_g", "incident_ref": "om_a"}
    card = {
        "type": "card.action.trigger",
        "action": {"value": json.dumps(card_value)},
        "context": {"open_message_id": "om_card1"},
        "operator": {"operator_id": {"open_id": "ou_1", "user_id": "u_1"}},
    }
    status, body = _req(port, "POST", "/lark/card", card)
    assert status == 200 and body["toast"]["content"] == "Đã nhận, bot đang xử lý."
    status, body = _req(port, "POST", "/lark/card", card)
    assert status == 200 and body["toast"]["content"] == "Đã xử lý rồi."


def test_platform_down_returns_500(server, monkeypatch):
    from services.lark_frame.platform import PlatformError

    port, _ = server
    monkeypatch.setattr(
        "services.lark_frame.platform.enqueue",
        lambda *a, **k: (_ for _ in ()).throw(PlatformError("platform down")),
    )
    status, _ = _req(port, "POST", "/lark/events", _event(message_id="om_down"))
    assert status == 500


def test_unexpected_platform_error_also_returns_500(server, monkeypatch):
    """A bug in the platform callers must not kill the connection."""
    port, _ = server
    monkeypatch.setattr(
        "services.lark_frame.platform.enqueue",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stray bug")),
    )
    status, _ = _req(port, "POST", "/lark/events", _event(message_id="om_bug"))
    assert status == 500


def test_outsider_gets_one_refusal(server, monkeypatch):
    port, _ = server
    sent: list = []
    monkeypatch.setattr(
        "services.lark_frame.platform.send_lark_text",
        lambda *a, **k: sent.append(k.get("text")),
    )
    status, _ = _req(
        port, "POST", "/lark/events", _event(message_id="om_o1", open_id="ou_out")
    )
    assert status == 200
    assert sent == [REFUSAL_TEXT]
    # Same outsider, new message: refusal is rate-limited to 1/hour.
    status, _ = _req(
        port, "POST", "/lark/events", _event(message_id="om_o2", open_id="ou_out")
    )
    assert status == 200
    assert sent == [REFUSAL_TEXT]


def test_send_lark_text_puts_receive_id_type_in_query(monkeypatch):
    """receive_id_type is a query parameter on im/v1/messages, not a body
    field — Lark returned 99992402 "receive_id_type is required" live on
    2026-11-26 when it was sent in the body."""
    import services.lark_frame.platform as lark_platform

    captured: dict = {}

    def fake_post(url, payload, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return 200, {"code": 0, "data": {"message_id": "om_x"}}

    monkeypatch.setattr(lark_platform, "_tenant_access_token", lambda *a: "t-tenant")
    monkeypatch.setattr(lark_platform, "_post_json", fake_post)
    assert lark_platform.send_lark_text("cli_x", "sec", chat_id="oc_1", text="hi") is True
    assert captured["url"] == (
        "https://open.larksuite.com/open-apis/im/v1/messages?receive_id_type=chat_id"
    )
    assert captured["payload"]["receive_id"] == "oc_1"
    assert "receive_id_type" not in captured["payload"]
    assert json.loads(captured["payload"]["content"]) == {"text": "hi"}
