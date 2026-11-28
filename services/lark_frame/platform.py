"""Outbound HTTP for the Lark frame (agent_lark_me, Step 2).

Stdlib ``urllib`` only — the webhook frame must not need the Lark SDK.
Two directions:

- Platform (the agent's brain): ``POST /lark/users/sync`` + ``POST
  /executions`` with ``X-API-Key``. Every call has a 5-second timeout
  (D-LARK-19); a timeout or HTTP error raises :class:`PlatformError`
  and the caller returns 500 to Lark so the event is retried.
- Lark (the frame's own sends — refusal / rate-limit notice /
  platform-down notice): the frame fetches its own tenant token
  (``POST /open-apis/auth/v3/tenant_access_token/internal``) and
  posts to ``/open-apis/im/v1/messages``. The token is cached until
  expiry. These sends never raise — a failed send is only logged.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

LARK_DOMAIN = "https://open.larksuite.com"
_TIMEOUT_SECONDS = 5.0

_tenant_token: dict = {"token": None, "expires_at": 0.0}


class PlatformError(Exception):
    """The platform call failed (timeout / HTTP error / bad body)."""


def _post_json(
    url: str, payload: dict, headers: dict, timeout: float
) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 — body is not JSON
            body = {"detail": exc.reason}
        raise PlatformError(f"{url} -> HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PlatformError(f"{url} -> {exc}") from exc


def _get_json(url: str, headers: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 — body is not JSON
            body = {"detail": exc.reason}
        raise PlatformError(f"{url} -> HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PlatformError(f"{url} -> {exc}") from exc


def sync_user(
    base_url: str,
    api_key: str,
    *,
    open_id: str,
    union_id: str = "",
    timeout: float = _TIMEOUT_SECONDS,
) -> dict:
    """POST /lark/users/sync — resolve a Lark sender to a platform user.

    ``union_id`` is the tenant-stable identity (same human, any Lark
    app of the tenant); it is sent when the event carries it, so the
    platform can keep one user across an app switch. Empty means the
    event had none — sync is then exactly the old open_id-only flow.
    """
    payload: dict = {"provider": "lark", "external_id": open_id}
    if union_id:
        payload["union_id"] = union_id
    status, body = _post_json(
        f"{base_url.rstrip('/')}/lark/users/sync",
        payload,
        {"X-API-Key": api_key},
        timeout,
    )
    if status != 200:
        raise PlatformError(f"/lark/users/sync -> HTTP {status}: {body}")
    return body


def enqueue(
    base_url: str,
    api_key: str,
    *,
    mission_id: str,
    input_data: dict,
    session_id: str,
    user_id: str | None,
    timeout: float = _TIMEOUT_SECONDS,
) -> dict:
    """POST /executions — enqueue one run. Returns the execution record."""
    payload: dict = {
        "mission_id": mission_id,
        "input": input_data,
        "session_id": session_id,
    }
    if user_id:
        payload["user_id"] = user_id
    status, body = _post_json(
        f"{base_url.rstrip('/')}/executions",
        payload,
        {"X-API-Key": api_key},
        timeout,
    )
    if status not in (200, 202):
        raise PlatformError(f"/executions -> HTTP {status}: {body}")
    return body


def platform_health(base_url: str, timeout: float = 2.0) -> bool:
    """GET /health on the platform (for the frame's /readyz)."""
    try:
        status, _ = _get_json(f"{base_url.rstrip('/')}/health", {}, timeout)
        return status == 200
    except PlatformError:
        return False


def _tenant_access_token(app_id: str, app_secret: str) -> str | None:
    """The frame's own tenant token, cached until expiry (never logged)."""
    if _tenant_token["token"] and _tenant_token["expires_at"] > time.time() + 60:
        return _tenant_token["token"]
    try:
        _, body = _post_json(
            f"{LARK_DOMAIN}/open-apis/auth/v3/tenant_access_token/internal",
            {"app_id": app_id, "app_secret": app_secret},
            {},
            _TIMEOUT_SECONDS,
        )
    except PlatformError as exc:
        logger.warning("Lark tenant token fetch failed: %s", exc)
        return None
    if body.get("code") != 0:
        logger.warning(
            "Lark tenant token failed: code=%s", body.get("code")
        )
        return None
    token = body.get("tenant_access_token")
    if not token:
        return None
    _tenant_token["token"] = token
    _tenant_token["expires_at"] = time.time() + int(body.get("expire") or 7200)
    return token


def send_lark_text(
    app_id: str, app_secret: str, *, chat_id: str, text: str
) -> bool:
    """Send a short text message via the Lark API (frame's own sends).

    Best-effort: returns False on any failure, never raises. Used for
    the refusal, the rate-limit notice and the platform-down notice —
    the only messages the frame ever sends itself (D-LARK-09).
    """
    token = _tenant_access_token(app_id, app_secret)
    if token is None:
        return False
    try:
        status, body = _post_json(
            # receive_id_type is a QUERY parameter on im/v1/messages, not a
            # body field — Lark returns 99992402 "receive_id_type is
            # required" when it is sent in the body (seen live 2026-11-26).
            f"{LARK_DOMAIN}/open-apis/im/v1/messages?receive_id_type=chat_id",
            {
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            },
            {"Authorization": f"Bearer {token}"},
            _TIMEOUT_SECONDS,
        )
    except PlatformError as exc:
        logger.warning("Lark send failed: %s", exc)
        return False
    if status != 200 or body.get("code") != 0:
        logger.warning(
            "Lark send failed: HTTP %s code=%s", status, body.get("code")
        )
        return False
    return True


def fetch_bot_open_id(app_id: str, app_secret: str) -> str | None:
    """The bot's own open_id, via GET /open-apis/bot/v3/info.

    Used for @bot detection in groups (the mention carries the bot's
    open_id). None on any failure — the caller falls back to loose
    mode (any mention routes) and logs a warning.
    """
    token = _tenant_access_token(app_id, app_secret)
    if token is None:
        return None
    try:
        status, body = _get_json(
            f"{LARK_DOMAIN}/open-apis/bot/v3/info",
            {"Authorization": f"Bearer {token}"},
            _TIMEOUT_SECONDS,
        )
    except PlatformError as exc:
        logger.warning("Lark bot info fetch failed: %s", exc)
        return None
    if status != 200 or body.get("code") != 0:
        logger.warning(
            "Lark bot info failed: HTTP %s code=%s", status, body.get("code")
        )
        return None
    data = body.get("data") or {}
    bot = data.get("bot") or data
    open_id = bot.get("open_id") or bot.get("openid") or ""
    return open_id or None
