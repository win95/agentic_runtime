"""Lark client for the worker side (agent_lark_me).

Wraps the official ``lark-oapi`` SDK (``lark.Client``). The App ID /
Secret come from the platform ``.env`` (``LARK_APP_ID`` /
``LARK_APP_SECRET`` — the *agent's* credentials, never Muse's vault);
the SDK fetches + caches + refreshes the tenant access token itself.

Two rules the whole file follows (same as the Tavily tools):
- Lark returns HTTP 200 even on API errors — every call checks the
  body ``code`` and turns a non-zero code into a structured error dict,
  never an exception.
- No secret, token or message content is logged at INFO (or below);
  failures log only the Lark error code + message.

This module is the worker-side client only. The webhook frame
(``services/lark_frame/``) uses stdlib HTTP + urllib and does not
import this module.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

logger = logging.getLogger(__name__)

LARK_DOMAIN = "https://open.larksuite.com"

# One shared client per process — the SDK caches the tenant token
# inside it, so rebuilding per call would re-fetch the token each time.
_client = None


def _build_client():
    """Build the SDK client, or raise a clear error when unusable."""
    app_id = os.environ.get("LARK_APP_ID", "").strip()
    app_secret = os.environ.get("LARK_APP_SECRET", "").strip()
    if not app_id or not app_secret:
        raise RuntimeError(
            "LARK_APP_ID / LARK_APP_SECRET are not set — add the agent's "
            "own Custom App credentials to .env (Step 1 of the lark_me plan); "
            "Lark tools stay unavailable until then"
        )
    try:
        import lark_oapi as lark
    except ImportError as exc:  # pragma: no cover - missing optional dep
        raise RuntimeError(
            "lark-oapi is not installed — run `pip install -r requirements.txt`"
        ) from exc
    return (
        lark.Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .domain(LARK_DOMAIN)
        .build()
    )


def get_client():
    """The shared SDK client (built lazily, once per process)."""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def _check(resp, *, what: str) -> dict | None:
    """None when the Lark call succeeded, else a structured error dict.

    Lark answers HTTP 200 even when the call failed, so the body
    ``code`` is the only signal. ``resp`` is the SDK response object
    (``.code``, ``.msg``, ``.data``).
    """
    code = getattr(resp, "code", None)
    if code == 0:
        return None
    msg = str(getattr(resp, "msg", "") or "")[:500]
    logger.warning("Lark %s failed: code=%s msg=%s", what, code, msg)
    return {"error": f"lark {what} failed (code {code}): {msg}"}


def _content_text(msg_type: str, text: str) -> str:
    """The ``content`` JSON string for a text / card message."""
    if msg_type == "text":
        return json.dumps({"text": text}, ensure_ascii=False)
    # Card content arrives as a ready-made JSON string; pass it through.
    return text


def send_message(
    *,
    receive_id: str,
    receive_id_type: str = "chat_id",
    msg_type: str = "text",
    text: str = "",
    uuid_str: str | None = None,
) -> dict:
    """POST /open-apis/im/v1/messages — send a message to a chat/user.

    Returns ``{"message_id", "chat_id", "root_id"}`` on success, or
    ``{"error": ...}``. ``text`` is the plain text for ``msg_type="text"``
    or the full card JSON string for ``msg_type="interactive"``.
    """
    if not receive_id or not receive_id.strip():
        return {"error": "receive_id must not be empty"}
    if receive_id_type not in ("chat_id", "user_id", "open_id"):
        return {"error": f"unknown receive_id_type: {receive_id_type}"}
    if msg_type not in ("text", "interactive"):
        return {"error": f"unsupported msg_type: {msg_type}"}
    if not text:
        return {"error": "text must not be empty"}
    try:
        client = get_client()
        from lark_oapi.api.im.v1 import model
    except RuntimeError as exc:
        return {"error": str(exc)}
    body = (
        model.CreateMessageRequestBody.builder()
        .receive_id(receive_id.strip())
        .msg_type(msg_type)
        .content(_content_text(msg_type, text))
        .uuid(uuid_str or uuid.uuid4().hex)
        .build()
    )
    req = (
        model.CreateMessageRequest.builder()
        .receive_id_type(receive_id_type)
        .request_body(body)
        .build()
    )
    try:
        resp = client.im.v1.message.create(req)
    except Exception as exc:  # noqa: BLE001 — provider failure is data
        logger.warning("Lark send_message raised: %s", exc)
        return {"error": f"lark send_message raised: {exc}"[:500]}
    err = _check(resp, what="send_message")
    if err is not None:
        return err
    data = resp.data
    return {
        "message_id": getattr(data, "message_id", None),
        "chat_id": getattr(data, "chat_id", None),
        "root_id": getattr(data, "root_id", None),
    }


def reply_message(
    *,
    message_id: str,
    msg_type: str = "text",
    text: str = "",
    reply_in_thread: bool = True,
    uuid_str: str | None = None,
) -> dict:
    """POST /open-apis/im/v1/messages/:message_id/reply.

    Returns ``{"message_id", "root_id"}`` on success, or
    ``{"error": ...}``.
    """
    if not message_id or not message_id.strip():
        return {"error": "message_id must not be empty"}
    if not text:
        return {"error": "text must not be empty"}
    if msg_type not in ("text", "interactive"):
        return {"error": f"unsupported msg_type: {msg_type}"}
    try:
        client = get_client()
        from lark_oapi.api.im.v1 import model
    except RuntimeError as exc:
        return {"error": str(exc)}
    body = (
        model.ReplyMessageRequestBody.builder()
        .msg_type(msg_type)
        .content(_content_text(msg_type, text))
        .reply_in_thread(reply_in_thread)
        .uuid(uuid_str or uuid.uuid4().hex)
        .build()
    )
    req = (
        model.ReplyMessageRequest.builder()
        .message_id(message_id.strip())
        .request_body(body)
        .build()
    )
    try:
        resp = client.im.v1.message.reply(req)
    except Exception as exc:  # noqa: BLE001 — provider failure is data
        logger.warning("Lark reply_message raised: %s", exc)
        return {"error": f"lark reply_message raised: {exc}"[:500]}
    err = _check(resp, what="reply_message")
    if err is not None:
        return err
    data = resp.data
    return {
        "message_id": getattr(data, "message_id", None),
        "root_id": getattr(data, "root_id", None),
    }


def create_group(
    *,
    name: str,
    description: str = "",
    user_id_list: list[str] | None = None,
) -> dict:
    """POST /open-apis/im/v1/chats — create a group chat.

    Returns ``{"chat_id", "name"}`` on success, or ``{"error": ...}``.
    """
    if not name or not name.strip():
        return {"error": "name must not be empty"}
    try:
        client = get_client()
        from lark_oapi.api.im.v1 import model
    except RuntimeError as exc:
        return {"error": str(exc)}
    body = (
        model.CreateChatRequestBody.builder()
        .name(name.strip())
        .description(description)
        .user_id_list(list(user_id_list or []))
        .build()
    )
    req = model.CreateChatRequest.builder().request_body(body).build()
    try:
        resp = client.im.v1.chat.create(req)
    except Exception as exc:  # noqa: BLE001 — provider failure is data
        logger.warning("Lark create_group raised: %s", exc)
        return {"error": f"lark create_group raised: {exc}"[:500]}
    err = _check(resp, what="create_group")
    if err is not None:
        return err
    data = resp.data
    return {"chat_id": getattr(data, "chat_id", None), "name": getattr(data, "name", None)}


def get_group(*, chat_id: str) -> dict:
    """GET /open-apis/im/v1/chats/:chat_id — read a group's info.

    Returns ``{"chat_id", "name", "description", "owner_id"}`` on
    success, or ``{"error": ...}``.
    """
    if not chat_id or not chat_id.strip():
        return {"error": "chat_id must not be empty"}
    try:
        client = get_client()
        from lark_oapi.api.im.v1 import model
    except RuntimeError as exc:
        return {"error": str(exc)}
    req = model.GetChatRequest.builder().chat_id(chat_id.strip()).build()
    try:
        resp = client.im.v1.chat.get(req)
    except Exception as exc:  # noqa: BLE001 — provider failure is data
        logger.warning("Lark get_group raised: %s", exc)
        return {"error": f"lark get_group raised: {exc}"[:500]}
    err = _check(resp, what="get_group")
    if err is not None:
        return err
    data = resp.data
    return {
        "chat_id": getattr(data, "chat_id", None),
        "name": getattr(data, "name", None),
        "description": getattr(data, "description", None),
        "owner_id": getattr(data, "owner_id", None),
    }


def update_group(*, chat_id: str, name: str | None = None, description: str | None = None) -> dict:
    """PUT /open-apis/im/v1/chats/:chat_id — rename / edit a group.

    Returns ``{"chat_id", "ok": True}`` on success, or ``{"error": ...}``.
    The update API returns no data body, so success is code 0 alone.
    """
    if not chat_id or not chat_id.strip():
        return {"error": "chat_id must not be empty"}
    if name is None and description is None:
        return {"error": "nothing to update: pass name or description"}
    try:
        client = get_client()
        from lark_oapi.api.im.v1 import model
    except RuntimeError as exc:
        return {"error": str(exc)}
    builder = model.UpdateChatRequestBody.builder()
    if name is not None:
        builder.name(name)
    if description is not None:
        builder.description(description)
    req = (
        model.UpdateChatRequest.builder()
        .chat_id(chat_id.strip())
        .request_body(builder.build())
        .build()
    )
    try:
        resp = client.im.v1.chat.update(req)
    except Exception as exc:  # noqa: BLE001 — provider failure is data
        logger.warning("Lark update_group raised: %s", exc)
        return {"error": f"lark update_group raised: {exc}"[:500]}
    err = _check(resp, what="update_group")
    if err is not None:
        return err
    return {"chat_id": chat_id.strip(), "ok": True}


def add_members(
    *,
    chat_id: str,
    member_ids: list[str],
    member_id_type: str = "open_id",
) -> dict:
    """POST /open-apis/im/v1/chats/:chat_id/members — add members.

    Returns ``{"chat_id", "invalid_id_list": [...]}`` on success, or
    ``{"error": ...}``. A partially-invalid list still succeeds — the
    invalid ids are reported, not hidden.
    """
    if not chat_id or not chat_id.strip():
        return {"error": "chat_id must not be empty"}
    ids = [m.strip() for m in (member_ids or []) if isinstance(m, str) and m.strip()]
    if not ids:
        return {"error": "member_ids must not be empty"}
    if member_id_type not in ("open_id", "user_id", "union_id"):
        return {"error": f"unknown member_id_type: {member_id_type}"}
    try:
        client = get_client()
        from lark_oapi.api.im.v1 import model
    except RuntimeError as exc:
        return {"error": str(exc)}
    body = model.CreateChatMembersRequestBody.builder().id_list(ids).build()
    req = (
        model.CreateChatMembersRequest.builder()
        .chat_id(chat_id.strip())
        .member_id_type(member_id_type)
        .request_body(body)
        .build()
    )
    try:
        resp = client.im.v1.chat_members.create(req)
    except Exception as exc:  # noqa: BLE001 — provider failure is data
        logger.warning("Lark add_members raised: %s", exc)
        return {"error": f"lark add_members raised: {exc}"[:500]}
    err = _check(resp, what="add_members")
    if err is not None:
        return err
    data = resp.data
    return {
        "chat_id": chat_id.strip(),
        "invalid_id_list": list(getattr(data, "invalid_id_list", None) or []),
    }


def upload_image(*, image_bytes: bytes, image_type: str = "message") -> dict:
    """POST /open-apis/im/v1/images — upload an image for a card.

    Returns ``{"image_key": ...}`` on success, or ``{"error": ...}``.
    """
    if not image_bytes:
        return {"error": "image_bytes must not be empty"}
    if image_type not in ("message", "avatar"):
        return {"error": f"unknown image_type: {image_type}"}
    try:
        client = get_client()
        from lark_oapi.api.im.v1 import model
    except RuntimeError as exc:
        return {"error": str(exc)}
    import io

    body = (
        model.CreateImageRequestBody.builder()
        .image_type(image_type)
        .image(io.BytesIO(image_bytes))
        .build()
    )
    req = model.CreateImageRequest.builder().request_body(body).build()
    try:
        resp = client.im.v1.image.create(req)
    except Exception as exc:  # noqa: BLE001 — provider failure is data
        logger.warning("Lark upload_image raised: %s", exc)
        return {"error": f"lark upload_image raised: {exc}"[:500]}
    err = _check(resp, what="upload_image")
    if err is not None:
        return err
    return {"image_key": getattr(resp.data, "image_key", None)}


def list_messages(
    *,
    chat_id: str,
    limit: int = 100,
    start_time: int | None = None,
    end_time: int | None = None,
    sort_type: str = "ByCreateTimeDesc",
) -> dict:
    """GET /open-apis/im/v1/messages — paginated history of a chat.

    Follows ``page_token`` until ``limit`` messages are collected or
    there are no more pages. ``limit`` is capped at 100 (D-LARK-20).
    Fetches newest-first (``ByCreateTimeDesc`` — the default) so a
    capped history is the NEWEST messages, then reverses the result
    so callers still see oldest-first order. Returns
    ``{"messages": [...], "count": n, "truncated": bool}`` where
    ``truncated`` is the API's own ``has_more`` at the point paging
    stopped (not "we collected exactly the cap"), and each message is
    ``{message_id, sender_id, create_time, message_type, content}``
    (text content only; other types are named, never invented), or
    ``{"error": ...}``.
    """
    if not chat_id or not chat_id.strip():
        return {"error": "chat_id must not be empty"}
    capped = max(1, min(int(limit), 100))
    try:
        client = get_client()
        from lark_oapi.api.im.v1 import model
    except RuntimeError as exc:
        return {"error": str(exc)}
    messages: list[dict] = []
    page_token: str | None = None
    has_more = False
    while len(messages) < capped:
        builder = (
            model.ListMessageRequest.builder()
            .container_id(chat_id.strip())
            .container_id_type("chat")
            .page_size(min(50, capped - len(messages)))
            .sort_type(sort_type)
        )
        if page_token:
            builder.page_token(page_token)
        if start_time is not None:
            builder.start_time(str(start_time))
        if end_time is not None:
            builder.end_time(str(end_time))
        try:
            resp = client.im.v1.message.list(builder.build())
        except Exception as exc:  # noqa: BLE001 — provider failure is data
            logger.warning("Lark list_messages raised: %s", exc)
            return {"error": f"lark list_messages raised: {exc}"[:500]}
        err = _check(resp, what="list_messages")
        if err is not None:
            return err
        data = resp.data
        for item in getattr(data, "items", None) or []:
            if len(messages) >= capped:
                break
            sender = getattr(item, "sender", None)
            sender_id = getattr(sender, "id", None) if sender is not None else None
            body = getattr(item, "body", None)
            content = getattr(body, "content", None) if body is not None else None
            messages.append(
                {
                    "message_id": getattr(item, "message_id", None),
                    "sender_id": sender_id,
                    "create_time": getattr(item, "create_time", None),
                    "message_type": getattr(item, "msg_type", None),
                    "content": content if isinstance(content, str) else None,
                }
            )
        has_more = bool(getattr(data, "has_more", False))
        if not has_more:
            break
        page_token = getattr(data, "page_token", None)
        if not page_token:
            break
    if sort_type == "ByCreateTimeDesc":
        # Fetched newest-first; callers (token guard, retrospective)
        # expect oldest-first.
        messages.reverse()
    return {"messages": messages, "count": len(messages), "truncated": has_more}
