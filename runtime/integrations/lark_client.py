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
from datetime import datetime, timedelta, timezone

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


# -- Lark Tasks (Task API v2, lark-oapi 1.9.0) ---------------------------
# Read-only tools for the task-reminder flow (design doc §3.6):
# tasklists -> tasks, "in-process" = status todo (+ due filter, done
# by the skill flow, not here). All three calls pass
# user_id_type="open_id" so Member.id is an open_id — the id the
# reminder DM needs (lark_send_message receive_id_type="open_id").
# Verified SDK shapes (lark_oapi/api/task/v2/model/):
# - Task: guid, summary, description, due (Due), reminders, creator
#   (Member), members (List[Member]), completed_at (int, ms),
#   tasklists (List[TaskInTasklistInfo]), status ("todo"/"done"),
#   url, created_at/updated_at (int, ms), ...
# - Member: ONLY {id, type, role, name} — there is NO open_id /
#   user_id / union_id field; which id Member.id holds is decided by
#   the user_id_type query param.
# - Due: {timestamp (int, ms since epoch), is_all_day (bool)}.
# - TaskInTasklistInfo: {tasklist_guid, section_guid}.
# - ListTaskRequest params: page_size, page_token, completed (bool),
#   type, user_id_type. List responses carry items / page_token /
#   has_more; GetTaskResponseBody carries a single task.
# Due serialization: BOTH forms are returned — "due" is the UTC ISO
# 8601 string (human/LLM-readable, e.g. "2026-11-28T02:00:00+00:00")
# and "due_ms" is the raw millisecond timestamp (exact, for the
# skill's overdue/today comparison). Either may be None (no due date).

_MAX_TASKS = 100  # same cap spirit as list_messages (D-LARK-20)


def _serialize_member(member) -> dict:
    """One Member -> {id, type, role, name} (id is an open_id here)."""
    if member is None:
        return {"id": None, "type": None, "role": None, "name": None}
    return {
        "id": getattr(member, "id", None),
        "type": getattr(member, "type", None),
        "role": getattr(member, "role", None),
        "name": getattr(member, "name", None),
    }


def _serialize_due(due) -> tuple[str | None, int | None, bool | None]:
    """Due -> (ISO-8601 UTC string, ms timestamp, is_all_day)."""
    if due is None:
        return None, None, None
    ts_ms = getattr(due, "timestamp", None)
    iso = None
    if isinstance(ts_ms, int):
        iso = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
    return iso, ts_ms, getattr(due, "is_all_day", None)


def _serialize_task(task) -> dict:
    """One Task -> the small JSON dict the reminder flow needs.

    Description / attachments / custom fields are deliberately left
    out — a reminder needs summary + due + url + members, and a long
    description would eat the run's context budget.
    """
    due_iso, due_ms, due_all_day = _serialize_due(getattr(task, "due", None))
    return {
        "guid": getattr(task, "guid", None),
        "summary": getattr(task, "summary", None),
        "due": due_iso,
        "due_ms": due_ms,
        "due_is_all_day": due_all_day,
        "status": getattr(task, "status", None),
        "completed_at": getattr(task, "completed_at", None),
        "members": [_serialize_member(m) for m in (getattr(task, "members", None) or [])],
        "creator": _serialize_member(getattr(task, "creator", None)),
        "url": getattr(task, "url", None),
        "tasklists": [
            {
                "tasklist_guid": getattr(t, "tasklist_guid", None),
                "section_guid": getattr(t, "section_guid", None),
            }
            for t in (getattr(task, "tasklists", None) or [])
        ],
        "created_at": getattr(task, "created_at", None),
        "updated_at": getattr(task, "updated_at", None),
    }


def _serialize_tasklist(tasklist) -> dict:
    """One Tasklist -> {guid, name, creator, owner, members, url, ...}."""
    return {
        "guid": getattr(tasklist, "guid", None),
        "name": getattr(tasklist, "name", None),
        "creator": _serialize_member(getattr(tasklist, "creator", None)),
        "owner": _serialize_member(getattr(tasklist, "owner", None)),
        "members": [
            _serialize_member(m) for m in (getattr(tasklist, "members", None) or [])
        ],
        "url": getattr(tasklist, "url", None),
        "created_at": getattr(tasklist, "created_at", None),
        "updated_at": getattr(tasklist, "updated_at", None),
    }


def _task_model():
    """The task/v2 SDK model module, or a RuntimeError to surface."""
    try:
        client = get_client()
        from lark_oapi.api.task.v2 import model
    except RuntimeError as exc:
        raise exc
    return client, model


def list_tasklists(
    *,
    page_size: int = 50,
    page_token: str | None = None,
    limit: int = _MAX_TASKS,
) -> dict:
    """GET /open-apis/task/v2/tasklists — tasklists the app can read.

    Follows ``page_token`` until ``limit`` tasklists are collected
    (capped at 100, same spirit as ``list_messages``) or the pages
    end. Returns ``{"tasklists": [...], "count": n,
    "truncated": bool}``, or ``{"error": ...}`` (e.g. Lark 99991679
    when the app lacks the ``task:tasklist:read`` scope).
    """
    capped = max(1, min(int(limit), _MAX_TASKS))
    try:
        client, model = _task_model()
    except RuntimeError as exc:
        return {"error": str(exc)}
    tasklists: list[dict] = []
    token = page_token
    has_more = False
    while len(tasklists) < capped:
        builder = (
            model.ListTasklistRequest.builder()
            .page_size(min(50, capped - len(tasklists)))
            .user_id_type("open_id")
        )
        if token:
            builder.page_token(token)
        try:
            resp = client.task.v2.tasklist.list(builder.build())
        except Exception as exc:  # noqa: BLE001 — provider failure is data
            logger.warning("Lark list_tasklists raised: %s", exc)
            return {"error": f"lark list_tasklists raised: {exc}"[:500]}
        err = _check(resp, what="list_tasklists")
        if err is not None:
            return err
        data = resp.data
        for item in getattr(data, "items", None) or []:
            if len(tasklists) >= capped:
                break
            tasklists.append(_serialize_tasklist(item))
        has_more = bool(getattr(data, "has_more", False))
        if not has_more:
            break
        token = getattr(data, "page_token", None)
        if not token:
            break
    return {"tasklists": tasklists, "count": len(tasklists), "truncated": has_more}


def list_tasks(
    *,
    completed: bool = False,
    page_size: int = 50,
    page_token: str | None = None,
    limit: int = _MAX_TASKS,
) -> dict:
    """GET /open-apis/task/v2/tasks — tasks, paginated to a cap.

    ``completed=False`` (the default) returns only unfinished tasks —
    the reminder flow's "in-process" set is ``status == "todo"`` among
    these, with the due filter applied by the skill flow. Follows
    ``page_token`` until ``limit`` tasks are collected (capped at 100,
    same spirit as ``list_messages``) or the pages end. Returns
    ``{"tasks": [...], "count": n, "truncated": bool,
    "completed": bool}``, or ``{"error": ...}`` (e.g. Lark 99991679
    when the app lacks the ``task:task:read`` scope).
    """
    capped = max(1, min(int(limit), _MAX_TASKS))
    try:
        client, model = _task_model()
    except RuntimeError as exc:
        return {"error": str(exc)}
    tasks: list[dict] = []
    token = page_token
    has_more = False
    while len(tasks) < capped:
        builder = (
            model.ListTaskRequest.builder()
            .page_size(min(50, capped - len(tasks)))
            .completed(bool(completed))
            .user_id_type("open_id")
        )
        if token:
            builder.page_token(token)
        try:
            resp = client.task.v2.task.list(builder.build())
        except Exception as exc:  # noqa: BLE001 — provider failure is data
            logger.warning("Lark list_tasks raised: %s", exc)
            return {"error": f"lark list_tasks raised: {exc}"[:500]}
        err = _check(resp, what="list_tasks")
        if err is not None:
            return err
        data = resp.data
        for item in getattr(data, "items", None) or []:
            if len(tasks) >= capped:
                break
            tasks.append(_serialize_task(item))
        has_more = bool(getattr(data, "has_more", False))
        if not has_more:
            break
        token = getattr(data, "page_token", None)
        if not token:
            break
    return {
        "tasks": tasks,
        "count": len(tasks),
        "truncated": has_more,
        "completed": bool(completed),
    }


def get_task(*, task_guid: str) -> dict:
    """GET /open-apis/task/v2/tasks/:task_guid — one task by guid.

    Returns ``{"task": {...}}`` on success, or ``{"error": ...}``.
    """
    if not task_guid or not task_guid.strip():
        return {"error": "task_guid must not be empty"}
    try:
        client, model = _task_model()
    except RuntimeError as exc:
        return {"error": str(exc)}
    req = (
        model.GetTaskRequest.builder()
        .task_guid(task_guid.strip())
        .user_id_type("open_id")
        .build()
    )
    try:
        resp = client.task.v2.task.get(req)
    except Exception as exc:  # noqa: BLE001 — provider failure is data
        logger.warning("Lark get_task raised: %s", exc)
        return {"error": f"lark get_task raised: {exc}"[:500]}
    err = _check(resp, what="get_task")
    if err is not None:
        return err
    task = getattr(resp.data, "task", None)
    if task is None:
        return {"error": f"lark get_task returned no task for guid {task_guid.strip()}"}
    return {"task": _serialize_task(task)}


_MAX_FREEBUSY_RANGE = timedelta(days=90)
_MISSING_FREEBUSY_SCOPE = 99991672


def _calendar_model():
    """The calendar/v4 SDK model module, or a RuntimeError to surface."""
    try:
        client = get_client()
        from lark_oapi.api.calendar.v4 import model
    except RuntimeError as exc:
        raise exc
    return client, model


def _parse_rfc3339(value: str) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def get_freebusy(
    *,
    user_open_id: str,
    time_min: str,
    time_max: str,
) -> dict:
    """POST /open-apis/calendar/v4/freebusy/list — busy intervals only.

    ``user_id_type`` is ``open_id``. Body carries ``time_min``,
    ``time_max`` (RFC3339) and ``user_id``. Returns
    ``{"busy_intervals": [{"start_time", "end_time"}, ...], "count": n}``
    on success, or ``{"error": ...}``. The requested span must not exceed
    90 days (Lark API limit). Error 99991672 is turned into an operator
    hint to enable ``calendar:calendar.free_busy:read`` or
    ``calendar:calendar:readonly``.
    """
    if not user_open_id or not str(user_open_id).strip():
        return {"error": "user_open_id must not be empty"}
    if not time_min or not time_max:
        return {"error": "time_min and time_max are required (RFC3339)"}
    start = _parse_rfc3339(time_min)
    end = _parse_rfc3339(time_max)
    if start is None or end is None:
        return {"error": "time_min and time_max must be valid RFC3339 datetimes"}
    if end <= start:
        return {"error": "time_max must be after time_min"}
    if end - start > _MAX_FREEBUSY_RANGE:
        return {"error": "freebusy query range must not exceed 90 days"}
    try:
        client, model = _calendar_model()
    except RuntimeError as exc:
        return {"error": str(exc)}
    req = (
        model.ListFreebusyRequest.builder()
        .user_id_type("open_id")
        .request_body(
            model.ListFreebusyRequestBody.builder()
            .time_min(time_min.strip())
            .time_max(time_max.strip())
            .user_id(str(user_open_id).strip())
            .build()
        )
        .build()
    )
    try:
        resp = client.calendar.v4.freebusy.list(req)
    except Exception as exc:  # noqa: BLE001 — provider failure is data
        logger.warning("Lark get_freebusy raised: %s", exc)
        return {"error": f"lark get_freebusy raised: {exc}"[:500]}
    code = getattr(resp, "code", None)
    if code == _MISSING_FREEBUSY_SCOPE:
        msg = str(getattr(resp, "msg", "") or "")[:200]
        logger.warning("Lark get_freebusy missing scope: code=%s msg=%s", code, msg)
        return {
            "error": (
                "lark get_freebusy failed (code 99991672): missing calendar scope — "
                "enable calendar:calendar.free_busy:read or calendar:calendar:readonly "
                "on the Lark Custom App"
            )
        }
    err = _check(resp, what="get_freebusy")
    if err is not None:
        return err
    data = resp.data
    intervals: list[dict] = []
    for item in getattr(data, "freebusy_list", None) or []:
        intervals.append(
            {
                "start_time": getattr(item, "start_time", None),
                "end_time": getattr(item, "end_time", None),
            }
        )
    return {"busy_intervals": intervals, "count": len(intervals)}
