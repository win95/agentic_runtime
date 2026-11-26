"""Lark tools (Layer 5) — for the agent_lark_me agent.

Native Lark messaging via the official ``lark-oapi`` SDK, wrapped by
``runtime/integrations/lark_client.py``. The client is imported lazily
and the App ID / Secret are read from the environment at call time, so
importing this module never fails when the SDK or the credentials are
missing — the agent gets a structured error dict instead, per the
codebase's tool-failure pattern.

Capability: ``lark_messaging``. Write tools are NOT read-only; the two
reads (``lark_get_group``, ``lark_get_history``) are. ``lark_update_group``
renames a group, so it carries ``destructive_target_args=("chat_id",)``
— the permission layer asks before it runs (Layer 2).
"""

from __future__ import annotations

from langchain_core.tools import tool

from runtime.integrations import lark_client
from runtime.tools.registry import registry

_MAX_TEXT_CHARS = 4000


def _err(name: str, result: dict) -> dict:
    """Attach the tool name + provenance to a client error dict."""
    out = dict(result)
    out.setdefault("tool", name)
    out.setdefault("stub", False)
    out.setdefault("provenance", f"lark:{name}")
    return out


@tool
def lark_create_group(
    name: str,
    description: str = "",
    user_id_list: list[str] | None = None,
) -> dict:
    """Create a Lark group chat for an incident alert.

    Returns {"chat_id", "name"} — the new group's id, needed to send
    the alert card into it and to rename it when resolved.
    """
    result = lark_client.create_group(
        name=name, description=description, user_id_list=user_id_list
    )
    if "error" in result:
        return _err("lark_create_group", result)
    return {**result, "stub": False, "provenance": "lark:create_group"}


@tool
def lark_send_message(
    receive_id: str,
    receive_id_type: str = "chat_id",
    msg_type: str = "text",
    text: str = "",
) -> dict:
    """Send a Lark message to a chat or user.

    ``receive_id_type`` is chat_id / user_id / open_id; ``msg_type`` is
    text or interactive (card — ``text`` is then the full card JSON
    string). Text is capped at 4000 chars so one call cannot flood the
    chat. Returns {"message_id", "chat_id", "root_id"}.
    """
    if len(text or "") > _MAX_TEXT_CHARS:
        return {
            "error": f"text is too long ({len(text)} chars, max {_MAX_TEXT_CHARS})",
            "tool": "lark_send_message",
            "stub": False,
            "provenance": "lark:send_message",
        }
    result = lark_client.send_message(
        receive_id=receive_id,
        receive_id_type=receive_id_type,
        msg_type=msg_type,
        text=text,
    )
    if "error" in result:
        return _err("lark_send_message", result)
    return {**result, "stub": False, "provenance": "lark:send_message"}


@tool
def lark_get_group(chat_id: str) -> dict:
    """Read a Lark group's info (name, description, owner).

    Read-only. Use before renaming a group to confirm it exists and
    is the group this run created.
    """
    result = lark_client.get_group(chat_id=chat_id)
    if "error" in result:
        return _err("lark_get_group", result)
    return {**result, "stub": False, "provenance": "lark:get_group"}


@tool
def lark_upload_image(image_base64: str, image_type: str = "message") -> dict:
    """Upload an image to Lark for use in a card.

    ``image_base64`` is the image bytes as base64. Returns
    {"image_key"} — the key to put in the card JSON. Only needed when
    a card carries an image; plain cards skip this tool.
    """
    import base64

    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception:
        return {
            "error": "image_base64 is not valid base64",
            "tool": "lark_upload_image",
            "stub": False,
            "provenance": "lark:upload_image",
        }
    result = lark_client.upload_image(image_bytes=raw, image_type=image_type)
    if "error" in result:
        return _err("lark_upload_image", result)
    return {**result, "stub": False, "provenance": "lark:upload_image"}


@tool
def lark_add_members(
    chat_id: str,
    member_ids: list[str],
    member_id_type: str = "open_id",
) -> dict:
    """Add members to a Lark group chat.

    ``member_ids`` are open_ids (default) / user_ids / union_ids — the
    roster in skill `agent-lark` maps a team name to open_ids. Returns
    {"chat_id", "invalid_id_list"}; a partially-invalid list still
    succeeds and reports the invalid ids.
    """
    result = lark_client.add_members(
        chat_id=chat_id, member_ids=member_ids, member_id_type=member_id_type
    )
    if "error" in result:
        return _err("lark_add_members", result)
    return {**result, "stub": False, "provenance": "lark:add_members"}


@tool
def lark_reply_message(
    message_id: str,
    msg_type: str = "text",
    text: str = "",
    reply_in_thread: bool = True,
) -> dict:
    """Reply to a Lark message, in its thread by default.

    Use for the ask-answer flow: reply to the exact message the user
    sent (``root_id`` threads the reply). Returns
    {"message_id", "root_id"}.
    """
    if len(text or "") > _MAX_TEXT_CHARS:
        return {
            "error": f"text is too long ({len(text)} chars, max {_MAX_TEXT_CHARS})",
            "tool": "lark_reply_message",
            "stub": False,
            "provenance": "lark:reply_message",
        }
    result = lark_client.reply_message(
        message_id=message_id,
        msg_type=msg_type,
        text=text,
        reply_in_thread=reply_in_thread,
    )
    if "error" in result:
        return _err("lark_reply_message", result)
    return {**result, "stub": False, "provenance": "lark:reply_message"}


@tool
def lark_get_history(
    chat_id: str,
    limit: int = 100,
    start_time: int | None = None,
    end_time: int | None = None,
) -> dict:
    """Read a Lark chat's message history (paginated, capped).

    Read-only. ``limit`` defaults to 100 and is capped at 100
    (D-LARK-20) — the client follows page_token until the limit is met
    or the history ends. Returns {"messages": [...], "count": n,
    "truncated": bool}; each message is {message_id, sender_id,
    create_time, message_type, content} (text only — other types are
    named, never invented). Use for the post-incident retrospective
    and for the first-sync backfill (limit=50, sender-filtered).

    Token guard (D-LARK-20): the tool keeps only the newest messages
    that fit ~4,000 tokens (oldest dropped first) and notes it —
    a 100-message history of long texts must not blow the run's
    context budget. The skill only summarizes what the tool returns.
    """
    result = lark_client.list_messages(
        chat_id=chat_id, limit=limit, start_time=start_time, end_time=end_time
    )
    if "error" in result:
        return _err("lark_get_history", result)
    messages = result.get("messages") or []
    kept, token_truncated = _truncate_to_token_budget(messages)
    out = {
        **result,
        "messages": kept,
        "count": len(kept),
        "truncated": bool(result.get("truncated")) or token_truncated,
        "stub": False,
        "provenance": "lark:get_history",
    }
    if token_truncated:
        out["note"] = (
            "history truncated to fit the ~4000-token tool budget "
            "(oldest messages dropped first)"
        )
    return out


_TOKEN_BUDGET = 4000


def _truncate_to_token_budget(messages: list[dict]) -> tuple[list[dict], bool]:
    """Keep the newest messages that fit ~4,000 tokens (oldest first).

    Messages arrive oldest-first (ByCreateTimeAsc). Token cost is a
    rough ~4 chars/token + 10/message for the fields. At least the
    newest message is always kept, even if it alone exceeds the budget.
    """
    kept: list[dict] = []
    used = 0
    for msg in reversed(messages):
        cost = len(msg.get("content") or "") // 4 + 10
        if kept and used + cost > _TOKEN_BUDGET:
            break
        kept.append(msg)
        used += cost
    kept.reverse()
    return kept, len(kept) < len(messages)


@tool
def lark_update_group(
    chat_id: str,
    name: str | None = None,
    description: str | None = None,
) -> dict:
    """Rename / edit a Lark group chat (marks an incident resolved).

    Returns {"chat_id", "ok": True}. The agent may only rename a group
    this run created — enforced by LarkAgent.scope_tool, not by this
    tool alone.
    """
    result = lark_client.update_group(
        chat_id=chat_id, name=name, description=description
    )
    if "error" in result:
        return _err("lark_update_group", result)
    return {**result, "stub": False, "provenance": "lark:update_group"}


# Register the 8 tools under the lark_messaging capability.
# Reads are read-only; writes are not. update_group carries its
# destructive target so the permission layer asks first (Layer 2).
registry.register(
    lark_create_group, capability="lark_messaging", permission="lark:write",
    timeout_seconds=60, read_only=False,
)
registry.register(
    lark_send_message, capability="lark_messaging", permission="lark:write",
    timeout_seconds=60, read_only=False,
)
registry.register(
    lark_get_group, capability="lark_messaging", permission="lark:read",
    timeout_seconds=30, read_only=True,
)
registry.register(
    lark_upload_image, capability="lark_messaging", permission="lark:write",
    timeout_seconds=60, read_only=False,
)
registry.register(
    lark_add_members, capability="lark_messaging", permission="lark:write",
    timeout_seconds=60, read_only=False,
)
registry.register(
    lark_reply_message, capability="lark_messaging", permission="lark:write",
    timeout_seconds=60, read_only=False,
)
registry.register(
    lark_get_history, capability="lark_messaging", permission="lark:read",
    timeout_seconds=60, read_only=True,
)
registry.register(
    lark_update_group, capability="lark_messaging", permission="lark:write",
    timeout_seconds=60, read_only=False,
    destructive_target_args=("chat_id",),
)
