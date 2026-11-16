"""Parse tool names and JSON bodies from LangChain message lists."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


def tool_call_id_index(messages: list) -> dict[str, str]:
    """Map tool_call_id -> tool name from AIMessage tool_calls."""
    out: dict[str, str] = {}
    for message in messages or []:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict):
                cid = call.get("id")
                name = call.get("name")
            else:
                cid = getattr(call, "id", None)
                name = getattr(call, "name", None)
            if isinstance(cid, str) and isinstance(name, str) and cid and name:
                out[cid] = name
    return out


def tool_message_name(message: ToolMessage, id_index: dict[str, str]) -> str | None:
    direct = getattr(message, "name", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    cid = getattr(message, "tool_call_id", None)
    if isinstance(cid, str) and cid in id_index:
        return id_index[cid]
    return None


def tool_content_dict(message: Any, *, id_index: dict[str, str] | None = None) -> dict | None:
    if not isinstance(message, ToolMessage):
        return None
    content = getattr(message, "content", "")
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip().startswith("{"):
                    try:
                        parsed = json.loads(text)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(parsed, dict):
                        return parsed
        return None
    if isinstance(content, str) and content.strip().startswith("{"):
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def collect_tool_results(messages: list) -> list[tuple[str, dict]]:
    """Successful tool invocations as (tool_name, payload dict)."""
    id_index = tool_call_id_index(messages)
    out: list[tuple[str, dict]] = []
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "status", None) == "error":
            continue
        name = tool_message_name(message, id_index)
        if not name:
            continue
        data = tool_content_dict(message, id_index=id_index)
        if not isinstance(data, dict) or data.get("error"):
            continue
        out.append((name, data))
    return out