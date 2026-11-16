"""Idempotent write-tool calls within a session."""

from __future__ import annotations

import hashlib
import json
from typing import Any

_CACHE: dict[str, dict] = {}


def write_idempotency_key(
    session_id: str,
    tool_name: str,
    args: dict,
    *,
    execution_id: str = "",
) -> str:
    """Stable key for one logical write action (session + tool + args)."""
    _ = execution_id  # retained for logging; key is stable across retries
    payload = {
        "session_id": session_id or "",
        "tool": tool_name,
        "args": args,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def cached_write_result(key: str) -> dict | None:
    row = _CACHE.get(key)
    if isinstance(row, dict) and row.get("result") is not None:
        return dict(row["result"])
    return None


def store_write_result(key: str, result: dict) -> None:
    _CACHE[key] = {"result": dict(result)}


def invoke_idempotent_write(
    tool,
    args: dict,
    *,
    session_id: str,
    execution_id: str = "",
) -> dict:
    """Call a write tool once per idempotency key; return cached success."""
    name = getattr(tool, "name", "write")
    key = write_idempotency_key(session_id, name, args, execution_id=execution_id)
    cached = cached_write_result(key)
    if cached is not None and not cached.get("error"):
        return {**cached, "idempotent_replay": True}
    result = tool.invoke(args)
    if isinstance(result, dict) and not result.get("error"):
        store_write_result(key, result)
    return result if isinstance(result, dict) else {"result": result}


def reset_idempotency_cache_for_tests() -> None:
    _CACHE.clear()