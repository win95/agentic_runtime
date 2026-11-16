"""Idempotent write-tool calls — one key per logical write attempt + execution."""

from __future__ import annotations

import hashlib
import json
from typing import Any

_CACHE: dict[str, dict] = {}


def write_idempotency_key(
    *,
    write_attempt_id: str,
    execution_id: str,
    write_tool: str,
    action: str = "execute",
) -> str:
    """Stable key for one write attempt in one execution (Celery retries replay).

    Distinct intentional writes use a new ``write_attempt_id`` (per pipeline
  quote cycle), so identical room args in the same session are not blocked.
    """
    payload = {
        "write_attempt_id": write_attempt_id or "",
        "execution_id": execution_id or "",
        "tool": write_tool,
        "action": action,
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
    idempotency_key: str,
) -> dict:
    """Call a write tool once per idempotency key; return cached success."""
    cached = cached_write_result(idempotency_key)
    if cached is not None and not cached.get("error"):
        return {**cached, "idempotent_replay": True}
    result = tool.invoke(args)
    if isinstance(result, dict) and not result.get("error"):
        store_write_result(idempotency_key, result)
    return result if isinstance(result, dict) else {"result": result}


def reset_idempotency_cache_for_tests() -> None:
    _CACHE.clear()