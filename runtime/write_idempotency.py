"""Idempotent write-tool calls — durable via DB (and in-process cache)."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_CACHE: dict[str, dict] = {}


def write_idempotency_key(
    *,
    write_attempt_id: str,
    execution_id: str,
    write_tool: str,
    action: str = "execute",
) -> str:
    payload = {
        "write_attempt_id": write_attempt_id or "",
        "execution_id": execution_id or "",
        "tool": write_tool,
        "action": action,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _load_from_db(key: str) -> dict | None:
    try:
        from app.core.db import SessionLocal
        from app.models import WriteIdempotencyResult

        db = SessionLocal()
        try:
            row = db.get(WriteIdempotencyResult, key)
            if row is None:
                return None
            data = row.result
            return dict(data) if isinstance(data, dict) else None
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("idempotency DB read failed for %s: %s", key, exc)
        return None


def _store_db(key: str, result: dict) -> None:
    try:
        from app.core.db import SessionLocal
        from app.models import WriteIdempotencyResult

        db = SessionLocal()
        try:
            row = db.get(WriteIdempotencyResult, key)
            if row is None:
                db.add(WriteIdempotencyResult(idempotency_key=key, result=dict(result)))
            else:
                row.result = dict(result)
            db.commit()
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("idempotency DB write failed for %s: %s", key, exc)


def cached_write_result(key: str) -> dict | None:
    row = _CACHE.get(key)
    if isinstance(row, dict) and row.get("result") is not None:
        return dict(row["result"])
    loaded = _load_from_db(key)
    if loaded is not None and not loaded.get("error"):
        _CACHE[key] = {"result": loaded}
        return loaded
    return None


def store_write_result(key: str, result: dict) -> None:
    _CACHE[key] = {"result": dict(result)}
    if not result.get("error"):
        _store_db(key, result)


def invoke_idempotent_write(
    tool,
    args: dict,
    *,
    idempotency_key: str,
) -> dict:
    cached = cached_write_result(idempotency_key)
    if cached is not None and not cached.get("error"):
        return {**cached, "idempotent_replay": True}
    result = tool.invoke(args)
    if isinstance(result, dict) and not result.get("error"):
        store_write_result(idempotency_key, result)
    return result if isinstance(result, dict) else {"result": result}


def reset_idempotency_cache_for_tests() -> None:
    _CACHE.clear()