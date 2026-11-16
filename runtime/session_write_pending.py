"""Session-scoped pending writes (quote -> confirm -> execute)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MEMORY: dict[tuple[str, str], dict] = {}


def _key(session_id: str, pipeline_id: str) -> tuple[str, str]:
    return (session_id or "", pipeline_id or "")


def get_pending(session_id: str, pipeline_id: str) -> dict | None:
    if not session_id or not pipeline_id:
        return None
    mem = _MEMORY.get(_key(session_id, pipeline_id))
    if mem is not None:
        return dict(mem)
    try:
        from app.core.db import SessionLocal
        from app.models import SessionWritePending

        db = SessionLocal()
        try:
            row = (
                db.query(SessionWritePending)
                .filter(
                    SessionWritePending.session_id == session_id,
                    SessionWritePending.pipeline_id == pipeline_id,
                )
                .first()
            )
            if row is None:
                return None
            return {
                "status": row.status,
                "payload": dict(row.payload or {}),
                "idempotency_key": row.idempotency_key,
            }
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 — memory fallback
        logger.debug("Pending DB read failed (%s); using memory", exc)
        return _MEMORY.get(_key(session_id, pipeline_id))


def save_pending(
    session_id: str,
    pipeline_id: str,
    *,
    status: str,
    payload: dict,
    idempotency_key: str | None = None,
) -> None:
    record = {
        "status": status,
        "payload": dict(payload or {}),
        "idempotency_key": idempotency_key,
    }
    _MEMORY[_key(session_id, pipeline_id)] = record
    if not session_id or not pipeline_id:
        return
    try:
        from app.core.db import SessionLocal
        from app.models import SessionWritePending

        db = SessionLocal()
        try:
            row = (
                db.query(SessionWritePending)
                .filter(
                    SessionWritePending.session_id == session_id,
                    SessionWritePending.pipeline_id == pipeline_id,
                )
                .first()
            )
            if row is None:
                row = SessionWritePending(
                    session_id=session_id,
                    pipeline_id=pipeline_id,
                    status=status,
                    payload=payload,
                    idempotency_key=idempotency_key,
                )
                db.add(row)
            else:
                row.status = status
                row.payload = dict(payload or {})
                row.idempotency_key = idempotency_key
            db.commit()
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Pending DB write failed (%s); memory only", exc)


def merge_tool_result(
    session_id: str,
    pipeline_id: str,
    tool_name: str,
    result: Any,
    *,
    initial_status: str = "quoted",
) -> None:
    if not session_id or not pipeline_id:
        return
    if not isinstance(result, dict) or result.get("error"):
        return
    pending = get_pending(session_id, pipeline_id) or {
        "status": initial_status,
        "payload": {"tools": {}},
        "idempotency_key": None,
    }
    tools = pending.setdefault("payload", {}).setdefault("tools", {})
    tools[tool_name] = result
    save_pending(
        session_id,
        pipeline_id,
        status=pending.get("status") or initial_status,
        payload=pending["payload"],
        idempotency_key=pending.get("idempotency_key"),
    )


def reset_pending_store_for_tests() -> None:
    _MEMORY.clear()