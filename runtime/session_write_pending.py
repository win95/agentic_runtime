"""Session-scoped pending writes (quote -> confirm -> execute)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_MEMORY: dict[tuple[str, str], dict] = {}
_MEMORY_MAX = 1024


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
            payload = dict(row.payload or {})
            return {
                "status": row.status,
                "payload": payload,
                "idempotency_key": row.idempotency_key,
                "write_attempt_id": payload.get("write_attempt_id"),
            }
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 — memory fallback
        logger.warning("Pending DB read failed (%s); using memory", exc)
        return _MEMORY.get(_key(session_id, pipeline_id))


def save_pending(
    session_id: str,
    pipeline_id: str,
    *,
    status: str,
    payload: dict,
    idempotency_key: str | None = None,
) -> None:
    payload = dict(payload or {})
    record = {
        "status": status,
        "payload": payload,
        "idempotency_key": idempotency_key,
        "write_attempt_id": payload.get("write_attempt_id"),
    }
    key = _key(session_id, pipeline_id)
    if len(_MEMORY) >= _MEMORY_MAX and key not in _MEMORY:
        oldest = next(iter(_MEMORY))
        _MEMORY.pop(oldest, None)
    _MEMORY[key] = record
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
        logger.warning("Pending DB write failed (%s); memory only", exc)


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
    pending = get_pending(session_id, pipeline_id)
    if pending is None or pending.get("status") == "written":
        write_attempt_id = uuid.uuid4().hex[:16]
        pending = {
            "status": initial_status,
            "payload": {"tools": {}, "write_attempt_id": write_attempt_id},
            "idempotency_key": None,
            "write_attempt_id": write_attempt_id,
        }
    else:
        payload = pending.setdefault("payload", {})
        if not payload.get("write_attempt_id"):
            write_attempt_id = uuid.uuid4().hex[:16]
            payload["write_attempt_id"] = write_attempt_id
            pending["write_attempt_id"] = write_attempt_id
    tools = pending.setdefault("payload", {}).setdefault("tools", {})
    tools[tool_name] = result
    new_status = pending.get("status") or initial_status
    if new_status == "written":
        new_status = initial_status
    save_pending(
        session_id,
        pipeline_id,
        status=new_status,
        payload=pending["payload"],
        idempotency_key=pending.get("idempotency_key"),
    )


def reset_pending_store_for_tests() -> None:
    _MEMORY.clear()