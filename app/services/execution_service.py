"""Execution service — create an Execution and dispatch it to Celery.

The API only validates, persists the Execution record and enqueues the
execution id. It never runs the agent itself (HTTP lifecycle != agent
lifecycle). The worker (app/worker/tasks.py) does the actual run.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app import models
from app.services import mission_service
from runtime.security import (
    is_valid_session_id,
    sanitize_input_value,
    validate_execution_input_data,
)


def new_execution_id() -> str:
    return f"exec_{uuid.uuid4().hex[:12]}"


def validate_execution_input(mission: models.Mission, input_data: dict) -> list[str]:
    """Validate one execution input (Layer 1 of the prompt-injection
    defense). Returns a list of problems (empty = valid): missing
    required fields, oversized input, overlong string fields, an
    ``asset_id`` outside the allowlist, or malformed/out-of-order dates.
    """
    mission_dict = {
        "input_schema": mission.input_schema or {},
    }
    return validate_execution_input_data(mission_dict, input_data)


def dispatch_execution(execution_id: str) -> None:
    """Enqueue only the execution id. Imported lazily so the API process
    does not need a live broker at import time. Tests monkeypatch this
    function to assert the dispatch without a broker."""
    from app.worker.tasks import run_execution

    run_execution.delay(execution_id)


def create_execution(
    db: Session, mission_id: str, input_data: dict, session_id: str | None = None
) -> tuple[models.Execution | None, list[str]]:
    """Create a QUEUED execution. Returns (execution, problems).

    If the mission does not exist, returns (None, ["__mission_not_found__"]).
    If the input fails Layer 1 validation, returns (None, problems).
    Control characters are stripped from the stored input before it is
    persisted or handed to the agent.

    Sessions (Option B, docs/sessions.md): ``session_id`` is optional
    and client-supplied — omitted means a fresh ``sess_`` id is
    generated, so the run gets its own one-turn session and behaves
    exactly as before. A supplied id must match the session allowlist,
    and the session must not already belong to a different agent or
    mission (the first execution in a session binds both).
    """
    mission = mission_service.resolve_mission(db, mission_id)
    if mission is None:
        return None, ["__mission_not_found__"]
    input_data = sanitize_input_value(input_data)
    problems = validate_execution_input(mission, input_data)
    if problems:
        return None, problems
    if session_id is not None and not is_valid_session_id(session_id):
        return None, ["field 'session_id' must match ^[A-Za-z0-9_-]{1,64}$"]
    if session_id is None:
        from app.services import session_service

        session_id = session_service.new_session_id()
    else:
        from app.services import session_service

        binding_problem = session_service.check_session_binding(
            db, session_id, mission.id, mission.agent_id
        )
        if binding_problem is not None:
            return None, [binding_problem]

    execution = models.Execution(
        id=new_execution_id(),
        mission_id=mission.id,
        mission_version=mission.version,
        agent_id=mission.agent_id,
        session_id=session_id,
        input=input_data,
        status="QUEUED",
    )
    db.add(execution)
    db.commit()
    db.refresh(execution)

    try:
        dispatch_execution(execution.id)
    except Exception as exc:  # broker down etc. — do not leave a stuck QUEUED row
        execution.status = "FAILED"
        execution.error = f"dispatch failed: {exc}"
        execution.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        raise
    return execution, []


def get_execution(db: Session, execution_id: str) -> models.Execution | None:
    return db.get(models.Execution, execution_id)


def list_executions(
    db: Session, limit: int = 50, session_id: str | None = None
) -> list[models.Execution]:
    """Executions, newest first. ``session_id`` narrows to one session
    (a filter, not an identifier — an unknown id just matches nothing).
    """
    query = db.query(models.Execution)
    if session_id is not None:
        query = query.filter(models.Execution.session_id == session_id)
    return query.order_by(models.Execution.created_at.desc()).limit(limit).all()


def cancel_execution(db: Session, execution_id: str) -> models.Execution | None:
    """Mark QUEUED/RUNNING as CANCELLED. A running worker checks this flag
    before starting and after finishing, and will not overwrite the result
    of a cancelled execution."""
    execution = db.get(models.Execution, execution_id)
    if execution is None:
        return None
    if execution.status in ("QUEUED", "RUNNING"):
        execution.status = "CANCELLED"
        execution.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        db.refresh(execution)
    return execution


def get_execution_result(db: Session, execution_id: str) -> models.ExecutionResult | None:
    return (
        db.query(models.ExecutionResult)
        .filter(models.ExecutionResult.execution_id == execution_id)
        .first()
    )
