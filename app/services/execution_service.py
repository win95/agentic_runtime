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


def new_execution_id() -> str:
    return f"exec_{uuid.uuid4().hex[:12]}"


def validate_execution_input(mission: models.Mission, input_data: dict) -> list[str]:
    """Return a list of missing required input fields (empty = valid)."""
    required = (mission.input_schema or {}).get("required", []) or []
    return [field for field in required if field not in input_data]


def dispatch_execution(execution_id: str) -> None:
    """Enqueue only the execution id. Imported lazily so the API process
    does not need a live broker at import time. Tests monkeypatch this
    function to assert the dispatch without a broker."""
    from app.worker.tasks import run_execution

    run_execution.delay(execution_id)


def create_execution(
    db: Session, mission_id: str, input_data: dict
) -> tuple[models.Execution | None, list[str]]:
    """Create a QUEUED execution. Returns (execution, missing_fields).

    If the mission does not exist, returns (None, ["__mission_not_found__"]).
    If required inputs are missing, returns (None, missing_fields).
    """
    mission = mission_service.resolve_mission(db, mission_id)
    if mission is None:
        return None, ["__mission_not_found__"]
    missing = validate_execution_input(mission, input_data)
    if missing:
        return None, missing

    execution = models.Execution(
        id=new_execution_id(),
        mission_id=mission.id,
        mission_version=mission.version,
        agent_id=mission.agent_id,
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
        db.commit()
        raise
    return execution, []


def get_execution(db: Session, execution_id: str) -> models.Execution | None:
    return db.get(models.Execution, execution_id)


def list_executions(db: Session, limit: int = 50) -> list[models.Execution]:
    return (
        db.query(models.Execution)
        .order_by(models.Execution.created_at.desc())
        .limit(limit)
        .all()
    )


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
