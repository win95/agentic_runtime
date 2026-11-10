"""Execution Manager — create, track and cancel executions.

POST /executions validates the request, creates the Execution record and
dispatches it to Celery. It returns immediately with status QUEUED; the
agent runs asynchronously in the worker.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import schemas
from app.core.db import get_db
from app.core.security import require_api_key
from app.services import execution_service

router = APIRouter(
    prefix="/executions", tags=["executions"], dependencies=[Depends(require_api_key)]
)


def _to_read(execution) -> schemas.ExecutionRead:
    return schemas.ExecutionRead(
        execution_id=execution.id,
        mission_id=execution.mission_id,
        mission_version=execution.mission_version,
        agent_id=execution.agent_id,
        agent_version=execution.agent_version,
        agent_version_drift=execution.agent_version_drift,
        status=execution.status,
        error=execution.error,
        created_at=execution.created_at,
        started_at=execution.started_at,
        finished_at=execution.finished_at,
    )


@router.post("", response_model=schemas.ExecutionRead, status_code=202)
def create_execution(payload: schemas.ExecutionCreate, db: Session = Depends(get_db)):
    try:
        execution, problems = execution_service.create_execution(
            db, payload.mission_id, payload.input
        )
    except Exception as exc:  # dispatch failure (broker down, ...)
        raise HTTPException(status_code=502, detail=f"dispatch failed: {exc}") from exc
    if execution is None:
        if problems == ["__mission_not_found__"]:
            raise HTTPException(status_code=404, detail="Mission not found")
        raise HTTPException(
            status_code=422, detail=f"Invalid execution input: {problems}"
        )
    return _to_read(execution)


@router.get("", response_model=list[schemas.ExecutionRead])
def list_executions(
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return [_to_read(e) for e in execution_service.list_executions(db, limit=limit)]


@router.get("/{execution_id}", response_model=schemas.ExecutionRead)
def get_execution(execution_id: str, db: Session = Depends(get_db)):
    execution = execution_service.get_execution(db, execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    return _to_read(execution)


@router.post("/{execution_id}/cancel", response_model=schemas.ExecutionRead)
def cancel_execution(execution_id: str, db: Session = Depends(get_db)):
    execution = execution_service.cancel_execution(db, execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    return _to_read(execution)
