"""Human approval for destructive tool calls (Option C, docs/hooks.md).

The ``request_permission`` hook denies ``delete_mission`` /
``cancel_execution`` until a human approves the pending request here.
Approving marks the request approved (single-use, 60-minute window)
and re-dispatches the original execution, so the agent retries the
call with the approval in place. Denying ends it: no re-dispatch.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import schemas
from app.core.db import get_db
from app.core.security import require_api_key
from app.services import execution_service, permission_service

router = APIRouter(
    prefix="/permissions", tags=["permissions"], dependencies=[Depends(require_api_key)]
)


def _to_read(row, redispatched_execution_id: str | None = None) -> schemas.PermissionRequestRead:
    return schemas.PermissionRequestRead(
        request_id=row.id,
        execution_id=row.execution_id,
        mission_id=row.mission_id,
        agent_id=row.agent_id,
        tool_name=row.tool_name,
        target=row.target,
        tool_input=dict(row.tool_input or {}),
        status=row.status,
        decision_note=row.decision_note,
        created_at=row.created_at,
        decided_at=row.decided_at,
        expires_at=row.expires_at,
        redispatched_execution_id=redispatched_execution_id,
    )


@router.get("", response_model=list[schemas.PermissionRequestRead])
def list_pending(
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Pending approval requests, oldest first."""
    return [_to_read(row) for row in permission_service.list_pending(db, limit=limit)]


@router.get("/{request_id}", response_model=schemas.PermissionRequestRead)
def get_one(request_id: str, db: Session = Depends(get_db)):
    row = permission_service.get_request(db, request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Permission request not found")
    return _to_read(row)


@router.post("/{request_id}/approve", response_model=schemas.PermissionRequestRead)
def approve(
    request_id: str,
    payload: schemas.PermissionDecision | None = None,
    db: Session = Depends(get_db),
):
    """Approve a pending request and re-dispatch the original execution.

    The approval is single-use with a 60-minute window; the re-run
    claims it when the agent retries the destructive call. When the
    original execution cannot be re-dispatched (mission deleted,
    input no longer valid) the approval still stands and
    ``redispatched_execution_id`` is null — re-dispatch manually
    within the window.
    """
    row = permission_service.get_request(db, request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Permission request not found")
    approved = permission_service.approve_request(
        db, request_id, note=payload.note if payload else None
    )
    if approved is None:
        raise HTTPException(status_code=409, detail="Permission request is not pending")
    redispatched = None
    execution = (
        execution_service.get_execution(db, row.execution_id)
        if row.execution_id
        else None
    )
    if execution is not None:
        new_execution, _problems = execution_service.create_execution(
            db,
            execution.mission_id,
            dict(execution.input or {}),
            session_id=execution.session_id,
        )
        if new_execution is not None:
            redispatched = new_execution.id
    return _to_read(approved, redispatched_execution_id=redispatched)


@router.post("/{request_id}/deny", response_model=schemas.PermissionRequestRead)
def deny(
    request_id: str,
    payload: schemas.PermissionDecision | None = None,
    db: Session = Depends(get_db),
):
    """Deny a pending request. No re-dispatch happens."""
    row = permission_service.get_request(db, request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Permission request not found")
    denied = permission_service.deny_request(
        db, request_id, note=payload.note if payload else None
    )
    if denied is None:
        raise HTTPException(status_code=409, detail="Permission request is not pending")
    return _to_read(denied)
