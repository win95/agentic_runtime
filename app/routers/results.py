"""Result Manager — retrieve the result of a finished execution."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import schemas
from app.core.db import get_db
from app.core.security import (
    get_current_user,
    require_api_key_or_user,
    require_own_execution,
)
from app.services import execution_service
from runtime.guest_result import is_guest_api_execution, project_guest_result

router = APIRouter(
    prefix="/executions",
    tags=["results"],
    dependencies=[Depends(require_api_key_or_user)],
)


@router.get("/{execution_id}/result", response_model=schemas.ExecutionResultRead)
def get_result(
    execution_id: str,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    execution = execution_service.get_execution(db, execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    require_own_execution(execution, user)
    result = execution_service.get_execution_result(db, execution_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not ready yet")
    output = dict(result.output or {})
    evidence = list(result.evidence or [])
    if is_guest_api_execution(execution.input):
        output, evidence = project_guest_result(
            output,
            evidence,
            session_id=execution.session_id,
            execution_input=execution.input,
        )
    return schemas.ExecutionResultRead(
        execution_id=result.execution_id,
        output=output,
        evidence=evidence,
        created_at=result.created_at,
    )
