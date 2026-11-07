"""Result Manager — retrieve the result of a finished execution."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import schemas
from app.core.db import get_db
from app.core.security import require_api_key
from app.services import execution_service

router = APIRouter(
    prefix="/executions", tags=["results"], dependencies=[Depends(require_api_key)]
)


@router.get("/{execution_id}/result", response_model=schemas.ExecutionResultRead)
def get_result(execution_id: str, db: Session = Depends(get_db)):
    execution = execution_service.get_execution(db, execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    result = execution_service.get_execution_result(db, execution_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not ready yet")
    return schemas.ExecutionResultRead(
        execution_id=result.execution_id,
        output=result.output,
        evidence=result.evidence,
        created_at=result.created_at,
    )
