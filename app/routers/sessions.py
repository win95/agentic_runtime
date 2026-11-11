"""Session reader — one conversation of one agent (Option B).

``GET /sessions/{session_id}`` returns the session summary plus its
executions, oldest first (conversation order). A session with no
executions is a 404; a malformed id is a 422. Sessions are created
implicitly by ``POST /executions`` (docs/sessions.md) — there is no
separate create endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import schemas
from app.core.db import get_db
from app.core.security import require_api_key
from app.routers.executions import _to_read
from app.services import session_service
from runtime.security import is_valid_session_id

router = APIRouter(
    prefix="/sessions", tags=["sessions"], dependencies=[Depends(require_api_key)]
)


@router.get("/{session_id}", response_model=schemas.SessionRead)
def get_session(session_id: str, db: Session = Depends(get_db)):
    if not is_valid_session_id(session_id):
        raise HTTPException(
            status_code=422,
            detail="Invalid session_id: must match ^[A-Za-z0-9_-]{1,64}$",
        )
    summary = session_service.get_session_summary(db, session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return schemas.SessionRead(
        session_id=summary["session_id"],
        agent_id=summary["agent_id"],
        mission_id=summary["mission_id"],
        execution_count=summary["execution_count"],
        first_created_at=summary["first_created_at"],
        last_created_at=summary["last_created_at"],
        executions=[_to_read(e) for e in summary["executions"]],
    )
