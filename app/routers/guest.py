"""Guest API (demo) — public guest chat.

JWT-only: guests register/login via the user service and call with a
Bearer token. The mission comes from harness config (the mission with
``guest_channel.exposes_public_chat``). Guests cannot invoke other
missions (``POST /executions`` is not public for the hotel mission only
when ``security.user_runnable`` allows it).

Flow: ``POST /guest/chat`` creates a QUEUED execution and returns
immediately; the client polls ``GET /executions/{id}/result`` with
the same Bearer token (the results router enforces ownership, so a
guest only ever sees their own runs). ``GET /guest/sessions`` lists
the caller's own sessions.

``guest_id`` and ``channel`` in the execution input come from the
server (JWT sub), never from the request body — the guest cannot
impersonate another user to the agent.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import schemas
from app.core.db import get_db
from app.core.rate_limit import check_guest_rate_limit
from app.core.security import require_jwt_user
from app.services import execution_service, session_service

router = APIRouter(prefix="/guest", tags=["guest"])

@router.post("/chat", response_model=schemas.GuestChatRead, status_code=202)
def guest_chat(
    payload: schemas.GuestChatCreate,
    db: Session = Depends(get_db),
    user: dict = Depends(require_jwt_user),
):
    from runtime.harness.registry import (
        GuestChatMisconfiguredError,
        guest_chat_mission_id,
    )

    sub = user.get("sub")
    check_guest_rate_limit(sub)
    try:
        mission_id = guest_chat_mission_id()
    except GuestChatMisconfiguredError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Guest chat is not configured: {exc}",
        ) from exc
    execution, problems = execution_service.create_execution(
        db,
        mission_id,
        {"message": payload.message, "guest_id": sub, "channel": "guest_api"},
        session_id=payload.session_id,
        user_id=sub,
    )
    if execution is None:
        if problems == ["__mission_not_found__"]:
            raise HTTPException(
                status_code=404, detail=f"Mission {mission_id} not found"
            )
        raise HTTPException(
            status_code=422, detail=f"Invalid execution input: {problems}"
        )
    return schemas.GuestChatRead(
        execution_id=execution.id,
        session_id=execution.session_id,
        status=execution.status,
    )


@router.get("/sessions", response_model=list[schemas.GuestSessionRead])
def guest_sessions(
    db: Session = Depends(get_db),
    user: dict = Depends(require_jwt_user),
):
    return [
        schemas.GuestSessionRead(
            session_id=row["session_id"],
            mission_id=row["mission_id"],
            execution_count=row["execution_count"],
            last_created_at=row["last_created_at"],
        )
        for row in session_service.list_user_sessions(db, user.get("sub"))
    ]
