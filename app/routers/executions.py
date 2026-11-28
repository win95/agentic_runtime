"""Execution Manager — create, track and cancel executions.

POST /executions validates the request, creates the Execution record and
dispatches it to Celery. It returns immediately with status QUEUED; the
agent runs asynchronously in the worker.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app import schemas
from app.core.db import SessionLocal, get_db
from app.core.security import (
    get_current_user,
    require_api_key_or_user,
    require_mission_for_role,
    require_own_execution,
)
from app.services import execution_service
from runtime.events import (
    HEARTBEAT_INTERVAL_SECONDS,
    MAX_STREAM_SECONDS,
    TERMINAL_STATUSES,
    build_event,
    subscribe_execution_events,
)
from runtime.execution_error_public import sanitize_from_execution

router = APIRouter(
    prefix="/executions",
    tags=["executions"],
    dependencies=[Depends(require_api_key_or_user)],
)


def _to_read(execution) -> schemas.ExecutionRead:
    return schemas.ExecutionRead(
        execution_id=execution.id,
        mission_id=execution.mission_id,
        mission_version=execution.mission_version,
        agent_id=execution.agent_id,
        agent_version=execution.agent_version,
        agent_version_drift=execution.agent_version_drift,
        session_id=execution.session_id,
        user_id=execution.user_id,
        status=execution.status,
        error=sanitize_from_execution(execution),
        created_at=execution.created_at,
        started_at=execution.started_at,
        finished_at=execution.finished_at,
    )


@router.post("", response_model=schemas.ExecutionRead, status_code=202)
def create_execution(
    payload: schemas.ExecutionCreate,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    # A plain JWT user may only run the guest mission; anything else is
    # 404 (never 403) so the API does not confirm the mission exists.
    require_mission_for_role(user, payload.mission_id)
    # Owner: a JWT user's own sub always wins (a JWT user cannot assert
    # someone else's id); a service-key call may assert an owner via
    # payload.user_id (agent_lark_me sync, D-LARK-11) — None means the
    # run stays a service-principal run, as before.
    asserted_user_id = payload.user_id if user is None else None
    try:
        execution, problems = execution_service.create_execution(
            db,
            payload.mission_id,
            payload.input,
            session_id=payload.session_id,
            user_id=user.get("sub") if user is not None else asserted_user_id,
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
    session_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    return [
        _to_read(e)
        for e in execution_service.list_executions(
            db,
            limit=limit,
            session_id=session_id,
            user_id=user.get("sub") if user is not None else None,
        )
    ]


@router.get("/{execution_id}")
def get_execution(
    execution_id: str,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    execution = execution_service.get_execution(db, execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    require_own_execution(execution, user)
    return _to_read(execution)


def _sse_event(payload: dict) -> str:
    return f"event: status\ndata: {json.dumps(payload)}\n\n"


def _execution_event_stream(execution_id: str):
    """Yield SSE events for one execution until it ends or times out.

    Opens its own DB session (the request's session is closed once the
    route returns). Redis pub/sub first: the worker publishes after
    every status commit, so no event is missed — a status change is
    committed before its publish, and a row already terminal at
    subscribe time is emitted from the DB. Without Redis, fall back
    to polling the executions table every second (same event shape,
    1s granularity). A heartbeat comment keeps proxies from closing
    an idle stream; the stream closes on the terminal event or after
    MAX_STREAM_SECONDS — the client reconnects or polls from there.
    """
    db = SessionLocal()
    pubsub = None
    try:
        last_status = None
        execution = execution_service.get_execution(db, execution_id)
        if execution is not None:
            last_status = execution.status
            if last_status in TERMINAL_STATUSES:
                yield _sse_event(
                    build_event("status", execution_id, status=last_status)
                )
                return
            if last_status == "INTERRUPTED":
                result = execution_service.get_execution_result(db, execution_id)
                interrupt = (result.output or {}).get("interrupt") or {}
                yield _sse_event(
                    build_event(
                        "status", execution_id, status=last_status,
                        interrupt=interrupt,
                    )
                )
                return
        elif last_status is None:
            # Execution vanished after the ownership check — treat as
            # terminal rather than streaming forever.
            return
        pubsub = subscribe_execution_events(execution_id)
        deadline = time.monotonic() + MAX_STREAM_SECONDS
        last_heartbeat = time.monotonic()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if pubsub is not None:
                try:
                    message = pubsub.get_message(timeout=min(1.0, remaining))
                except Exception:  # noqa: BLE001 — Redis died mid-stream
                    try:
                        pubsub.close()
                    except Exception:  # noqa: BLE001 — best effort
                        pass
                    pubsub = None
                    message = None
                if message is not None and message.get("type") == "message":
                    payload = json.loads(message["data"])
                    yield _sse_event(payload)
                    if payload.get("status") in TERMINAL_STATUSES:
                        return
                    last_status = payload.get("status") or last_status
                else:
                    # No message this tick — re-read the row so a
                    # publish lost to a Redis hiccup still surfaces.
                    db.expire_all()
                    execution = execution_service.get_execution(db, execution_id)
                    if execution is not None and execution.status != last_status:
                        last_status = execution.status
                        extra: dict = {}
                        if last_status == "INTERRUPTED":
                            result = execution_service.get_execution_result(
                                db, execution_id
                            )
                            extra["interrupt"] = (
                                (result.output or {}).get("interrupt") or {}
                            )
                        yield _sse_event(
                            build_event(
                                "status", execution_id, status=last_status, **extra
                            )
                        )
                        if last_status in TERMINAL_STATUSES:
                            return
            else:
                time.sleep(min(1.0, remaining))
                db.expire_all()
                execution = execution_service.get_execution(db, execution_id)
                if execution is not None and execution.status != last_status:
                    last_status = execution.status
                    extra = {}
                    if last_status == "INTERRUPTED":
                        result = execution_service.get_execution_result(
                            db, execution_id
                        )
                        extra["interrupt"] = (
                            (result.output or {}).get("interrupt") or {}
                        )
                    yield _sse_event(
                        build_event(
                            "status", execution_id, status=last_status, **extra
                        )
                    )
                    if last_status in TERMINAL_STATUSES:
                        return
            if time.monotonic() - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                last_heartbeat = time.monotonic()
                yield ": ping\n\n"
    finally:
        if pubsub is not None:
            try:
                pubsub.close()
            except Exception:  # noqa: BLE001 — best effort
                pass
        db.close()


@router.get("/{execution_id}/stream")
def stream_execution(
    execution_id: str,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    """Server-Sent Events for one execution's lifecycle (A.2).

    Emits ``status`` events (RUNNING, INTERRUPTED, terminal) and
    closes on the terminal event or after MAX_STREAM_SECONDS. The
    payload carries no output — the client fetches the full result
    from ``GET /executions/{id}/result`` when the stream ends. Same
    ownership rule as GET: another user's run is 404.
    """
    execution = execution_service.get_execution(db, execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    require_own_execution(execution, user)
    return StreamingResponse(
        _execution_event_stream(execution_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{execution_id}/cancel", response_model=schemas.ExecutionRead)
def cancel_execution(
    execution_id: str,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    execution = execution_service.get_execution(db, execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    require_own_execution(execution, user)
    execution = execution_service.cancel_execution(db, execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    return _to_read(execution)


@router.post("/{execution_id}/resume", response_model=schemas.ExecutionRead)
def resume_execution(
    execution_id: str,
    payload: schemas.ExecutionResumeRequest,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    """Resume an execution paused on ``interrupt_on`` (docs/interrupt.md).

    Only an INTERRUPTED execution resumes, and only with its owner's
    decision (same ownership rule as GET — another user's run is 404).
    The row goes INTERRUPTED -> QUEUED and the worker resumes the SAME
    execution with the decision; the gated tool then runs (approve) or
    is skipped (reject). If the dispatch fails (broker down) the row
    goes back to INTERRUPTED, so the pause survives and the resume can
    be retried — the pause is only consumed by a decision that reaches
    the worker.
    """
    execution = execution_service.get_execution(db, execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    require_own_execution(execution, user)
    if execution.status != "INTERRUPTED":
        raise HTTPException(
            status_code=409,
            detail=f"Execution is {execution.status}, not INTERRUPTED",
        )
    execution.status = "QUEUED"
    execution.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()
    try:
        execution_service.dispatch_resume(
            execution_id, payload.decision, execution.mission_id
        )
    except Exception as exc:  # noqa: BLE001 — broker down: keep the pause
        execution.status = "INTERRUPTED"
        execution.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        raise HTTPException(
            status_code=502, detail=f"resume dispatch failed: {exc}"
        ) from exc
    return _to_read(execution)
