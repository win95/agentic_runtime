"""Threads / Runs — the MDA Agent Server shape over sessions/executions.

A thread is a session (``thread_id`` is the ``session_id``) and a run
is an execution (``run_id`` is the ``execution_id``). These routes are
thin aliases: they create no tables and run no agent themselves — they
read and dispatch through the existing session/execution services, so
``GET /threads/{id}`` and ``GET /sessions/{id}`` always agree, as do
``GET /runs/{id}`` and ``GET /executions/{id}``.

Ownership mirrors the aliased routes exactly: a JWT user only touches
their own threads/runs (another user's is 404); service-key calls stay
unrestricted. Creating a run in a thread needs no ownership check —
like ``POST /executions``, the run is owned by whoever creates it, and
a thread with no runs yet does not exist to own.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import schemas
from app.core.db import get_db
from app.core.security import (
    get_current_user,
    require_api_key_or_user,
    require_mission_for_role,
    require_own_execution,
)
from app.services import execution_service, session_service
from runtime.guest_result import is_guest_api_execution, project_guest_result
from runtime.security import is_valid_session_id

threads_router = APIRouter(
    prefix="/threads",
    tags=["threads"],
    dependencies=[Depends(require_api_key_or_user)],
)

runs_router = APIRouter(
    prefix="/runs",
    tags=["runs"],
    dependencies=[Depends(require_api_key_or_user)],
)


def _to_run_read(execution) -> schemas.RunRead:
    return schemas.RunRead(
        run_id=execution.id,
        thread_id=execution.session_id,
        mission_id=execution.mission_id,
        agent_id=execution.agent_id,
        status=execution.status,
        error=execution.error,
        created_at=execution.created_at,
        started_at=execution.started_at,
        finished_at=execution.finished_at,
    )


def _to_thread_read(summary: dict) -> schemas.ThreadRead:
    return schemas.ThreadRead(
        thread_id=summary["session_id"],
        agent_id=summary["agent_id"],
        mission_id=summary["mission_id"],
        execution_count=summary["execution_count"],
        first_created_at=summary["first_created_at"],
        last_created_at=summary["last_created_at"],
        runs=[_to_run_read(e) for e in summary["executions"]],
    )


def _check_thread_id(thread_id: str) -> None:
    if not is_valid_session_id(thread_id):
        raise HTTPException(
            status_code=422,
            detail="Invalid thread_id: must match ^[A-Za-z0-9_-]{1,64}$",
        )


def _check_thread_ownership(
    db: Session, thread_id: str, user: dict | None
) -> None:
    if user is not None:
        problem = session_service.check_session_ownership(
            db, thread_id, user.get("sub")
        )
        if problem is not None:
            raise HTTPException(status_code=404, detail="Thread not found")


def _get_thread_summary(
    db: Session, thread_id: str, user: dict | None
) -> dict:
    _check_thread_id(thread_id)
    _check_thread_ownership(db, thread_id, user)
    summary = session_service.get_session_summary(db, thread_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return summary


@threads_router.post("", response_model=schemas.ThreadRead, status_code=201)
def create_thread():
    """Mint a thread id. The thread has no row until its first run —
    like sessions, threads are created implicitly by their first run,
    so GET on this id is 404 until then.
    """
    return schemas.ThreadRead(thread_id=session_service.new_session_id())


@threads_router.get("/{thread_id}", response_model=schemas.ThreadRead)
def get_thread(
    thread_id: str,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    return _to_thread_read(_get_thread_summary(db, thread_id, user))


@threads_router.get("/{thread_id}/runs", response_model=list[schemas.RunRead])
def list_thread_runs(
    thread_id: str,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    summary = _get_thread_summary(db, thread_id, user)
    return [_to_run_read(e) for e in summary["executions"]]


@threads_router.post(
    "/{thread_id}/runs", response_model=schemas.RunRead, status_code=202
)
def create_thread_run(
    thread_id: str,
    payload: schemas.ThreadRunCreate,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    """Run a mission in a thread — the MDA-shaped ``POST /executions``."""
    _check_thread_id(thread_id)
    require_mission_for_role(user, payload.mission_id)
    try:
        execution, problems = execution_service.create_execution(
            db,
            payload.mission_id,
            payload.input,
            session_id=thread_id,
            user_id=user.get("sub") if user is not None else None,
        )
    except Exception as exc:  # dispatch failure (broker down, ...)
        raise HTTPException(status_code=502, detail=f"dispatch failed: {exc}") from exc
    if execution is None:
        if problems == ["__mission_not_found__"]:
            raise HTTPException(status_code=404, detail="Mission not found")
        raise HTTPException(
            status_code=422, detail=f"Invalid execution input: {problems}"
        )
    return _to_run_read(execution)


@runs_router.get("/{run_id}", response_model=schemas.RunRead)
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    execution = execution_service.get_execution(db, run_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Run not found")
    require_own_execution(execution, user)
    return _to_run_read(execution)


@runs_router.get("/{run_id}/result", response_model=schemas.ExecutionResultRead)
def get_run_result(
    run_id: str,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    # Same read as GET /executions/{id}/result (see app/routers/results.py).
    execution = execution_service.get_execution(db, run_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Run not found")
    require_own_execution(execution, user)
    result = execution_service.get_execution_result(db, run_id)
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
