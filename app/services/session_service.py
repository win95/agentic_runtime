"""Session service — Option B sessions (docs/sessions.md).

A session is the conversation of one agent: executions that share a
``session_id`` are bound to the agent + mission of the session's
first execution. This module owns the session reads — grouping,
summary, binding check, and the prior-turn history the worker loads
before a run — but never runs the agent itself.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app import models
from app.services import execution_service
from runtime.session_history import TERMINAL_STATUSES, cap_turns, summarize_turn


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


def list_session_executions(db: Session, session_id: str) -> list[models.Execution]:
    """All executions in one session, oldest first (conversation order).

    ``created_at`` ties (same-second creates) break on the id so the
    order is deterministic.
    """
    return (
        db.query(models.Execution)
        .filter(models.Execution.session_id == session_id)
        .order_by(models.Execution.created_at.asc(), models.Execution.id.asc())
        .all()
    )


def get_session_summary(db: Session, session_id: str) -> dict | None:
    """Summary of one session, or None when it has no executions.

    The first execution defines the session's agent + mission; the
    summary carries the executions (oldest first) so one call reads a
    whole conversation.
    """
    executions = list_session_executions(db, session_id)
    if not executions:
        return None
    first = executions[0]
    return {
        "session_id": session_id,
        "agent_id": first.agent_id,
        "mission_id": first.mission_id,
        "execution_count": len(executions),
        "first_created_at": first.created_at,
        "last_created_at": executions[-1].created_at,
        "executions": executions,
    }


def check_session_binding(
    db: Session, session_id: str, mission_id: str, agent_id: str | None
) -> str | None:
    """Binding check for a new execution joining a session.

    Returns a problem string when the session's first execution ran a
    different mission or agent, else None. A session with no
    executions yet binds freely (this execution becomes its first).
    Best-effort, not a transactional guarantee: two concurrent
    creates in a brand-new session can both pass (documented in
    docs/sessions.md).
    """
    prior = (
        db.query(models.Execution)
        .filter(models.Execution.session_id == session_id)
        .order_by(models.Execution.created_at.asc(), models.Execution.id.asc())
        .first()
    )
    if prior is None:
        return None
    if prior.mission_id != mission_id or prior.agent_id != agent_id:
        return (
            f"session '{session_id}' belongs to mission '{prior.mission_id}' "
            f"with agent '{prior.agent_id}', but this execution is for "
            f"mission '{mission_id}' with agent '{agent_id}'"
        )
    return None


def load_session_history(
    db: Session, session_id: str, exclude_execution_id: str
) -> dict:
    """Prior turns of a session, capped, for the next run's agent.

    Only TERMINAL executions (SUCCEEDED / FAILED / CANCELLED) count:
    QUEUED / RUNNING rows are still in flight and their results do
    not exist yet. The current execution itself is excluded. Returns
    ``{"turns": [...], "omitted": int, "total_prior": int}`` — turns
    are oldest first, already capped (see
    ``runtime/session_history.py::cap_turns``).
    """
    priors = (
        db.query(models.Execution)
        .filter(models.Execution.session_id == session_id)
        .filter(models.Execution.id != exclude_execution_id)
        .filter(models.Execution.status.in_(TERMINAL_STATUSES))
        .order_by(models.Execution.created_at.asc(), models.Execution.id.asc())
        .all()
    )
    turns = [
        summarize_turn(
            execution,
            execution_service.get_execution_result(db, execution.id),
        )
        for execution in priors
    ]
    kept, omitted = cap_turns(turns)
    return {"turns": kept, "omitted": omitted, "total_prior": len(turns)}
