"""Session service — Option B sessions (docs/sessions.md).

A session is the conversation of one agent: executions that share a
``session_id`` are bound to the agent + mission of the session's
first execution. This module owns the session reads — grouping,
summary, binding check, and the prior-turn history the worker loads
before a run — but never runs the agent itself.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models
from app.services import execution_service
from runtime.session_history import TERMINAL_STATUSES, cap_turns, render_turn, summarize_turn


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


def check_session_ownership(
    db: Session, session_id: str, user_id: str | None
) -> str | None:
    """Ownership check for a user joining a session.

    Returns a problem string when the session holds executions owned
    by a different user, else None. Service-key calls (user_id None)
    are unrestricted, as in ``require_own_execution``. Executions
    with user_id NULL are pre-user legacy rows (seed/demo, service-key
    runs) and do not bind the session to anyone.
    """
    if user_id is None:
        return None
    owners = {
        row[0]
        for row in db.query(models.Execution.user_id)
        .filter(models.Execution.session_id == session_id)
        .distinct()
        .all()
    }
    if any(owner is not None and owner != user_id for owner in owners):
        return f"session '{session_id}' belongs to another user"
    return None


def list_user_sessions(db: Session, user_id: str) -> list[dict]:
    """Sessions the user has executions in, newest first."""
    rows = (
        db.query(
            models.Execution.session_id,
            models.Execution.mission_id,
            func.count(models.Execution.id),
            func.max(models.Execution.created_at),
        )
        .filter(models.Execution.user_id == user_id)
        .filter(models.Execution.session_id.isnot(None))
        .group_by(models.Execution.session_id, models.Execution.mission_id)
        .order_by(func.max(models.Execution.created_at).desc())
        .all()
    )
    return [
        {
            "session_id": session_id,
            "mission_id": mission_id,
            "execution_count": count,
            "last_created_at": last_created_at,
        }
        for session_id, mission_id, count, last_created_at in rows
    ]


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
    db: Session,
    session_id: str,
    exclude_execution_id: str,
    user_id: str | None = None,
) -> dict:
    """Prior turns of a session, capped, for the next run's agent.

    Only TERMINAL executions (SUCCEEDED / FAILED / CANCELLED) count:
    QUEUED / RUNNING rows are still in flight and their results do
    not exist yet. The current execution itself is excluded. Returns
    ``{"turns": [...], "omitted": int, "total_prior": int}`` — turns
    are oldest first, already capped (see
    ``runtime/session_history.py::cap_turns``).

    ``user_id`` scopes the load to the caller's own turns plus
    pre-user (NULL-owner) rows; a turn owned by another user is never
    shown to this run's agent. None means unfiltered (the pre-user
    behaviour — the worker passes the execution's owner).
    """
    query = (
        db.query(models.Execution)
        .filter(models.Execution.session_id == session_id)
        .filter(models.Execution.id != exclude_execution_id)
        .filter(models.Execution.status.in_(TERMINAL_STATUSES))
    )
    if user_id is not None:
        query = query.filter(
            (models.Execution.user_id.is_(None))
            | (models.Execution.user_id == user_id)
        )
    priors = query.order_by(
        models.Execution.created_at.asc(), models.Execution.id.asc()
    ).all()
    pairs = [
        (execution, execution_service.get_execution_result(db, execution.id))
        for execution in priors
    ]
    turns = [summarize_turn(execution, result) for execution, result in pairs]
    # Tools that already succeeded in this session (any terminal turn):
    # a prerequisite recorded in a prior turn still counts — e.g. a
    # payment recorded last turn satisfies confirm_payment's
    # ``requires: [record_payment]`` this turn. ``actions_taken`` lists
    # only calls whose results did not report failure, so it is the
    # successful-call set by construction. Computed from ALL priors,
    # not the capped turns: a state fact must not fall out of the
    # history window.
    prior_actions: list[str] = []
    seen_actions: set[str] = set()
    for execution, result in pairs:
        if execution.status != "SUCCEEDED":
            continue
        output = (result.output or {}) if result is not None else {}
        for name in output.get("actions_taken") or []:
            if isinstance(name, str) and name and name not in seen_actions:
                seen_actions.add(name)
                prior_actions.append(name)
    kept, omitted = cap_turns(turns)
    if omitted:
        # The load-time cap drops the oldest turns; encode them into
        # long-term memory before they are forgotten (best-effort —
        # encode_texts never raises, and it dedupes on
        # (session, execution, content hash), so the same dropped
        # turns are not re-stored on every later execution).
        kept_ids = {t["execution_id"] for t in kept}
        dropped = [t for t in turns if t["execution_id"] not in kept_ids]
        if dropped:
            from runtime.long_term_memory import encode_texts

            encode_texts(
                session_id,
                exclude_execution_id,
                [render_turn(t, i + 1) for i, t in enumerate(dropped)],
                source="session_cap",
            )
    return {
        "turns": kept,
        "omitted": omitted,
        "total_prior": len(turns),
        "prior_actions": prior_actions,
    }
