"""Read helpers for the middleware tables (token usage + audit log).

The admin bot and any future cost view use these — no HTTP endpoints
are added in this change.
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models


def get_usage_for_execution(db: Session, execution_id: str) -> list[models.LLMUsage]:
    """Token-usage rows for one execution, oldest first."""
    return (
        db.query(models.LLMUsage)
        .filter(models.LLMUsage.execution_id == execution_id)
        .order_by(models.LLMUsage.created_at)
        .all()
    )


def get_usage_totals(
    db: Session, mission_id: str | None = None, tenant: str | None = None
) -> dict:
    """Summed tokens across matching usage rows (0 when there are none).

    Only rows with provider-reported usage are summed; rows with
    ``usage_source="missing"`` contribute NULL and are ignored by SUM.
    """
    query = db.query(
        func.coalesce(func.sum(models.LLMUsage.prompt_tokens), 0).label("prompt_tokens"),
        func.coalesce(func.sum(models.LLMUsage.completion_tokens), 0).label(
            "completion_tokens"
        ),
        func.coalesce(func.sum(models.LLMUsage.total_tokens), 0).label("total_tokens"),
        func.count(models.LLMUsage.id).label("calls"),
    )
    if mission_id is not None:
        query = query.filter(models.LLMUsage.mission_id == mission_id)
    if tenant is not None:
        query = query.filter(models.LLMUsage.tenant == tenant)
    row = query.one()
    return {
        "prompt_tokens": row.prompt_tokens,
        "completion_tokens": row.completion_tokens,
        "total_tokens": row.total_tokens,
        "calls": row.calls,
    }


def get_audit_for_execution(db: Session, execution_id: str) -> list[models.AuditLog]:
    """Audit rows for one execution, oldest first."""
    return (
        db.query(models.AuditLog)
        .filter(models.AuditLog.execution_id == execution_id)
        .order_by(models.AuditLog.created_at)
        .all()
    )


def get_tool_call_counts(
    db: Session,
    execution_id: str | None = None,
    mission_id: str | None = None,
) -> dict[str, int]:
    """Tool name -> call count for the matching audit rows."""
    query = db.query(models.AuditLog.tool_name, func.count(models.AuditLog.id))
    if execution_id is not None:
        query = query.filter(models.AuditLog.execution_id == execution_id)
    if mission_id is not None:
        query = query.filter(models.AuditLog.mission_id == mission_id)
    query = query.group_by(models.AuditLog.tool_name)
    return {name: count for name, count in query.all()}
