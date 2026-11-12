"""Human approval for destructive tool calls — Option C (docs/hooks.md).

The ``request_permission`` hook (PRE_TOOL_USE) calls
:func:`check_permission` before ``delete_mission`` /
``cancel_execution`` run. The approval is scoped to one (tool,
target) ACTION, not to one execution: a pending request is shared
by every run that asks for the same action, and approving it
re-dispatches the original execution — the re-run finds the approved
record and the hook lets the call through.

Rules:
- pending requests expire after 24h; approvals are single-use and
  expire 60 minutes after approval.
- the claim is atomic (UPDATE ... WHERE status='approved'): two
  concurrent runs cannot both consume one approval.
- this module never raises into the hook: the hook treats any error
  here as a deny (fail-closed).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app import models

logger = logging.getLogger(__name__)

PENDING_TTL = timedelta(hours=24)
APPROVED_TTL = timedelta(minutes=60)


def new_request_id() -> str:
    return f"perm_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _expire_stale(db: Session, tool_name: str, target: str) -> None:
    """Mark past-expiry pending/approved rows for one (tool, target) expired."""
    db.query(models.PermissionRequest).filter(
        models.PermissionRequest.tool_name == tool_name,
        models.PermissionRequest.target == target,
        models.PermissionRequest.status.in_(["pending", "approved"]),
        models.PermissionRequest.expires_at <= _now(),
    ).update({"status": "expired"}, synchronize_session=False)
    db.commit()


def find_usable_approval(
    db: Session, tool_name: str, target: str
) -> models.PermissionRequest | None:
    """An approved, unexpired, unused record for one (tool, target)."""
    return (
        db.query(models.PermissionRequest)
        .filter(
            models.PermissionRequest.tool_name == tool_name,
            models.PermissionRequest.target == target,
            models.PermissionRequest.status == "approved",
            models.PermissionRequest.expires_at > _now(),
        )
        .order_by(models.PermissionRequest.decided_at.desc())
        .first()
    )


def claim_approval(db: Session, request: models.PermissionRequest) -> bool:
    """Atomically consume one approval. True iff this caller claimed it."""
    rows = (
        db.query(models.PermissionRequest)
        .filter(
            models.PermissionRequest.id == request.id,
            models.PermissionRequest.status == "approved",
            models.PermissionRequest.expires_at > _now(),
        )
        .update(
            {"status": "used", "decided_at": _now()},
            synchronize_session=False,
        )
    )
    db.commit()
    return rows == 1


def find_pending(
    db: Session, tool_name: str, target: str
) -> models.PermissionRequest | None:
    """An unexpired pending record for one (tool, target), if any."""
    return (
        db.query(models.PermissionRequest)
        .filter(
            models.PermissionRequest.tool_name == tool_name,
            models.PermissionRequest.target == target,
            models.PermissionRequest.status == "pending",
            models.PermissionRequest.expires_at > _now(),
        )
        .order_by(models.PermissionRequest.created_at.asc())
        .first()
    )


def _notify_created(row: models.PermissionRequest) -> None:
    """Best-effort Slack ping for a new pending request. Never raises."""
    try:
        from app.config import settings
        from runtime.notifications import slack as slack_notify

        slack_notify.notify_permission_request(
            request_id=row.id,
            tool_name=row.tool_name,
            target=row.target,
            execution_id=row.execution_id,
            mission_id=row.mission_id,
            enabled=settings.slack_notifications_enabled,
            webhook_url=settings.slack_webhook_url,
        )
    except Exception as exc:  # noqa: BLE001 — notification never fails a run
        logger.warning("Permission-request Slack ping failed: %s", exc)


def create_request(
    db: Session,
    *,
    execution_id: str | None,
    mission_id: str | None,
    agent_id: str | None,
    tool_name: str,
    target: str,
    tool_input: dict,
) -> models.PermissionRequest:
    """Create a pending request (24h expiry) and ping Slack about it."""
    row = models.PermissionRequest(
        id=new_request_id(),
        execution_id=execution_id,
        mission_id=mission_id,
        agent_id=agent_id,
        tool_name=tool_name,
        target=target,
        tool_input=dict(tool_input or {}),
        status="pending",
        expires_at=_now() + PENDING_TTL,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    _notify_created(row)
    return row


def check_permission(
    db: Session,
    *,
    execution_id: str | None,
    mission_id: str | None,
    agent_id: str | None,
    tool_name: str,
    target: str,
    tool_input: dict,
) -> tuple[str, models.PermissionRequest | None]:
    """Decide one destructive call: ("allowed" | "pending" | "requested", row).

    "allowed" — an approval was claimed (single-use, now consumed).
    "pending" — a request for this action already waits for a human.
    "requested" — a fresh request was just created.
    """
    _expire_stale(db, tool_name, target)
    approval = find_usable_approval(db, tool_name, target)
    if approval is not None and claim_approval(db, approval):
        return "allowed", approval
    pending = find_pending(db, tool_name, target)
    if pending is not None:
        return "pending", pending
    return "requested", create_request(
        db,
        execution_id=execution_id,
        mission_id=mission_id,
        agent_id=agent_id,
        tool_name=tool_name,
        target=target,
        tool_input=tool_input,
    )


def get_request(db: Session, request_id: str) -> models.PermissionRequest | None:
    return db.get(models.PermissionRequest, request_id)


def list_pending(db: Session, limit: int = 50) -> list[models.PermissionRequest]:
    """Pending requests, oldest first (the ones needing a human)."""
    return (
        db.query(models.PermissionRequest)
        .filter(models.PermissionRequest.status == "pending")
        .order_by(models.PermissionRequest.created_at.asc())
        .limit(limit)
        .all()
    )


def approve_request(
    db: Session, request_id: str, note: str | None = None
) -> models.PermissionRequest | None:
    """Approve a pending request (60-minute single-use window).

    Returns None when the request does not exist or is not pending.
    """
    row = get_request(db, request_id)
    if row is None or row.status != "pending":
        return None
    row.status = "approved"
    row.decided_at = _now()
    row.expires_at = _now() + APPROVED_TTL
    row.decision_note = note
    db.commit()
    db.refresh(row)
    return row


def deny_request(
    db: Session, request_id: str, note: str | None = None
) -> models.PermissionRequest | None:
    """Deny a pending request. Returns None when missing or not pending."""
    row = get_request(db, request_id)
    if row is None or row.status != "pending":
        return None
    row.status = "denied"
    row.decided_at = _now()
    row.decision_note = note
    db.commit()
    db.refresh(row)
    return row
