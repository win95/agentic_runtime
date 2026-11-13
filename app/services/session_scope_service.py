"""Session -> guest binding for the hotel data scope (Layer 1/L2).

One session is one guest. The guest scope wrapper in
``runtime/tools/registry.py`` binds a session on first sight of a
guest_id (a find_guest / register_guest / get_booking /
create_booking call or result) and enforces later calls against it;
``validate_results`` reads the same binding to fail runs whose
evidence names another guest.

The binding is best-effort: a DB failure leaves the session unbound
and the wrapper fails open (no binding -> no guest check), because a
scope-store outage must not break the receptionist's normal flow.
The guarantee holds whenever the store is up, which is the normal
case — executions themselves live in the same database.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app import models

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_session_guest(db: Session, session_id: str) -> models.SessionGuestScope | None:
    """The guest bound to one session, or None when unbound."""
    if not session_id:
        return None
    return (
        db.query(models.SessionGuestScope)
        .filter(models.SessionGuestScope.session_id == session_id)
        .one_or_none()
    )


def bind_session_guest(
    db: Session,
    session_id: str,
    guest_id: str,
    guest_phone: str | None = None,
    mission_id: str | None = None,
) -> models.SessionGuestScope | None:
    """Bind a session to a guest (first sight wins — never rebind).

    Returns the row (existing or new), or None when the binding could
    not be written. A session bound to one guest is never rebound to
    another: a second guest_id for the same session is ignored and the
    original binding is returned.
    """
    if not session_id or not guest_id:
        return None
    existing = get_session_guest(db, session_id)
    if existing is not None:
        return existing
    row = models.SessionGuestScope(
        session_id=session_id,
        guest_id=guest_id,
        guest_phone=guest_phone,
        mission_id=mission_id,
        created_at=_now(),
        updated_at=_now(),
    )
    db.add(row)
    db.commit()
    return row
