"""Layer enforcement — execution scope helpers (L0/L1/L3).

The scope is what one execution may touch, fixed once at the start:
the input ``asset_id`` (fault missions), the session's guest (hotel —
guest identity is the session), and the tenant. Reads stay open so
the agent can build context; the *data* is what gets scoped, never
the tool name:

- L1: the scope wrappers in ``runtime/tools/registry.py`` refuse or
  withhold out-of-scope data at the tool boundary (asset wrapper for
  ``asset_id`` tools, guest wrapper for guest/booking tools).
- L3: :func:`scope_violations_in_evidence` lets ``validate_results``
  fail a run whose collected evidence names data outside the scope.

The session -> guest binding lives in ``session_guest_scopes``
(``app/services/session_scope_service.py``). Access here is
best-effort: a scope-store failure returns None (unbound) and the
wrappers fail open, because a store outage must not break the
receptionist's normal flow — the guarantee holds whenever the store
is up, which is the normal case.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_bound_guest_id(session_id: str) -> str | None:
    """Guest id bound to one session, or None when unbound/unknown."""
    if not session_id:
        return None
    try:
        from app.core.db import SessionLocal
        from app.services import session_scope_service

        db = SessionLocal()
        try:
            row = session_scope_service.get_session_guest(db, session_id)
            return row.guest_id if row is not None else None
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 — binding is best-effort
        logger.warning("Session guest lookup failed for %s: %s", session_id, exc)
        return None


def bind_guest(
    session_id: str,
    guest_id: str,
    guest_phone: str | None = None,
    mission_id: str | None = None,
) -> str | None:
    """Bind a session to a guest (first sight wins); returns the bound id."""
    if not session_id or not guest_id:
        return None
    try:
        from app.core.db import SessionLocal
        from app.services import session_scope_service

        db = SessionLocal()
        try:
            row = session_scope_service.bind_session_guest(
                db, session_id, guest_id, guest_phone, mission_id
            )
            return row.guest_id if row is not None else None
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 — binding is best-effort
        logger.warning("Session guest bind failed for %s: %s", session_id, exc)
        return None


def guest_id_from_result(result) -> str | None:
    """Guest id carried by one tool result, if it names exactly one guest."""
    if not isinstance(result, dict):
        return None
    guest_id = result.get("guest_id")
    if isinstance(guest_id, str) and guest_id:
        return guest_id
    guests = result.get("guests")
    if isinstance(guests, list) and len(guests) == 1:
        single = guests[0].get("guest_id") if isinstance(guests[0], dict) else None
        if isinstance(single, str) and single:
            return single
    return None


def _guest_ids_in_data(data) -> list[str]:
    """Guest ids named anywhere in one evidence data payload."""
    ids: list[str] = []
    if not isinstance(data, dict):
        return ids
    guest_id = data.get("guest_id")
    if isinstance(guest_id, str) and guest_id:
        ids.append(guest_id)
    guests = data.get("guests")
    if isinstance(guests, list):
        for item in guests:
            if isinstance(item, dict):
                gid = item.get("guest_id")
                if isinstance(gid, str) and gid:
                    ids.append(gid)
    return ids


def scope_violations_in_evidence(
    evidence: list,
    asset_id: str | None = None,
    guest_id: str | None = None,
) -> list[dict]:
    """Out-of-scope data in collected evidence (L3, pure function).

    An evidence item violates the scope when its data names an
    ``asset_id`` different from the execution's, or a ``guest_id``
    different from the session's bound guest. Returns one violation
    per offending item (empty = in scope). Only the evidence *data*
    is checked — the prose answer is not scanned.
    """
    violations: list[dict] = []
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        data = item.get("data") or {}
        if asset_id:
            item_asset = data.get("asset_id") if isinstance(data, dict) else None
            if isinstance(item_asset, str) and item_asset and item_asset != asset_id:
                violations.append(
                    {
                        "type": "evidence_outside_asset_scope",
                        "asset_id": item_asset,
                        "source": str(item.get("source") or ""),
                    }
                )
        if guest_id:
            for gid in _guest_ids_in_data(data):
                if gid != guest_id:
                    violations.append(
                        {
                            "type": "evidence_outside_guest_scope",
                            "guest_id": gid,
                            "source": str(item.get("source") or ""),
                        }
                    )
    return violations
