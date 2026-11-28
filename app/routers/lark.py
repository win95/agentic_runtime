"""Lark user sync — POST /lark/users/sync (agent_lark_me, D-LARK-11).

The Lark frame calls this for every allowlisted sender (after the
allowlist check, before enqueue): the sender's Lark identity is
resolved to a platform user, creating + linking one on first contact
(lazy sync — no bulk import, no pre-fetch of the Lark user list).
The frame then passes the returned ``user_id`` into POST /executions,
so the run is owned by the right Lark user.

Auth: service API key only (``X-API-Key``) — a JWT user must never be
able to sync an arbitrary Lark identity to an arbitrary user.

The platform talks to the user service with a shared service key
(``X-Service-Key: <USER_SERVICE_API_KEY>``), not a minted admin JWT:
the user service requires a JWT ``sub`` to be an existing User row,
and the platform cannot know an admin's id.

The ``union_id`` is the tenant-stable identity (the same human keeps
it across Lark apps of the tenant); the ``open_id`` changes per app.
When the frame sends it, resolution is union first, then open_id —
and whichever identity resolved the user gets the other one linked to
it (POST /admin/users/{id}/identities), so the user_id stays stable
across an app switch instead of coming back as a brand-new user.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app import schemas
from app.config import settings
from app.core.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/lark",
    tags=["lark"],
    dependencies=[Depends(require_api_key)],
)

_USER_SERVICE_TIMEOUT_SECONDS = 5.0


def _user_service_headers() -> dict[str, str]:
    if not settings.user_service_api_key:
        raise HTTPException(
            status_code=503,
            detail="USER_SERVICE_API_KEY is not configured",
        )
    return {"X-Service-Key": settings.user_service_api_key}


def _user_service_get_by_identity(provider: str, external_id: str) -> dict | None:
    """The linked user, or None when the identity is unknown."""
    url = f"{settings.user_service_url.rstrip('/')}/admin/users/by-identity"
    try:
        resp = httpx.get(
            url,
            params={"provider": provider, "external_id": external_id},
            headers=_user_service_headers(),
            timeout=_USER_SERVICE_TIMEOUT_SECONDS,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502, detail=f"user service unreachable: {exc}"
        ) from exc
    if resp.status_code == 404:
        return None
    if resp.status_code == 403:
        # The user service refuses disabled accounts (P1-5) — pass
        # that through instead of masking it as a 502.
        raise HTTPException(status_code=403, detail="user account is disabled")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"user service lookup failed ({resp.status_code})",
        )
    return resp.json()


def _user_service_create_user(email: str, provider: str, external_id: str) -> dict:
    """Create a user + link the identity (idempotent on the service)."""
    url = f"{settings.user_service_url.rstrip('/')}/admin/users"
    try:
        resp = httpx.post(
            url,
            json={
                "email": email,
                "provider": provider,
                "external_id": external_id,
                "role": "user",
            },
            headers=_user_service_headers(),
            timeout=_USER_SERVICE_TIMEOUT_SECONDS,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502, detail=f"user service unreachable: {exc}"
        ) from exc
    if resp.status_code not in (200, 201):
        if resp.status_code == 403:
            raise HTTPException(
                status_code=403, detail="user account is disabled"
            )
        raise HTTPException(
            status_code=502,
            detail=f"user service create failed ({resp.status_code})",
        )
    return {"user": resp.json(), "created": resp.status_code == 201}


def _user_service_link_identity(user_id: str, provider: str, external_id: str) -> None:
    """Attach one more identity to a user (POST /admin/users/{id}/identities).

    Idempotent on the service (200 when already linked). A 409 means
    the identity belongs to a different user — the two accounts are
    NOT merged (that needs a manual decision), but the link is
    best-effort: the sync still serves the user it resolved, and the
    split is logged so it can be fixed by hand. This is the live
    state after an app switch: the old-app row and the new-app row
    each own one open_id, and neither may steal the other's.
    """
    url = f"{settings.user_service_url.rstrip('/')}/admin/users/{user_id}/identities"
    try:
        resp = httpx.post(
            url,
            json={"provider": provider, "external_id": external_id},
            headers=_user_service_headers(),
            timeout=_USER_SERVICE_TIMEOUT_SECONDS,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502, detail=f"user service unreachable: {exc}"
        ) from exc
    if resp.status_code in (200, 201):
        return
    if resp.status_code == 403:
        raise HTTPException(status_code=403, detail="user account is disabled")
    if resp.status_code == 409:
        logger.warning(
            "Lark identity %s:%s belongs to another user — not linking "
            "to user %s; accounts stay split until merged by hand",
            provider,
            external_id,
            user_id,
        )
        return
    raise HTTPException(
        status_code=502,
        detail=f"user service link failed ({resp.status_code})",
    )


@router.post("/users/sync", response_model=schemas.LarkUserSyncRead)
def sync_lark_user(payload: schemas.LarkUserSyncCreate):
    """Resolve a Lark sender to a platform user (lazy, idempotent).

    The union_id is the tenant-stable identity (same human, any Lark
    app of the tenant); the open_id changes per app. Resolution order
    is union first, then open_id, and whichever identity resolved the
    user gets the other one linked to it — unless that identity
    already belongs to a different user (the split left by an app
    switch): then the link is skipped with a warning and the sync
    still serves the resolved user. Without a union_id the flow is
    exactly the old open_id-only one.
    """
    provider = (payload.provider or "").strip()
    external_id = (payload.external_id or "").strip()
    if provider != "lark":
        raise HTTPException(
            status_code=422, detail=f"unsupported provider: {provider!r}"
        )
    if not external_id:
        raise HTTPException(status_code=422, detail="external_id must not be empty")
    union_id = (payload.union_id or "").strip() or None
    union_external_id = f"union:{union_id}" if union_id else None

    def _linked_or_403(linked: dict | None) -> dict | None:
        # Defense in depth (P1-5): the user service refuses disabled
        # accounts itself, but a stale one could still return one.
        if linked is not None and linked.get("is_active", True) is False:
            raise HTTPException(status_code=403, detail="user account is disabled")
        return linked

    if union_external_id:
        linked = _linked_or_403(
            _user_service_get_by_identity(provider, union_external_id)
        )
        if linked is not None:
            _user_service_link_identity(linked["id"], provider, external_id)
            return schemas.LarkUserSyncRead(
                user_id=linked["id"], email=linked["email"], created=False
            )
    linked = _linked_or_403(_user_service_get_by_identity(provider, external_id))
    if linked is not None:
        if union_external_id:
            _user_service_link_identity(linked["id"], provider, union_external_id)
        return schemas.LarkUserSyncRead(
            user_id=linked["id"], email=linked["email"], created=False
        )
    email = (payload.email or "").strip().lower()
    if not email:
        # D-LARK-12 default: synthetic address; real name/email
        # enrichment (contact scope) is a later step, not v1.
        email = f"lark-{external_id}@lark.local"
    created = _user_service_create_user(email, provider, external_id)
    user = created["user"]
    if union_external_id:
        _user_service_link_identity(user["id"], provider, union_external_id)
    logger.info(
        "Lark user synced: provider=%s user_id=%s created=%s",
        provider, user["id"], created["created"],
    )
    return schemas.LarkUserSyncRead(
        user_id=user["id"], email=user["email"], created=created["created"]
    )
