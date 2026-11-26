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
        raise HTTPException(
            status_code=502,
            detail=f"user service create failed ({resp.status_code})",
        )
    return {"user": resp.json(), "created": resp.status_code == 201}


@router.post("/users/sync", response_model=schemas.LarkUserSyncRead)
def sync_lark_user(payload: schemas.LarkUserSyncCreate):
    """Resolve a Lark sender to a platform user (lazy, idempotent)."""
    provider = (payload.provider or "").strip()
    external_id = (payload.external_id or "").strip()
    if provider != "lark":
        raise HTTPException(
            status_code=422, detail=f"unsupported provider: {provider!r}"
        )
    if not external_id:
        raise HTTPException(status_code=422, detail="external_id must not be empty")
    linked = _user_service_get_by_identity(provider, external_id)
    if linked is not None:
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
    logger.info(
        "Lark user synced: provider=%s user_id=%s created=%s",
        provider, user["id"], created["created"],
    )
    return schemas.LarkUserSyncRead(
        user_id=user["id"], email=user["email"], created=created["created"]
    )
