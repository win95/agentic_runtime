"""Admin routes: list users, enable / disable a user, create a user
with a linked external identity, look a user up by identity.

An admin cannot disable their own account — that would lock the system
out of its last admin with no way back except direct DB access.

The identity endpoints (POST /admin/users, GET /admin/users/by-identity)
are service-to-service only (X-Service-Key, see
services/user_service/dependencies.py): the platform's Lark sync calls
them. They are NOT for human admins — a human admin creates users
through /auth/register.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from services.user_service import schemas
from services.user_service.auth import hash_password
from services.user_service.db import get_db
from services.user_service.dependencies import require_admin, require_service_key
from services.user_service.models import User, UserIdentity

router = APIRouter(tags=["admin"])


def _require_active(user: User) -> None:
    """A disabled account must not be (re)linked or returned by sync.

    The JWT path rejects disabled users with 401 (dependencies.py);
    these service-key endpoints use 403 — the caller is
    authenticated, the target account is disabled.
    """
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is disabled")


@router.get("/admin/users", response_model=list[schemas.UserRead])
def list_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    return db.query(User).order_by(User.created_at).all()


@router.patch("/admin/users/{user_id}", response_model=schemas.UserRead)
def update_user(
    user_id: str,
    payload: schemas.UserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id and not payload.is_active:
        raise HTTPException(
            status_code=400, detail="An admin cannot disable their own account"
        )
    user.is_active = payload.is_active
    db.commit()
    db.refresh(user)
    return user


@router.post(
    "/admin/users", response_model=schemas.UserRead, status_code=201,
    dependencies=[Depends(require_service_key)],
)
def create_user_with_identity(
    payload: schemas.AdminUserCreate,
    response: Response,
    db: Session = Depends(get_db),
):
    """Create a user + link one external identity, idempotently.

    Called by the platform's Lark sync (agent_lark_me, D-LARK-11):
    a Lark user chatting with the bot gets a platform account on first
    contact. Idempotent — a second sync for the same identity returns
    the same user (200), so concurrent syncs cannot create duplicates:
    the (provider, external_id) pair and the email are both unique,
    and an IntegrityError on the insert falls back to re-reading the
    row the other request just wrote.
    """
    email = payload.email.strip().lower()
    provider = payload.provider.strip()
    external_id = payload.external_id.strip()
    if not provider or not external_id:
        raise HTTPException(
            status_code=422, detail="provider and external_id must not be empty"
        )
    existing = (
        db.query(UserIdentity)
        .filter(
            UserIdentity.provider == provider,
            UserIdentity.external_id == external_id,
        )
        .first()
    )
    if existing is not None:
        user = db.get(User, existing.user_id)
        if user is None:  # identity without a user — cannot happen (FK)
            raise HTTPException(status_code=500, detail="Identity has no user")
        _require_active(user)
        response.status_code = 200
        return user
    user = db.query(User).filter(User.email == email).first()
    if user is not None:
        # Email already registered (e.g. the user signed up by hand):
        # attach the identity to that account instead of a second one.
        # A disabled account is not linked — it stays unusable.
        _require_active(user)
        db.add(UserIdentity(provider=provider, external_id=external_id, user_id=user.id))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = (
                db.query(UserIdentity)
                .filter(
                    UserIdentity.provider == provider,
                    UserIdentity.external_id == external_id,
                )
                .first()
            )
            if existing is None:
                raise
            user = db.get(User, existing.user_id)
        response.status_code = 200
        return user
    # New user: a random unusable password hash — a synced user never
    # logs in with a password (login is via Lark, or not at all).
    user = User(
        email=email,
        password_hash=hash_password(secrets.token_hex(32)),
        role=payload.role if payload.role in ("user", "admin") else "user",
    )
    db.add(user)
    db.flush()  # assigns user.id before the identity row
    db.add(UserIdentity(provider=provider, external_id=external_id, user_id=user.id))
    try:
        db.commit()
    except IntegrityError:
        # Lost a race with a concurrent sync for the same user — the
        # other request's row is the answer; re-read it.
        db.rollback()
        existing = (
            db.query(UserIdentity)
            .filter(
                UserIdentity.provider == provider,
                UserIdentity.external_id == external_id,
            )
            .first()
        )
        if existing is None:
            raise
        user = db.get(User, existing.user_id)
        response.status_code = 200
        return user
    db.refresh(user)
    return user


@router.get(
    "/admin/users/by-identity",
    response_model=schemas.UserRead,
    dependencies=[Depends(require_service_key)],
)
def get_user_by_identity(
    provider: str,
    external_id: str,
    db: Session = Depends(get_db),
):
    """Look a user up by a linked external identity (Lark open_id)."""
    identity = (
        db.query(UserIdentity)
        .filter(
            UserIdentity.provider == provider.strip(),
            UserIdentity.external_id == external_id.strip(),
        )
        .first()
    )
    if identity is None:
        raise HTTPException(status_code=404, detail="Identity not found")
    user = db.get(User, identity.user_id)
    if user is None:
        raise HTTPException(status_code=500, detail="Identity has no user")
    _require_active(user)
    return user
