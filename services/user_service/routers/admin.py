"""Admin routes: list users, enable / disable a user.

An admin cannot disable their own account — that would lock the system
out of its last admin with no way back except direct DB access.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.user_service import schemas
from services.user_service.db import get_db
from services.user_service.dependencies import require_admin
from services.user_service.models import User

router = APIRouter(tags=["admin"])


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
