"""Auth routes: register, login, me.

Login failures always return the same 401, whether the email is
unknown, the password is wrong, or the account is disabled — the API
never tells a caller which one it was.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.user_service import schemas
from services.user_service.auth import (
    create_access_token,
    hash_password,
    verify_password,
)
from services.user_service.config import settings
from services.user_service.db import get_db
from services.user_service.dependencies import get_current_user
from services.user_service.models import User

router = APIRouter(tags=["auth"])


@router.post("/auth/register", response_model=schemas.UserRead, status_code=201)
def register(payload: schemas.RegisterRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    if db.query(User).filter(User.email == email).first() is not None:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(email=email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/auth/login", response_model=schemas.TokenResponse)
def login(payload: schemas.LoginRequest, db: Session = Depends(get_db)):
    if not settings.jwt_secret:
        raise HTTPException(status_code=503, detail="JWT auth is not configured")
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password, user.password_hash)
    ):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(
        user_id=user.id,
        email=user.email,
        role=user.role,
        secret=settings.jwt_secret,
        expires_hours=settings.jwt_expire_hours,
    )
    return schemas.TokenResponse(
        access_token=token, expires_in=settings.jwt_expire_hours * 3600
    )


@router.get("/auth/me", response_model=schemas.UserRead)
def me(user: User = Depends(get_current_user)):
    return user
