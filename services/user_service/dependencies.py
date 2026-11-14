"""Auth dependencies for the user service's own routes.

Bearer JWT -> User row (must exist and be active). Admin routes add a
role check on top. A disabled user fails auth the same way an unknown
token does — no information leaks about which one it was.
"""

from __future__ import annotations

import jwt
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from services.user_service.auth import decode_access_token
from services.user_service.config import settings
from services.user_service.db import get_db
from services.user_service.models import User


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if not settings.jwt_secret:
        raise HTTPException(status_code=401, detail="JWT auth is not configured")
    try:
        payload = decode_access_token(authorization[7:], secret=settings.jwt_secret)
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401, detail="Invalid or expired token"
        ) from None
    user = db.get(User, payload.get("sub"))
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user
