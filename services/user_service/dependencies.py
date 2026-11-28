"""Auth dependencies for the user service's own routes.

Bearer JWT -> User row (must exist and be active). Admin routes add a
role check on top. A disabled user fails auth the same way an unknown
token does — no information leaks about which one it was.
"""

from __future__ import annotations

import secrets

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


def require_service_key(
    x_service_key: str | None = Header(default=None),
) -> None:
    """Service-to-service auth for the platform (agent_lark_me sync).

    The platform calls the user service with a shared service key
    (``X-Service-Key: <USER_SERVICE_API_KEY>``) — NOT a minted admin
    JWT: ``get_current_user`` requires the JWT ``sub`` to be an
    existing User row, and the platform cannot know an admin's id.
    The key is fail-closed: an empty ``USER_SERVICE_API_KEY`` refuses
    every call, in every environment. The value must match in both
    services' env, like JWT_SECRET.
    """
    expected = settings.service_api_key
    if not expected:
        raise HTTPException(
            status_code=401, detail="USER_SERVICE_API_KEY is not configured"
        )
    if not x_service_key or not secrets.compare_digest(x_service_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing service key")
