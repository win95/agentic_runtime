"""API-key auth — minimal stub for the scaffold.

The Platform API accepts an `X-API-Key` header compared (constant-time)
against the `API_KEY` env var. If `API_KEY` is empty, auth is DISABLED —
this is a dev-only convenience and must never be used in production.

Users authenticate with a JWT instead (see below): the user service
signs HS256 tokens with the shared `JWT_SECRET`, and this module
verifies them. A route can accept either credential via
`require_api_key_or_user`.
"""

from __future__ import annotations

import logging
import secrets

import jwt
from fastapi import Header, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)
_warned = False


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str | None:
    """Dependency: require a valid API key unless auth is disabled."""
    global _warned
    expected = settings.api_key
    if not expected:
        if not _warned:
            logger.warning("API_KEY is empty — API-key auth is DISABLED (dev only).")
            _warned = True
        return None
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


def decode_bearer_token(authorization: str | None) -> dict | None:
    """Verify a ``Bearer <JWT>`` header from the user service.

    Returns the token payload (``{"sub": user_id, "email", "role", ...}``)
    for a valid token, None when no Bearer token was sent or JWT auth is
    not configured (empty ``JWT_SECRET``). Raises 401 for a Bearer token
    that fails verification — a bad token must not silently fall through
    to the API-key path.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    if not settings.jwt_secret:
        return None
    try:
        return jwt.decode(
            authorization[7:], settings.jwt_secret, algorithms=["HS256"]
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def get_current_user(
    authorization: str | None = Header(default=None),
) -> dict | None:
    """Dependency: the JWT user payload, or None for a service-key call."""
    return decode_bearer_token(authorization)


async def require_api_key_or_user(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict | None:
    """Dependency: accept either a valid user JWT or the service API key.

    Returns the JWT payload for a user call, None for a service-key call.
    Raises 401 when neither credential is valid.
    """
    user = decode_bearer_token(authorization)
    if user is not None:
        return user
    await require_api_key(x_api_key)
    return None


def require_own_execution(execution, user: dict | None) -> None:
    """A JWT user may only touch their own executions.

    Deliberately 404 (not 403): the API never confirms that another
    user's execution exists. Service-key calls (user is None) are
    unrestricted, as before.
    """
    if user is not None and getattr(execution, "user_id", None) != user.get("sub"):
        raise HTTPException(status_code=404, detail="Execution not found")
