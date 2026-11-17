"""API-key auth + JWT user auth + mission authorization.

The Platform API accepts an `X-API-Key` header compared (constant-time)
against the `API_KEY` env var. If `API_KEY` is empty, auth is DISABLED —
but ONLY when `ENVIRONMENT=dev`. In any other environment an empty
`API_KEY` is a configuration error and every API-key check fails closed
(401). Production must set API_KEY (docker-compose requires it).

Users authenticate with a JWT instead (see below): the user service
signs HS256 tokens with the shared `JWT_SECRET`, and this module
verifies them. A route can accept either credential via
`require_api_key_or_user`.

A JWT user is not an operator: `require_mission_for_role` limits which
missions a plain user may run through POST /executions (missions with
``harness.security.user_runnable``; default false). Admins and service-key
calls are unrestricted.
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
    """Dependency: require a valid API key unless auth is disabled.

    Auth is disabled ONLY in dev (`ENVIRONMENT=dev`) with an empty
    `API_KEY`. In any other environment an empty `API_KEY` fails
    closed — a missing key is a deployment error, not an open door.
    """
    global _warned
    expected = settings.api_key
    if not expected:
        if settings.environment != "dev":
            raise HTTPException(
                status_code=401,
                detail="API_KEY is not configured (ENVIRONMENT != dev)",
            )
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


async def require_jwt_user(
    authorization: str | None = Header(default=None),
) -> dict:
    """Dependency: a valid user JWT — API keys are NOT accepted.

    Guest endpoints use this: a service key must never stand in for
    a guest, or one caller could read every guest's sessions.
    """
    user = decode_bearer_token(authorization)
    if user is None:
        raise HTTPException(
            status_code=401, detail="Missing or invalid Bearer token"
        )
    return user


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


def require_mission_for_role(user: dict | None, mission_id: str) -> None:
    """Refuse a mission the caller's role may not run.

    Deliberately 404 (not 403), like `require_own_execution`: the API
    never confirms that a mission outside the caller's scope exists.
    """
    if user is None:
        return
    if user.get("role", "user") == "admin":
        return
    from runtime.harness.registry import user_runnable_mission_ids

    if mission_id not in user_runnable_mission_ids():
        raise HTTPException(status_code=404, detail="Mission not found")
