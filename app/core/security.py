"""API-key auth — minimal stub for the scaffold.

The Platform API accepts an `X-API-Key` header compared (constant-time)
against the `API_KEY` env var. If `API_KEY` is empty, auth is DISABLED —
this is a dev-only convenience and must never be used in production.

Production should replace this with JWT/OAuth2 behind the Policy Manager.
"""

from __future__ import annotations

import logging
import secrets

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
