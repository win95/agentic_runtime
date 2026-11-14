"""Pydantic schemas for the user service API.

Email validation is a plain regex on purpose: ``EmailStr`` would pull
in the ``email-validator`` package, which the platform does not use.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

_EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


class RegisterRequest(BaseModel):
    email: str = Field(max_length=255, pattern=_EMAIL_PATTERN)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(max_length=255, pattern=_EMAIL_PATTERN)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    # Seconds until the access token expires.
    expires_in: int


class UserRead(BaseModel):
    id: str
    email: str
    role: str
    is_active: bool
    created_at: datetime | None = None


class UserUpdate(BaseModel):
    is_active: bool
