"""Password hashing (argon2) + JWT (PyJWT, HS256).

Both libraries are already pinned in the platform's requirements, so
the service adds no new dependency.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

_password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def create_access_token(
    *,
    user_id: str,
    email: str,
    role: str,
    secret: str,
    algorithm: str = "HS256",
    expires_hours: int = 24,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=expires_hours),
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_access_token(
    token: str, *, secret: str, algorithms: list[str] | None = None
) -> dict:
    """Decode + verify signature and expiry. Raises jwt.InvalidTokenError."""
    return jwt.decode(token, secret, algorithms=algorithms or ["HS256"])
