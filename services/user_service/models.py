"""User model — owned by the user service, no other service imports this.

Roles are a plain string column (``user`` | ``admin``) on purpose: the
base needs exactly two roles, and adding a roles table before any
feature needs it would be speculative.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from services.user_service.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Lowercased at write time; unique so one email is one account.
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    # argon2 hash, never a plaintext password.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    # "user" | "admin"
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="user")
    # False = login refused (admin-disabled); the row is kept for audit.
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )


class UserIdentity(Base):
    """One external login linked to a user (agent_lark_me, D-LARK-11).

    ``provider`` is ``"lark"`` for now; ``external_id`` is the Lark
    ``open_id``. One (provider, external_id) pair maps to exactly one
    user — the pair is unique, so a sync for the same Lark user always
    returns the same user id. A user may hold several identities (a
    Lark link + a password login, ...).
    """

    __tablename__ = "user_identities"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("provider", "external_id", name="uq_user_identities_provider_external"),
    )
