"""Database engine / session for the user service.

Own engine, own Base, own tables — nothing is shared with the platform
app's ``app.core.db`` / ``app.models``. SQLite is supported for tests
only (StaticPool, same pattern as the platform).
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from services.user_service.config import settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    url = settings.user_service_database_url
    if url.startswith("sqlite"):
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    return create_engine(url, pool_pre_ping=True)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create tables if they do not exist (dev convenience).

    Production should use migrations instead of create_all.
    """
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()


def get_db():
    """FastAPI dependency yielding a session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
