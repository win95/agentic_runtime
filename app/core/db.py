"""Database engine / session.

PostgreSQL in production (see DATABASE_URL). SQLite is supported for tests
only — detected by URL scheme and run with a StaticPool so an in-memory DB
is shared across sessions in the same process.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.models import Base


def _make_engine():
    url = settings.database_url
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
    """Schema is owned by Alembic — do not create tables on Postgres.

    SQLite-only convenience for local scripts/tests that are not
    migrated; production and CI Postgres must run ``alembic upgrade head``.
    """
    url = settings.database_url
    if url.startswith("sqlite"):
        Base.metadata.create_all(bind=engine)
        return
    import logging

    logging.getLogger(__name__).debug(
        "init_db skipped for non-SQLite URL (use Alembic migrations)"
    )


def get_session() -> Session:
    return SessionLocal()


def get_db():
    """FastAPI dependency yielding a session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
