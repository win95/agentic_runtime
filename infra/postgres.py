"""PostgreSQL access — reuses the app's SQLAlchemy session (Layer 6).

Repositories that need raw SQL use `run_query` here instead of opening
their own connections.
"""

from __future__ import annotations

from sqlalchemy import text

from app.core.db import SessionLocal, engine, get_db

__all__ = ["SessionLocal", "engine", "get_db", "run_query"]


def run_query(sql: str, params: dict | None = None) -> list[dict]:
    """Run a read-only SQL query and return rows as dicts.

    Raises on any DB error (missing table, connection failure, ...). Callers
    in `infra/repositories` catch this and fall back to stub data.
    """
    session = SessionLocal()
    try:
        result = session.execute(text(sql), params or {})
        return [dict(row) for row in result.mappings()]
    finally:
        session.close()
