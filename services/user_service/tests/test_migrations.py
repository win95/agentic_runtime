"""User-service migration chain — upgrade head on a scratch SQLite DB.

Runs the real Alembic API (same code `make migrate-users` runs) with
the service's DATABASE_URL pointed at a tmp file, so the test proves
the chain works end-to-end rather than only checking that the files
exist.
"""

import sqlite3

from alembic.config import Config

from alembic import command
from services.user_service.config import settings


def _cfg(tmp_path, monkeypatch):
    db_path = tmp_path / "users-migrate-test.db"
    # env.py takes the URL from settings, so point settings at the file.
    monkeypatch.setattr(
        settings, "user_service_database_url", f"sqlite:///{db_path}"
    )
    cfg = Config("services/user_service/alembic.ini")
    return cfg, db_path


def test_user_service_migrations_upgrade_head(tmp_path, monkeypatch):
    cfg, db_path = _cfg(tmp_path, monkeypatch)
    command.upgrade(cfg, "head")

    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute(
            "SELECT version_num FROM alembic_version_user_service"
        ).fetchall()
        assert version == [("0001",)]
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"users", "alembic_version_user_service"} <= tables
        cols = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
        assert {"id", "email", "password_hash", "role", "is_active"} <= set(cols)
    finally:
        conn.close()

    command.downgrade(cfg, "base")
    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert tables == {"alembic_version_user_service"}
    finally:
        conn.close()
