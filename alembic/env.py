"""Alembic environment — migrations for the platform's PostgreSQL schema.

The database URL comes from the app settings (``DATABASE_URL`` env var,
see ``app/config.py``), never from alembic.ini — the ini file's
``sqlalchemy.url`` is only a placeholder. ``target_metadata`` is the
SQLAlchemy metadata from ``app.models`` so autogenerate sees every
table.

Two commands matter:

- Fresh database: ``alembic upgrade head`` (or ``make migrate``).
- Existing database created by ``init_db()`` / ``create_all`` (i.e. any
  database from before migrations existed): ``alembic stamp head``
  ONCE, which records the baseline as already applied without running
  it. Later migrations then apply normally with ``upgrade head``.
  Stamping is safe only because the baseline migration (0001) is the
  exact schema ``create_all`` produced; do NOT stamp a database whose
  tables were hand-altered.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# The repo root (parent of alembic/) must be importable so `app.*`
# resolves when alembic runs from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The live URL always wins over the placeholder in alembic.ini.
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL, no DBAPI connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (live DBAPI connection)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
