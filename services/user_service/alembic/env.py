"""Alembic environment for the user service — its own migration chain.

The database URL comes from the service settings
(``USER_SERVICE_DATABASE_URL``), never from alembic.ini — the ini
file's ``sqlalchemy.url`` is only a placeholder. ``target_metadata`` is
the service's own SQLAlchemy metadata, so autogenerate sees only the
service's tables.

The service records its migration state in its own version table
(``alembic_version_user_service``): the platform app uses
``alembic_version_agentic_runtime`` in the same Postgres server, and
MLflow owns the default ``alembic_version`` table — three chains must
never share one table.

Run from ``services/user_service`` (or ``make migrate-users``):
``alembic upgrade head``.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# The repo root (three levels up from alembic/) must be importable so
# `services.user_service.*` resolves when alembic runs from anywhere.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from services.user_service.config import settings  # noqa: E402
from services.user_service.db import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The live URL always wins over the placeholder in alembic.ini.
config.set_main_option("sqlalchemy.url", settings.user_service_database_url)

target_metadata = Base.metadata

VERSION_TABLE = "alembic_version_user_service"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL, no DBAPI connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
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
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=VERSION_TABLE,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
