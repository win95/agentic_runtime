"""Sessions (Option B): session_id on executions + index."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable: pre-existing rows keep NULL; the service auto-generates
    # a session id for every execution created after this migration.
    op.add_column(
        "executions", sa.Column("session_id", sa.String(128), nullable=True)
    )
    # Session-history lookups filter on session_id and order by
    # created_at; the index serves the filter (the sort is small).
    op.create_index(
        "ix_executions_session_id", "executions", ["session_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_executions_session_id", table_name="executions")
    op.drop_column("executions", "session_id")
