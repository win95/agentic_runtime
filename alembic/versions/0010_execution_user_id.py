"""Execution owner: user_id from the user service's JWT (nullable).

A logged-in user creates executions that carry their user id; service
API-key calls leave it NULL. The executions router filters by it, so a
user only ever sees their own runs.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("executions", sa.Column("user_id", sa.String(128), nullable=True))
    op.create_index("ix_executions_user_id", "executions", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_executions_user_id", table_name="executions")
    op.drop_column("executions", "user_id")
