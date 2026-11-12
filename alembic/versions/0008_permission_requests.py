"""Human-approval requests for destructive tool calls (Option C, docs/hooks.md)."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "permission_requests",
        sa.Column("id", sa.String(128), nullable=False),
        sa.Column("execution_id", sa.String(128), nullable=True),
        sa.Column("mission_id", sa.String(128), nullable=True),
        sa.Column("agent_id", sa.String(128), nullable=True),
        sa.Column("tool_name", sa.String(64), nullable=False),
        sa.Column("target", sa.String(256), nullable=False),
        sa.Column("tool_input", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("decision_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("permission_requests")
