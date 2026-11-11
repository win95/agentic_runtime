"""Session -> OpenSandbox mapping (docs/sandbox.md)."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "session_sandboxes",
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("sandbox_id", sa.String(128), nullable=False),
        sa.Column("agent_id", sa.String(128), nullable=True),
        sa.Column("mission_id", sa.String(128), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("image", sa.String(256), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("session_id"),
        sa.UniqueConstraint("sandbox_id"),
    )


def downgrade() -> None:
    op.drop_table("session_sandboxes")
