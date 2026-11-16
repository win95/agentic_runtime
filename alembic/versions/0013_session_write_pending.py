"""Session write pending — quote/confirm payloads per pipeline."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0013_session_write_pending"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "session_write_pending",
        sa.Column("session_id", sa.String(length=128), primary_key=True),
        sa.Column("pipeline_id", sa.String(length=64), primary_key=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="quoted"),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=64), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("session_write_pending")