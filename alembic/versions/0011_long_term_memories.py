"""Long-term memory table: turns the context budget trimmed away.

One row per trimmed turn (see ``runtime/long_term_memory.py``):
``HookMiddleware.before_model`` encodes turns removed by history
trimming, and the session-history loader encodes turns dropped by
the turn/token cap. A later execution in the same session retrieves
matching turns by keyword into a 1,024-token block.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "long_term_memories",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("execution_id", sa.String(128), nullable=True),
        sa.Column("turn_index", sa.Integer, nullable=True),
        sa.Column("source", sa.String(32), nullable=False, server_default="trim"),
        sa.Column("keywords", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("text", sa.Text, nullable=False, server_default=""),
        sa.Column("content_hash", sa.String(64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime, nullable=True),
    )
    op.create_index(
        "ix_long_term_memories_session_id", "long_term_memories", ["session_id"]
    )
    op.create_index(
        "ix_long_term_memories_session_hash",
        "long_term_memories",
        ["session_id", "execution_id", "content_hash"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_long_term_memories_session_hash", table_name="long_term_memories"
    )
    op.drop_index("ix_long_term_memories_session_id", table_name="long_term_memories")
    op.drop_table("long_term_memories")
