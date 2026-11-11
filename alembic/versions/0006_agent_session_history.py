"""Per-agent session-history limits: session_history on agents + agent_versions."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable JSON: {"max_turns", "max_tokens"} or NULL (global caps).
    # Pre-existing rows keep NULL; re-seed fills them from the agent
    # YAMLs and snapshots them into a new AgentVersion.
    op.add_column(
        "agents", sa.Column("session_history", sa.JSON(), nullable=True)
    )
    op.add_column(
        "agent_versions", sa.Column("session_history", sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("agent_versions", "session_history")
    op.drop_column("agents", "session_history")
