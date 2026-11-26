"""Add the user_identities table (agent_lark_me, D-LARK-11).

Links an external identity (provider + external_id, e.g. Lark open_id)
to a user. One pair maps to exactly one user.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_identities",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("external_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider", "external_id", name="uq_user_identities_provider_external"
        ),
    )
    op.create_index("ix_user_identities_provider", "user_identities", ["provider"])
    op.create_index("ix_user_identities_external_id", "user_identities", ["external_id"])
    op.create_index("ix_user_identities_user_id", "user_identities", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_user_identities_user_id", table_name="user_identities")
    op.drop_index("ix_user_identities_external_id", table_name="user_identities")
    op.drop_index("ix_user_identities_provider", table_name="user_identities")
    op.drop_table("user_identities")
