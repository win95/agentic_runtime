"""Add tools.read_only and a unique (agent_id, version) on agent_versions.

- ``tools.read_only`` (Boolean, NOT NULL, default true): the Tool model
  never had this column, so GET /tools/db returned the schema default
  (True) for every tool — including the admin write tools. Existing rows
  are backfilled to true by the server default; re-running ``make seed``
  then corrects the write tools to false from the registry.
- ``uq_agent_versions_agent_version``: without it two snapshots could
  claim the same version number for one agent, breaking rollback /
  activation. The version service creates versions as max+1, so no
  duplicates should exist; if this migration fails on a duplicate,
  delete the extra row first.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tools",
        sa.Column("read_only", sa.Boolean(), server_default=sa.true(), nullable=False),
    )
    # SQLite cannot ALTER a constraint in place; batch mode (copy-and-move
    # there, plain ALTER on PostgreSQL) keeps this migration portable.
    with op.batch_alter_table("agent_versions", recreate="always") as batch_op:
        batch_op.create_unique_constraint(
            "uq_agent_versions_agent_version",
            ["agent_id", "version"],
        )


def downgrade() -> None:
    with op.batch_alter_table("agent_versions", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "uq_agent_versions_agent_version", type_="unique"
        )
    op.drop_column("tools", "read_only")
