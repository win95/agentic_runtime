"""Ontology L1 becomes state-only (design 2026-11-17, section 3.4).

- DROP COLUMN ``properties`` on ``ontology_entities`` and
  ``ontology_relationships``: object data now lives on Neo4j; L1 keeps
  only state (status, version, trace/provenance, projection state).
- DROP COLUMN ``state_hash`` on both tables: it hashed
  (properties + status), and with no properties left it has no
  meaning — change detection now compares against the Neo4j node.
- DROP TABLE ``ontology_versions``: full-object snapshots are gone by
  design (tran's decision 2); only version/change_type/trace remain.

DESTRUCTIVE, NO AUTOMATIC ROLLBACK: run only after the L1 -> Neo4j
backfill (``scripts/backfill_l1_to_neo4j.py``) has run AND been verified,
with a Postgres backup (``pg_dump``) taken first. ``downgrade()`` restores
the schema only (empty ``properties`` columns, empty
``ontology_versions`` table) — the dropped object data is NOT recovered.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0015_ontology_state_only"
down_revision: str | None = "0014_write_idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite cannot DROP COLUMN directly; batch mode (copy-and-move
    # there, plain ALTER on PostgreSQL) keeps this migration portable —
    # the migration test runs against a scratch SQLite DB.
    with op.batch_alter_table("ontology_entities", recreate="always") as batch_op:
        batch_op.drop_column("properties")
        batch_op.drop_column("state_hash")
    with op.batch_alter_table("ontology_relationships", recreate="always") as batch_op:
        batch_op.drop_column("properties")
        batch_op.drop_column("state_hash")
    # DROP TABLE also drops the table's indexes (including
    # ix_ontology_versions_subject_version from 0004) on both backends.
    op.drop_table("ontology_versions")


def downgrade() -> None:
    # Schema-only restore: the object data dropped by upgrade() is gone
    # for good. Recreate the versions table WITH its 0004 index so the
    # 0004 downgrade (which drops that index) still works on the way
    # down to base.
    op.create_table(
        "ontology_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("subject_kind", sa.String(length=16), nullable=False),
        sa.Column("subject_ref", sa.String(length=300), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("change_type", sa.String(length=16), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=True),
        sa.Column("trace_id", sa.String(length=100), nullable=False),
        sa.Column("source_event_id", sa.String(length=100), nullable=True),
        sa.Column("transaction_time", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "subject_kind",
            "subject_ref",
            "version",
            name="uq_ontology_versions_ref_version",
        ),
    )
    op.create_index(
        "ix_ontology_versions_subject_version",
        "ontology_versions",
        ["subject_ref", sa.text("version DESC")],
    )
    with op.batch_alter_table("ontology_entities", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("properties", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("state_hash", sa.String(length=64), nullable=True))
    with op.batch_alter_table("ontology_relationships", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("properties", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("state_hash", sa.String(length=64), nullable=True))
