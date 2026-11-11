"""Ontology canonical tables (L1) and projection queue."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ontology_entities",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("entity_key", sa.String(length=200), nullable=False),
        sa.Column("properties", sa.JSON(), nullable=True),
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("trace_id", sa.String(length=100), nullable=False),
        sa.Column("source_event_id", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("entity_type", "entity_key", name="uq_ontology_entities_key"),
    )
    op.create_index("ix_ontology_entities_type", "ontology_entities", ["entity_type"])

    op.create_table(
        "ontology_relationships",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("relationship_type", sa.String(length=64), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("source_key", sa.String(length=200), nullable=False),
        sa.Column("target_type", sa.String(length=64), nullable=False),
        sa.Column("target_key", sa.String(length=200), nullable=False),
        sa.Column("properties", sa.JSON(), nullable=True),
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("trace_id", sa.String(length=100), nullable=False),
        sa.Column("source_event_id", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "relationship_type",
            "source_type",
            "source_key",
            "target_type",
            "target_key",
            name="uq_ontology_relationships_edge",
        ),
    )
    op.create_index(
        "ix_ontology_relationships_source",
        "ontology_relationships",
        ["source_type", "source_key"],
    )

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

    op.create_table(
        "ontology_projection_state",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("subject_kind", sa.String(length=16), nullable=False),
        sa.Column("subject_ref", sa.String(length=300), nullable=False),
        sa.Column("canonical_version", sa.Integer(), nullable=False),
        sa.Column("projected_version", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("change_type", sa.String(length=16), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_enqueued_at", sa.DateTime(), nullable=True),
        sa.Column("last_projected_at", sa.DateTime(), nullable=True),
        sa.Column("trace_id", sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "subject_kind",
            "subject_ref",
            name="uq_ontology_projection_state_ref",
        ),
    )
    op.create_index(
        "ix_ontology_projection_state_status",
        "ontology_projection_state",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index("ix_ontology_projection_state_status", table_name="ontology_projection_state")
    op.drop_table("ontology_projection_state")
    op.drop_table("ontology_versions")
    op.drop_index("ix_ontology_relationships_source", table_name="ontology_relationships")
    op.drop_table("ontology_relationships")
    op.drop_index("ix_ontology_entities_type", table_name="ontology_entities")
    op.drop_table("ontology_entities")
