"""Ontology indexes: latest-version lookup + relationship target lookup."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # get_latest_version filters on subject_ref and orders by version DESC.
    op.create_index(
        "ix_ontology_versions_subject_version",
        "ontology_versions",
        ["subject_ref", sa.text("version DESC")],
    )
    # Relationship lookups by target endpoint (e.g. "what points at X").
    op.create_index(
        "ix_ontology_relationships_target",
        "ontology_relationships",
        ["target_type", "target_key"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ontology_relationships_target", table_name="ontology_relationships"
    )
    op.drop_index(
        "ix_ontology_versions_subject_version", table_name="ontology_versions"
    )
