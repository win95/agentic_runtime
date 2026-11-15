"""Document Management tables: managed documents + ACL grants.

``documents`` holds one row per version (never overwritten — a new
upload is a new row with version + 1 and a new MinIO object).
``document_acls`` holds explicit grants; the owner needs no row.
See ``app/services/document_service.py`` and ``docs/documents.md``.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("doc_id", sa.String(128), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("title", sa.String(256), nullable=False, server_default=""),
        sa.Column("filename", sa.String(256), nullable=False, server_default=""),
        sa.Column("object_key", sa.String(512), nullable=False, server_default=""),
        sa.Column(
            "content_type",
            sa.String(128),
            nullable=False,
            server_default="application/octet-stream",
        ),
        sa.Column("content_hash", sa.String(64), nullable=False, server_default=""),
        sa.Column("size_bytes", sa.Integer, nullable=False, server_default="0"),
        sa.Column("owner_user_id", sa.String(128), nullable=True),
        sa.Column("asset_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=True),
        sa.UniqueConstraint("doc_id", "version", name="uq_documents_doc_id_version"),
    )
    op.create_index("ix_documents_doc_id", "documents", ["doc_id"])
    op.create_index("ix_documents_owner_user_id", "documents", ["owner_user_id"])
    op.create_table(
        "document_acls",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("doc_id", sa.String(128), nullable=False),
        sa.Column("principal_user_id", sa.String(128), nullable=False),
        sa.Column("permission", sa.String(16), nullable=False, server_default="read"),
        sa.Column("created_at", sa.DateTime, nullable=True),
        sa.UniqueConstraint(
            "doc_id",
            "principal_user_id",
            "permission",
            name="uq_document_acls_doc_principal_permission",
        ),
    )
    op.create_index("ix_document_acls_doc_id", "document_acls", ["doc_id"])


def downgrade() -> None:
    op.drop_index("ix_document_acls_doc_id", table_name="document_acls")
    op.drop_table("document_acls")
    op.drop_index("ix_documents_owner_user_id", table_name="documents")
    op.drop_index("ix_documents_doc_id", table_name="documents")
    op.drop_table("documents")
