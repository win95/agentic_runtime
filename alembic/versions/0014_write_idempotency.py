"""write idempotency results — durable across processes"""

from alembic import op
import sqlalchemy as sa

revision = "0014_write_idempotency"
down_revision = "0013_session_write_pending"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "write_idempotency_results",
        sa.Column("idempotency_key", sa.String(length=64), primary_key=True),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("write_idempotency_results")