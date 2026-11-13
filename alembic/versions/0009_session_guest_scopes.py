"""Session -> guest binding for the hotel data scope (Layer 1/L2).

One session is one guest: the first guest a session touches binds the
session to that guest_id, and later guest/booking calls in the same
session must name the same guest. The binding is written by the guest
scope wrapper in ``runtime/tools/registry.py`` (first sight of a
guest_id in a find_guest / register_guest / get_booking / create_booking
call or result) and read by the same wrapper plus ``validate_results``.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "session_guest_scopes",
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("guest_id", sa.String(128), nullable=False),
        sa.Column("guest_phone", sa.String(64), nullable=True),
        sa.Column("mission_id", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("session_id"),
    )


def downgrade() -> None:
    op.drop_table("session_guest_scopes")
