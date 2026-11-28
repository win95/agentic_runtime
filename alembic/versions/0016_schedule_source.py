"""Schedule provenance — ``source`` column on ``schedules`` (Feature B).

Records where a Schedule row came from: ``"loop"`` (materialized
from a harness loop at seed time) or ``"file"`` (seeded from
``seeds/schedules/*.yaml``). NULL means legacy / API-created /
manually-created rows whose provenance is unknown. The seed guards
in ``seeds/load_missions.py`` refuse cross-provenance overwrites
(loop-over-file and file-over-loop fail loudly); a NULL row is
overwritten with a warning and takes on the new source.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0016_schedule_source"
down_revision: str | None = "0015_ontology_state_only"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite cannot ADD COLUMN with a non-NULL default directly in
    # older versions and batch mode keeps this migration portable —
    # the migration test runs against a scratch SQLite DB. The
    # column is nullable, so existing rows simply get NULL.
    with op.batch_alter_table("schedules", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("source", sa.String(length=64), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("schedules", recreate="always") as batch_op:
        batch_op.drop_column("source")
