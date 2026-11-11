"""Opt-in edge-case seeds — NOT part of `make seed`.

Usage:
    python -m seeds.seed_edge_cases [--dry-run]
    make seed-edge

Plants exactly ONE intentionally-poisoned projection row so a live
drain shows the poison path: the row's ``subject_ref`` cannot be
parsed by the drain (``_parse_relationship_ref`` needs 6 ``:``-separated
parts), so each drain attempt fails and after
``projection_max_attempts`` (default 5) the row goes ``dead``.

The row is planted by direct SQL/ORM insert, bypassing validation on
purpose: the canonical write service now rejects keys containing ``:``
(see ontology/canonical/write_service.py), so a poison row can only
exist as a legacy/direct-DB artifact. Re-running this module is
idempotent — it detects the existing poison row and plants nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.core.db import SessionLocal, init_db

logger = logging.getLogger(__name__)

POISON_SUBJECT_REF = "rel:seed-edge-poison"


def plant_poison_row(db) -> dict:
    """Plant the poison row if absent; return {"planted": bool, ...}."""
    from ontology.models import OntologyProjectionState

    existing = (
        db.query(OntologyProjectionState)
        .filter_by(subject_ref=POISON_SUBJECT_REF)
        .first()
    )
    if existing is not None:
        logger.info(
            "Poison projection row already present (status=%s, attempts=%d); "
            "nothing planted.",
            existing.status,
            existing.attempts,
        )
        return {"planted": False, "status": existing.status}
    db.add(OntologyProjectionState(
        subject_kind="relationship",
        subject_ref=POISON_SUBJECT_REF,
        canonical_version=1,
        projected_version=None,
        status="pending",
        change_type="update",
        attempts=0,
        trace_id="seed-edge-poison",
    ))
    db.commit()
    logger.info("Planted poison projection row %r (status=pending).", POISON_SUBJECT_REF)
    return {"planted": True, "status": "pending"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plant the edge-case poison projection row.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report whether the poison row is already present; plant nothing",
    )
    args = parser.parse_args(argv)

    # Dry-run never creates tables — it only reads. (Unlike
    # `seed_demo_data --dry-run`, this check needs a live DB read, so
    # an unreachable DB is a clean exit 1, not a traceback.)
    if not args.dry_run:
        init_db()
    db = SessionLocal()
    try:
        from ontology.models import OntologyProjectionState

        existing = (
            db.query(OntologyProjectionState)
            .filter_by(subject_ref=POISON_SUBJECT_REF)
            .first()
        )
    except Exception as exc:  # noqa: BLE001 — DB unreachable / tables missing
        print(f"Database unreachable — start Postgres first: {exc}")
        return 1
    try:
        if args.dry_run:
            if existing is None:
                print(
                    f"Dry run OK: poison row {POISON_SUBJECT_REF!r} not present; "
                    "would plant 1 row."
                )
            else:
                print(
                    f"Dry run OK: poison row already present "
                    f"(status={existing.status}); would plant 0 rows."
                )
            return 0
        result = plant_poison_row(db)
        if result["planted"]:
            print(
                f"Seeded edge case: 1 poison projection row ({POISON_SUBJECT_REF!r}). "
                "Run the drain 5 times (or wait for 5 beats) and it goes dead; "
                "check with python -m scripts.project_ontology_queue --once."
            )
        else:
            print(f"Edge case already seeded (status={result['status']}); nothing planted.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
