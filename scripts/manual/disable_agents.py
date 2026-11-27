"""Delete the archived agents' rows from the platform DB.

Companion to archive/README.md (agents disabled 2026-11-27:
fault_investigator / fault_analysis and hotel_receptionist /
hotel_reception). Removing the code alone does NOT disable them:
``resolve_agent`` falls back to ``BaseAgent`` and ``make seed`` never
deletes a mission whose YAML is gone — so the rows must go, or the
missions keep running degraded (hotel without its guest scoping).

Deletes, in foreign-key order: execution results -> executions ->
schedules -> agent versions -> agents -> missions. Execution *history*
for these two missions is deleted with them (``Execution.mission_id``
is a foreign key, so the mission row cannot go while history points
at it). Long-term memory, sandboxes and guest scopes are keyed by
session, not mission, and are left alone.

Usage (from the repo root, against the platform DATABASE_URL):

    python -m scripts.manual.disable_agents --dry-run   # show counts only
    python -m scripts.manual.disable_agents --apply     # delete + commit

Idempotent: re-running deletes 0 rows.
"""

from __future__ import annotations

import argparse
import sys

MISSION_IDS = ("fault_analysis", "hotel_reception")
AGENT_IDS = ("fault_investigator", "hotel_receptionist")


def plan(db) -> dict[str, int]:
    """Row counts that --apply would delete, per table."""
    from sqlalchemy import func, select

    from app import models

    counts: dict[str, int] = {}
    counts["execution_results"] = db.scalar(
        select(func.count())
        .select_from(models.ExecutionResult)
        .join(
            models.Execution,
            models.ExecutionResult.execution_id == models.Execution.id,
        )
        .where(models.Execution.mission_id.in_(MISSION_IDS))
    )
    counts["executions"] = db.scalar(
        select(func.count())
        .select_from(models.Execution)
        .where(models.Execution.mission_id.in_(MISSION_IDS))
    )
    counts["schedules"] = db.scalar(
        select(func.count())
        .select_from(models.Schedule)
        .where(models.Schedule.mission_id.in_(MISSION_IDS))
    )
    counts["agent_versions"] = db.scalar(
        select(func.count())
        .select_from(models.AgentVersion)
        .where(models.AgentVersion.agent_id.in_(AGENT_IDS))
    )
    counts["agents"] = db.scalar(
        select(func.count())
        .select_from(models.Agent)
        .where(models.Agent.id.in_(AGENT_IDS))
    )
    counts["missions"] = db.scalar(
        select(func.count())
        .select_from(models.Mission)
        .where(models.Mission.id.in_(MISSION_IDS))
    )
    return counts


def apply(db) -> dict[str, int]:
    """Delete the rows; returns the deleted counts."""
    from app import models

    deleted: dict[str, int] = {}
    result_ids = (
        db.query(models.ExecutionResult.id)
        .join(
            models.Execution,
            models.ExecutionResult.execution_id == models.Execution.id,
        )
        .filter(models.Execution.mission_id.in_(MISSION_IDS))
        .all()
    )
    ids = [r[0] for r in result_ids]
    if ids:
        deleted["execution_results"] = (
            db.query(models.ExecutionResult)
            .filter(models.ExecutionResult.id.in_(ids))
            .delete(synchronize_session=False)
        )
    else:
        deleted["execution_results"] = 0
    deleted["executions"] = (
        db.query(models.Execution)
        .filter(models.Execution.mission_id.in_(MISSION_IDS))
        .delete(synchronize_session=False)
    )
    deleted["schedules"] = (
        db.query(models.Schedule)
        .filter(models.Schedule.mission_id.in_(MISSION_IDS))
        .delete(synchronize_session=False)
    )
    deleted["agent_versions"] = (
        db.query(models.AgentVersion)
        .filter(models.AgentVersion.agent_id.in_(AGENT_IDS))
        .delete(synchronize_session=False)
    )
    deleted["agents"] = (
        db.query(models.Agent)
        .filter(models.Agent.id.in_(AGENT_IDS))
        .delete(synchronize_session=False)
    )
    deleted["missions"] = (
        db.query(models.Mission)
        .filter(models.Mission.id.in_(MISSION_IDS))
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be deleted, delete nothing")
    parser.add_argument("--apply", action="store_true",
                        help="delete the rows and commit")
    args = parser.parse_args(argv)
    if args.apply and args.dry_run:
        parser.error("--dry-run and --apply are mutually exclusive")
    if not args.apply and not args.dry_run:
        parser.error("pass --dry-run or --apply")

    from app.core.db import SessionLocal

    db = SessionLocal()
    try:
        counts = plan(db)
        total = sum(counts.values())
        for table, n in counts.items():
            print(f"  {table}: {n}")
        if args.dry_run:
            print(f"dry-run: {total} rows would be deleted; nothing changed.")
            return 0
        deleted = apply(db)
        print(f"deleted {sum(deleted.values())} rows.")
        remaining = plan(db)
        if sum(remaining.values()):
            print(f"WARNING: {remaining} rows remain — delete incomplete.")
            return 1
        print("done: no rows remain for fault_analysis / hotel_reception "
              "and their agents. Restart the worker + API.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
