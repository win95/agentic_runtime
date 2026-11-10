"""Run fault_investigator through the real LLM (autonomous mode).

The seeded ``fault_analysis`` mission is hybrid, so a normal execution
usually stops in the deterministic capability graph and never calls the
LLM. This script creates a throwaway mission with the same input/output
schema but ``execution_config.mode = "autonomous"``, runs it, and then
deletes its executions + mission rows from Postgres (the API refuses to
delete a mission that has executions, by design).

Usage (from the repo root, with the API + worker running):

    python scripts/manual/fault_autonomous.py
    python scripts/manual/fault_autonomous.py --keep   # leave the mission

Expected today: the Deep Agent produces a real investigation summary
(it calls search_fault_history), but the execution ends FAILED with
``Result missing output-schema keys: ['root_causes', 'recommendations']``
— the agent outcome only fills answer/summary/actions_taken/evidence.
That is a product decision, documented in scripts/eval_mission.py.

The script needs the project venv (SQLAlchemy cleanup imports app.*).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _api import (  # noqa: E402
    call, get_result, print_result, require_worker, submit, wait_for,
)

TEMP_MISSION = "fault_auto_test"
O21_MONTH = {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"}

MISSION = {
    "id": TEMP_MISSION,
    "name": "Fault Auto Test",
    "objective": "Analyze equipment faults and identify root causes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "asset_id": {"type": "string"},
            "start_date": {"type": "string", "format": "date"},
            "end_date": {"type": "string", "format": "date"},
        },
        "required": ["asset_id", "start_date", "end_date"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "root_causes": {"type": "array"},
            "evidence": {"type": "array"},
            "recommendations": {"type": "array"},
        },
    },
    "execution_config": {"mode": "autonomous", "agent_id": "fault_investigator"},
    "capabilities": ["fault_history", "maintenance_analysis", "knowledge_retrieval"],
    "agent_id": "fault_investigator",
}


def cleanup() -> None:
    """Delete the temp mission's executions/results and the mission row."""
    from app import models
    from app.core.db import SessionLocal

    db = SessionLocal()
    try:
        execution_ids = [
            row.id for row in
            db.query(models.Execution).filter_by(mission_id=TEMP_MISSION)
        ]
        if execution_ids:
            db.query(models.ExecutionResult).filter(
                models.ExecutionResult.execution_id.in_(execution_ids)
            ).delete(synchronize_session=False)
        db.query(models.Execution).filter_by(mission_id=TEMP_MISSION).delete(
            synchronize_session=False
        )
        db.query(models.Mission).filter_by(id=TEMP_MISSION).delete(
            synchronize_session=False
        )
        db.commit()
        print(f"cleanup: removed mission {TEMP_MISSION} "
              f"and {len(execution_ids)} execution(s)")
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true",
                        help="keep the temp mission and its execution rows")
    parser.add_argument("--timeout", type=float, default=600,
                        help="seconds to wait for the run (default 600)")
    args = parser.parse_args()

    require_worker()
    status, payload = call("POST", "/missions", MISSION)
    if status == 409:
        print(f"mission {TEMP_MISSION} already exists — run cleanup first "
              f"(re-run without --keep) or delete it by hand")
        return 2
    if status != 201:
        print(f"create mission -> HTTP {status}: {payload}")
        return 2
    print(f"created mission {TEMP_MISSION} (mode=autonomous)")

    try:
        execution = submit(TEMP_MISSION, O21_MONTH)
        execution_id = execution["execution_id"]
        print(f"execution_id={execution_id} status={execution['status']}")
        execution, elapsed = wait_for(execution_id, timeout=args.timeout)
        print(f"final after {elapsed:.1f}s")
        print_result(execution, get_result(execution_id))
    finally:
        if not args.keep:
            cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
