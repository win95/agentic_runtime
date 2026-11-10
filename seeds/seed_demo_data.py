"""One-command seed: missions + agents + schedules + Postgres demo rows
+ Qdrant docs + FalkorDB relations.

Usage:
    python -m seeds.seed_demo_data [--dry-run]
    make seed
    # or via compose (seed service, profile `setup`):
    docker compose --profile setup run --rm seed

Postgres + missions are required and fail the command on error. Qdrant and
FalkorDB are best-effort: a failure prints a clear WARNING (usually a
missing LiteLLM key/proxy or a service that is not up yet) and the
command still exits 0 — re-run it after fixing the cause; everything is
idempotent, so nothing is duplicated.

--dry-run validates the seed data's own invariants and prints the
counts without touching any store.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.core.db import SessionLocal, init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed demo data (asset O21, Oct 2026).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the seed data and print counts without touching any store",
    )
    args = parser.parse_args(argv)

    from seeds.demo_data import validate

    counts = validate()
    if args.dry_run:
        print(
            f"Dry run OK: {counts['faults']} faults + {counts['maintenance']} "
            f"maintenance rows + {counts['sensor_readings']} sensor readings + "
            f"{counts['documents']} documents + "
            f"{counts['relations']} relations for asset O21 (October 2026). "
            "No store was touched."
        )
        return 0

    init_db()

    from seeds.demo_data import (
        seed_falkor_demo,
        seed_policies_demo,
        seed_postgres_demo,
        seed_qdrant_demo,
        seed_tools_demo,
    )
    from seeds.load_missions import main as seed_missions

    seed_missions()
    print("Missions/agents/schedules: seeded (fault_analysis, system_admin)")

    db = SessionLocal()
    try:
        counts = seed_postgres_demo(db)
        tools = seed_tools_demo(db)
        policies = seed_policies_demo(db)
    finally:
        db.close()
    print(
        f"PostgreSQL: seeded {counts['faults']} fault rows + "
        f"{counts['maintenance']} maintenance rows + "
        f"{counts['sensor_readings']} sensor readings for asset O21; "
        f"{tools['tools']} tool rows + {policies['policies']} policy row"
    )

    try:
        qdrant = seed_qdrant_demo()
        print(
            f"Qdrant: collection {qdrant['collection']} (dim={qdrant['dim']}), "
            f"upserted={qdrant['upserted']}, skipped={qdrant['skipped']}"
        )
    except Exception as exc:  # noqa: BLE001 — best-effort, reported clearly
        print(
            f"WARNING: Qdrant seed skipped: {exc}\n"
            "  Fix the LiteLLM key/proxy or start Qdrant, then re-run this seed."
        )

    try:
        falkor = seed_falkor_demo()
        print(f"FalkorDB: seeded {falkor['relations']} relations for asset O21")
    except Exception as exc:  # noqa: BLE001 — best-effort, reported clearly
        print(
            f"WARNING: FalkorDB seed skipped: {exc}\n"
            "  Start FalkorDB, then re-run this seed."
        )

    print("Seed complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
