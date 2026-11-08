"""One-command seed: missions + Postgres demo rows + Qdrant docs + FalkorDB.

Usage:
    python -m seeds.seed_demo_data
    # or via compose (seed service, profile `setup`):
    docker compose --profile setup run --rm seed

Postgres + missions are required and fail the command on error. Qdrant and
FalkorDB are best-effort: a failure prints a clear WARNING (usually a
missing LiteLLM key/proxy or a service that is not up yet) and the
command still exits 0 — re-run it after fixing the cause; everything is
idempotent, so nothing is duplicated.
"""

from __future__ import annotations

import logging
import sys

from app.core.db import SessionLocal, init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    init_db()

    from seeds.demo_data import seed_falkor_demo, seed_postgres_demo, seed_qdrant_demo
    from seeds.load_missions import main as seed_missions

    seed_missions()
    print("Missions: seeded (fault_analysis)")

    db = SessionLocal()
    try:
        counts = seed_postgres_demo(db)
    finally:
        db.close()
    print(
        f"PostgreSQL: seeded {counts['faults']} fault rows + "
        f"{counts['maintenance']} maintenance rows for asset O21"
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
