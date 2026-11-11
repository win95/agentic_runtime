"""CLI: drain ontology_projection_state queue into Neo4j."""

from __future__ import annotations

import argparse
import sys

from app.config import settings
from app.core.db import SessionLocal
from ontology.projection.engine import drain_pending_projections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drain ontology projection queue.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max rows to drain (default: projection_drain_batch_size)",
    )
    args = parser.parse_args(argv)
    limit = args.limit if args.limit is not None else settings.projection_drain_batch_size
    db = SessionLocal()
    try:
        result = drain_pending_projections(db, limit=limit)
        if result.get("detail") == "neo4j_unhealthy":
            print(
                "WARNING: Neo4j unhealthy; projection drain skipped "
                f"(processed={result.get('processed')} "
                f"errors={result.get('errors')})",
                file=sys.stderr,
            )
        print(result)
        return 0 if result.get("errors", 0) == 0 else 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
