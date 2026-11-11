"""CLI: drain ontology_projection_state queue into Neo4j."""

from __future__ import annotations

import argparse
import sys

from app.core.db import SessionLocal
from ontology.projection.engine import drain_pending_projections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drain ontology projection queue.")
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args(argv)
    db = SessionLocal()
    try:
        result = drain_pending_projections(db, limit=args.limit)
        print(result)
        return 0 if result.get("errors", 0) == 0 else 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
