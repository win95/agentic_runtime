"""Load the backfill fixture (seeds/backfill_fixture.json) into FalkorDB.

Usage:
    # Run FalkorDB externally (no compose service is shipped for it):
    docker run -d -p 6379:6379 falkordb/falkordb:latest
    python -m scripts.load_backfill_fixture
    # then run the backfill against that graph:
    python -m scripts.backfill_falkor_to_canonical \
        --falkor-graph backfill_fixture

Env: FALKOR_HOST (default 127.0.0.1), FALKOR_PORT (default 6379).
The loader is idempotent (MERGE on ``id``), so re-running it changes
nothing. Labels and relationship types are allowlisted against the
canonical ontology types — a fixture node with an unknown type is an
error, not a silent skip.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ontology.canonical.schema import ENTITY_TYPES_V1, RELATIONSHIP_TYPES_V1

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "seeds" / "backfill_fixture.json"


def load_fixture(graph, fixture: dict) -> dict:
    """Write every fixture node/edge into ``graph`` (a FalkorDB graph).

    ``graph`` only needs a ``query(cypher, params)`` method, so tests
    can pass a recording fake.
    """
    nodes = fixture.get("nodes", [])
    edges = fixture.get("edges", [])
    for node in nodes:
        labels = node.get("labels") or []
        props = node.get("properties") or {}
        if len(labels) != 1 or labels[0] not in ENTITY_TYPES_V1:
            raise ValueError(f"fixture node has unknown labels: {labels}")
        if not props.get("id"):
            raise ValueError(f"fixture node has no id: {node}")
        label = labels[0]
        graph.query(
            f"MERGE (n:{label} {{id: $id}}) SET n += $props",
            {"id": props["id"], "props": props},
        )
    for edge in edges:
        rel_type = edge.get("type")
        if rel_type not in RELATIONSHIP_TYPES_V1:
            raise ValueError(f"fixture edge has unknown type: {rel_type}")
        src = edge.get("from") or {}
        dst = edge.get("to") or {}
        src_label = (src.get("labels") or [None])[0]
        dst_label = (dst.get("labels") or [None])[0]
        src_id = (src.get("properties") or {}).get("id")
        dst_id = (dst.get("properties") or {}).get("id")
        if src_label not in ENTITY_TYPES_V1 or dst_label not in ENTITY_TYPES_V1:
            raise ValueError(f"fixture edge has unknown endpoint labels: {edge}")
        if not src_id or not dst_id:
            raise ValueError(f"fixture edge has no endpoint id: {edge}")
        graph.query(
            f"MATCH (a:{src_label} {{id: $src_id}}), (b:{dst_label} {{id: $dst_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b)",
            {"src_id": src_id, "dst_id": dst_id},
        )
    return {"nodes": len(nodes), "edges": len(edges)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the backfill fixture into FalkorDB.")
    parser.add_argument("--file", default=str(DEFAULT_FILE), help="fixture JSON path")
    parser.add_argument("--falkor-host", default=os.getenv("FALKOR_HOST", "127.0.0.1"))
    parser.add_argument("--falkor-port", type=int, default=int(os.getenv("FALKOR_PORT", "6379")))
    parser.add_argument(
        "--falkor-graph",
        default=os.getenv("FALKOR_GRAPH", "backfill_fixture"),
        help="FalkorDB graph name (default: backfill_fixture)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the fixture without touching FalkorDB",
    )
    args = parser.parse_args(argv)

    fixture = json.loads(Path(args.file).read_text(encoding="utf-8"))
    if args.dry_run:
        # Validate through the same code path, with a no-op graph.
        class _NoopGraph:
            def query(self, cypher, params=None):
                return None

        counts = load_fixture(_NoopGraph(), fixture)
        print(f"Dry run OK: fixture has {counts['nodes']} nodes + {counts['edges']} edges.")
        return 0

    try:
        from falkordb import FalkorDB
    except ImportError:
        print("falkordb package is not installed; pip install -r requirements.txt first.")
        return 1
    client = FalkorDB(host=args.falkor_host, port=args.falkor_port)
    graph = client.select_graph(args.falkor_graph)
    try:
        counts = load_fixture(graph, fixture)
    except Exception as exc:  # noqa: BLE001 — report the failure plainly
        print(f"Failed to load fixture into FalkorDB graph {args.falkor_graph!r}: {exc}")
        return 1
    print(
        f"Loaded fixture into FalkorDB graph {args.falkor_graph!r}: "
        f"{counts['nodes']} nodes + {counts['edges']} edges."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
