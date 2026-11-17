"""One-off backfill: copy L1 ontology object data into Neo4j.

Reads every row from ``ontology_entities`` / ``ontology_relationships``
on L1 and MERGEs it onto the matching Neo4j node/edge (``SET``
properties + ``status`` + ``version``), so migration 0015 can then drop
the ``properties`` columns and the ``ontology_versions`` table.

Run this while BOTH systems still hold the old data, and verify
(``--verify``) BEFORE running migration 0015. There is no automatic
rollback after 0015: back up Postgres (``pg_dump``) before the
migration and Neo4j before this backfill.

L1 is read with raw SQL (``text()``), NOT the ``ontology.models`` ORM
models: the ``properties`` column is deleted from the models in the
same push as this script, and this script must keep working after
that change. Inactive (archived/deleted) rows are backfilled too,
carrying their status — the drain/reconciler decides what to do with
them afterwards; the backfill itself never deletes Neo4j data.

Modes (``python -m scripts.backfill_l1_to_neo4j``):
- ``--dry-run``: count rows that would be written, write nothing.
- default: write to Neo4j.
- ``--verify``: compare Neo4j against L1 without writing — count
  active-status nodes/edges vs active L1 rows, and spot-check the
  properties of a few sample entities.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from sqlalchemy import text

from app.core.db import SessionLocal
from ontology.graph.neo4j_store import Neo4jGraphStore

DEFAULT_SPOT_CHECK_SAMPLE = 3


class BackfillError(RuntimeError):
    """Fatal backfill error: nothing was written."""


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _parse_properties(value: Any) -> dict[str, Any]:
    """Normalize a L1 ``properties`` cell to a dict.

    PostgreSQL JSON comes back as a dict; SQLite stores JSON as TEXT,
    so parse it. A cell that parses to nothing usable becomes ``{}``
    with a warning instead of aborting the whole backfill.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            _warn(f"dropping unparseable properties value: {value!r}")
            return {}
        if isinstance(parsed, dict):
            return parsed
        _warn(f"dropping non-object properties value: {value!r}")
        return {}
    _warn(f"dropping unexpected properties value of type {type(value).__name__}")
    return {}


def read_entities(session: Any) -> list[dict[str, Any]]:
    """All L1 entity rows, via raw SQL (no ORM model)."""
    rows = session.execute(
        text(
            "SELECT entity_type, entity_key, properties, status, version "
            "FROM ontology_entities"
        )
    ).all()
    return [
        {
            "entity_type": row[0],
            "entity_key": row[1],
            "properties": _parse_properties(row[2]),
            "status": row[3],
            "version": row[4],
        }
        for row in rows
    ]


def read_relationships(session: Any) -> list[dict[str, Any]]:
    """All L1 relationship rows, via raw SQL (no ORM model)."""
    rows = session.execute(
        text(
            "SELECT relationship_type, source_type, source_key, "
            "target_type, target_key, properties, status, version "
            "FROM ontology_relationships"
        )
    ).all()
    return [
        {
            "relationship_type": row[0],
            "source_type": row[1],
            "source_key": row[2],
            "target_type": row[3],
            "target_key": row[4],
            "properties": _parse_properties(row[5]),
            "status": row[6],
            "version": row[7],
        }
        for row in rows
    ]


def run_backfill(
    *,
    session: Any,
    store: Any | None,
    dry_run: bool = False,
) -> dict[str, int]:
    """MERGE every L1 row into Neo4j (or just count it, on dry run).

    Entities are written first: ``merge_relationship`` MATCHes its
    endpoints, so an edge whose endpoints were never merged would
    silently not be created. Returns counters.
    """
    if not dry_run and store is None:
        raise BackfillError("a Neo4j store is required for a real backfill run")
    entities = read_entities(session)
    relationships = read_relationships(session)
    counts = {
        "entities_seen": len(entities),
        "entities_written": 0,
        "relationships_seen": len(relationships),
        "relationships_written": 0,
    }
    if dry_run:
        counts["entities_written"] = len(entities)
        counts["relationships_written"] = len(relationships)
        return counts
    assert store is not None  # narrowed by the guard above
    for entity in entities:
        props = dict(entity["properties"])
        props["status"] = entity["status"]
        props["version"] = entity["version"]
        store.merge_node(entity["entity_type"], entity["entity_key"], props)
        counts["entities_written"] += 1
    for rel in relationships:
        props = dict(rel["properties"])
        props["status"] = rel["status"]
        props["version"] = rel["version"]
        store.merge_relationship(
            rel["source_type"],
            rel["source_key"],
            rel["relationship_type"],
            rel["target_type"],
            rel["target_key"],
            props,
        )
        counts["relationships_written"] += 1
    return counts


def _count_active_nodes(store: Any) -> int:
    rows = store.run_query(
        "MATCH (n) WHERE n.status = 'active' RETURN count(n) AS total"
    )
    return int(rows[0]["total"]) if rows else 0


def _count_active_edges(store: Any) -> int:
    rows = store.run_query(
        "MATCH ()-[r]->() WHERE r.status = 'active' RETURN count(r) AS total"
    )
    return int(rows[0]["total"]) if rows else 0


def _fetch_node_props(store: Any, label: str, node_id: str) -> dict[str, Any] | None:
    rows = store.run_query(
        f"MATCH (n:{label} {{id: $id}}) RETURN properties(n) AS props",
        {"id": node_id},
    )
    if not rows:
        return None
    return dict(rows[0].get("props") or {})


def verify_backfill(
    *,
    session: Any,
    store: Any,
    sample: int = DEFAULT_SPOT_CHECK_SAMPLE,
) -> dict[str, Any]:
    """Compare Neo4j against L1 without writing anything.

    Counts active-status nodes/edges on Neo4j vs active L1 rows (an
    archived L1 row is backfilled as a node carrying
    ``status='archived'``, so it must NOT be counted here), then
    spot-checks the properties/status/version of up to ``sample``
    active entities. Returns counts plus a ``match`` flag and per-entity
    spot-check results.
    """
    if store is None:
        raise BackfillError("a Neo4j store is required for --verify")
    entities = read_entities(session)
    relationships = read_relationships(session)
    active_entities = [e for e in entities if e["status"] == "active"]
    active_rels = [r for r in relationships if r["status"] == "active"]
    neo4j_nodes = _count_active_nodes(store)
    neo4j_edges = _count_active_edges(store)
    spot_checks: list[dict[str, Any]] = []
    for entity in active_entities[: max(sample, 0)]:
        expected = dict(entity["properties"])
        expected["status"] = entity["status"]
        expected["version"] = entity["version"]
        actual = _fetch_node_props(store, entity["entity_type"], entity["entity_key"])
        if actual is None:
            spot_checks.append(
                {
                    "entity_type": entity["entity_type"],
                    "entity_key": entity["entity_key"],
                    "ok": False,
                    "detail": "node missing on Neo4j",
                }
            )
            continue
        mismatches = {
            key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected.items()
            if actual.get(key) != value
        }
        spot_checks.append(
            {
                "entity_type": entity["entity_type"],
                "entity_key": entity["entity_key"],
                "ok": not mismatches,
                "detail": "ok" if not mismatches else f"mismatched properties: {mismatches}",
            }
        )
    result = {
        "l1_active_entities": len(active_entities),
        "neo4j_active_nodes": neo4j_nodes,
        "l1_active_relationships": len(active_rels),
        "neo4j_active_edges": neo4j_edges,
        "l1_total_entities": len(entities),
        "l1_total_relationships": len(relationships),
        "spot_checks": spot_checks,
    }
    result["match"] = (
        result["l1_active_entities"] == result["neo4j_active_nodes"]
        and result["l1_active_relationships"] == result["neo4j_active_edges"]
        and all(check["ok"] for check in spot_checks)
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-off backfill: copy L1 ontology object data "
        "(ontology_entities / ontology_relationships) into Neo4j. Run "
        "BEFORE migration 0015 drops the properties columns; verify "
        "with --verify before migrating."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count rows that would be written, write nothing to Neo4j",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare Neo4j against L1 without writing (counts + spot-check)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SPOT_CHECK_SAMPLE,
        help="entities to spot-check in --verify mode "
        f"(default: {DEFAULT_SPOT_CHECK_SAMPLE})",
    )
    args = parser.parse_args(argv)

    if args.verify and args.dry_run:
        print("ERROR: --verify and --dry-run cannot be combined", file=sys.stderr)
        return 2

    store: Neo4jGraphStore | None = None
    if not args.dry_run:
        try:
            store = Neo4jGraphStore()
        except Exception as exc:  # noqa: BLE001 — Neo4j down is a fatal error here
            print(f"ERROR: Neo4j unavailable: {exc}", file=sys.stderr)
            return 1

    db = SessionLocal()
    try:
        try:
            if args.verify:
                assert store is not None
                result = verify_backfill(session=db, store=store, sample=args.sample)
                print(
                    "Verify: L1 active entities "
                    f"{result['l1_active_entities']} vs Neo4j active nodes "
                    f"{result['neo4j_active_nodes']}; L1 active relationships "
                    f"{result['l1_active_relationships']} vs Neo4j active edges "
                    f"{result['neo4j_active_edges']} "
                    f"(L1 totals: {result['l1_total_entities']} entities, "
                    f"{result['l1_total_relationships']} relationships)."
                )
                for check in result["spot_checks"]:
                    print(
                        f"  spot-check {check['entity_type']}:{check['entity_key']}: "
                        f"{'OK' if check['ok'] else 'MISMATCH — ' + check['detail']}"
                    )
                if not result["match"]:
                    print(
                        "ERROR: Neo4j does not match L1 — do NOT run "
                        "migration 0015 yet.",
                        file=sys.stderr,
                    )
                    return 1
                print("Verify passed: Neo4j matches L1.")
                return 0
            counts = run_backfill(session=db, store=store, dry_run=args.dry_run)
        except BackfillError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: backfill failed: {exc}", file=sys.stderr)
            return 1
    finally:
        db.close()
        if store is not None:
            store.close()

    print(
        "Backfill complete"
        + (" (dry run — nothing written)" if args.dry_run else "")
        + f": {counts['entities_written']} entities "
        f"from {counts['entities_seen']} L1 rows, "
        f"{counts['relationships_written']} relationships "
        f"from {counts['relationships_seen']} L1 rows."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
