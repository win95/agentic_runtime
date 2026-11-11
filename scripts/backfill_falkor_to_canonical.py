"""One-off backfill: export FalkorDB -> canonical ontology (L1).

Reads every node and relationship from a FalkorDB graph and writes
them through ``OntologyWriteService`` into the canonical PostgreSQL
tables, so the normal L2 projection then carries them into Neo4j.

Run this BEFORE deleting the FalkorDB service. If there is no
production graph data to migrate, skip this script entirely and
re-seed / re-ingest instead (``make seed``, ``make ingest``).

Idempotent: every write carries ``trace_id="falkor-backfill"`` and a
``source_event_id`` of ``falkor:{type}:{key}`` (nodes) or
``falkor:rel:{from_key}:{rel_type}:{to_key}`` (relationships), so a
re-run replays instead of duplicating (see ``OntologyWriteService``).

Requires the FalkorDB driver, which is a backfill-only dependency
(``FalkorDB==1.2.0`` in requirements.txt, slated for removal after
migration). It is imported lazily here and nowhere else in the app:
if it is missing, this script exits non-zero telling you to
``pip install falkordb==1.2.0``.

Node identity: a FalkorDB node's label must be in
``ENTITY_TYPES_V1`` and its key is its ``id`` property (the same
convention the Neo4j projection uses). Nodes with another label, or
with no ``id`` property, are skipped with a warning. Relationship
types must be in ``RELATIONSHIP_TYPES_V1``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from app.core.db import SessionLocal
from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.schema import ENTITY_TYPES_V1, RELATIONSHIP_TYPES_V1
from ontology.canonical.write_service import OntologyWriteService

TRACE_ID = "falkor-backfill"
DEFAULT_GRAPH = "knowledge_graph"


class BackfillError(RuntimeError):
    """Fatal backfill error: nothing was written."""


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _jsonable_properties(properties: dict[str, Any] | None) -> dict[str, Any]:
    """Drop property values that cannot be stored as JSON.

    FalkorDB property values are normally str/int/float/bool/list, but
    a stray value must not abort the whole backfill at flush time.
    """
    clean: dict[str, Any] = {}
    for key, value in (properties or {}).items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            _warn(f"dropping non-JSON property {key!r} (value {value!r})")
            continue
        clean[key] = value
    return clean


def _node_identity(node: Any) -> tuple[str, str] | None:
    """(entity_type, entity_key) for a FalkorDB node, or None to skip."""
    labels = list(getattr(node, "labels", None) or [])
    if not labels:
        return None
    label = labels[0]
    if len(labels) > 1:
        _warn(f"node has extra labels {labels[1:]!r}; keeping only {label!r}")
    properties = getattr(node, "properties", None) or {}
    key = properties.get("id")
    if not isinstance(key, str) or not key:
        return None
    return label, key


def map_node(node: Any) -> dict[str, Any] | None:
    """Map a FalkorDB node to a canonical entity write, or None to skip."""
    identity = _node_identity(node)
    if identity is None:
        _warn(f"skipping node with no label / no 'id' property: {node!r}")
        return None
    entity_type, entity_key = identity
    if entity_type not in ENTITY_TYPES_V1:
        _warn(f"skipping node {entity_type}:{entity_key}: type not in ENTITY_TYPES_V1")
        return None
    properties = _jsonable_properties(getattr(node, "properties", None) or {})
    properties.pop("id", None)  # the key lives in entity_key, not properties
    return {
        "entity_type": entity_type,
        "entity_key": entity_key,
        "properties": properties,
    }


def map_edge(edge: Any) -> dict[str, Any] | None:
    """Map a FalkorDB edge to a canonical relationship write, or None to skip."""
    rel_type = str(getattr(edge, "relation", "") or "").strip().upper()
    if rel_type not in RELATIONSHIP_TYPES_V1:
        _warn(f"skipping relationship of type {rel_type!r}: not in RELATIONSHIP_TYPES_V1")
        return None
    source = _node_identity(getattr(edge, "src_node", None))
    target = _node_identity(getattr(edge, "dest_node", None))
    if source is None or target is None:
        _warn(f"skipping relationship {rel_type}: endpoint has no label / 'id'")
        return None
    (source_type, source_key), (target_type, target_key) = source, target
    if source_type not in ENTITY_TYPES_V1 or target_type not in ENTITY_TYPES_V1:
        _warn(
            "skipping relationship "
            f"{source_type}:{source_key} -[{rel_type}]-> {target_type}:{target_key}: "
            "endpoint type not in ENTITY_TYPES_V1"
        )
        return None
    properties = _jsonable_properties(getattr(edge, "properties", None) or {})
    return {
        "relationship_type": rel_type,
        "source_type": source_type,
        "source_key": source_key,
        "target_type": target_type,
        "target_key": target_key,
        "properties": properties,
    }


def connect_falkor(host: str, port: int, graph_name: str):
    """Connect to FalkorDB and select the graph.

    Raises BackfillError (nothing written) when the driver is missing
    or the server is unreachable. The ``FalkorDB`` constructor connects
    eagerly, so an unreachable host fails here, before any read.
    """
    try:
        from falkordb import FalkorDB
    except ImportError as exc:
        raise BackfillError(
            "falkordb driver is not installed; install it for this one-off run with "
            "`pip install falkordb==1.2.0`"
        ) from exc
    try:
        client = FalkorDB(host=host, port=port)
        return client.select_graph(graph_name)
    except Exception as exc:  # noqa: BLE001 — connection errors are data here
        raise BackfillError(
            f"FalkorDB unreachable at {host}:{port} (graph {graph_name!r}): {exc}"
        ) from exc


def _read_all(graph: Any, query: str) -> list[list[Any]]:
    try:
        result = graph.query(query)
    except Exception as exc:  # noqa: BLE001
        raise BackfillError(f"FalkorDB query failed ({query!r}): {exc}") from exc
    return list(getattr(result, "result_set", []) or [])


def run_backfill(
    *,
    graph: Any,
    session: Any,
    dry_run: bool = False,
) -> dict[str, int]:
    """Export one FalkorDB graph into the canonical tables.

    Entities are committed first, then relationships, on the one
    session the caller owns. On error the caller rolls back; this
    function raises and writes nothing further. Returns counters.
    """
    writer = OntologyWriteService(CanonicalOntologyRepository(session))
    counts = {
        "nodes_seen": 0,
        "entities_written": 0,
        "entities_skipped": 0,
        "edges_seen": 0,
        "relationships_written": 0,
        "relationships_skipped": 0,
    }

    node_rows = _read_all(graph, "MATCH (n) RETURN n")
    mapped_nodes: list[dict[str, Any]] = []
    for row in node_rows:
        counts["nodes_seen"] += 1
        mapped = map_node(row[0])
        if mapped is None:
            counts["entities_skipped"] += 1
            continue
        mapped_nodes.append(mapped)
    if not dry_run:
        for entity in mapped_nodes:
            result = writer.write_entity(
                entity_type=entity["entity_type"],
                entity_key=entity["entity_key"],
                properties=entity["properties"],
                provenance=WriteProvenance(
                    trace_id=TRACE_ID,
                    source_event_id=f"falkor:{entity['entity_type']}:{entity['entity_key']}",
                ),
            )
            if result.applied:
                counts["entities_written"] += 1
        # Entities first: a relationship's endpoints must exist before
        # the relationship row is written.
        session.commit()
    else:
        counts["entities_written"] = len(mapped_nodes)

    edge_rows = _read_all(graph, "MATCH (a)-[r]->(b) RETURN a, r, b")
    mapped_edges: list[dict[str, Any]] = []
    for row in edge_rows:
        counts["edges_seen"] += 1
        mapped = map_edge(row[1])
        if mapped is None:
            counts["relationships_skipped"] += 1
            continue
        mapped_edges.append(mapped)
    if not dry_run:
        prov = WriteProvenance(trace_id=TRACE_ID)
        for rel in mapped_edges:
            for entity_type, entity_key in (
                (rel["source_type"], rel["source_key"]),
                (rel["target_type"], rel["target_key"]),
            ):
                writer.ensure_entity(
                    entity_type=entity_type,
                    entity_key=entity_key,
                    properties=None,
                    provenance=prov,
                )
            result = writer.write_relationship(
                relationship_type=rel["relationship_type"],
                source_type=rel["source_type"],
                source_key=rel["source_key"],
                target_type=rel["target_type"],
                target_key=rel["target_key"],
                properties=rel["properties"],
                provenance=WriteProvenance(
                    trace_id=TRACE_ID,
                    source_event_id=(
                        f"falkor:rel:{rel['source_key']}:{rel['relationship_type']}:"
                        f"{rel['target_key']}"
                    ),
                ),
            )
            if result.applied:
                counts["relationships_written"] += 1
        session.commit()
    else:
        counts["relationships_written"] = len(mapped_edges)

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-off backfill: export a FalkorDB graph into the "
        "canonical ontology tables (PostgreSQL). Run BEFORE deleting the "
        "FalkorDB service; if there is no production graph data, skip "
        "this script and re-seed / re-ingest instead."
    )
    parser.add_argument(
        "--falkor-host",
        default=os.environ.get("FALKOR_HOST", "localhost"),
        help="FalkorDB host (default: $FALKOR_HOST or localhost)",
    )
    parser.add_argument(
        "--falkor-port",
        type=int,
        default=int(os.environ.get("FALKOR_PORT", "6379")),
        help="FalkorDB port (default: $FALKOR_PORT or 6379)",
    )
    parser.add_argument(
        "--falkor-graph",
        default=DEFAULT_GRAPH,
        help=f"FalkorDB graph name (default: {DEFAULT_GRAPH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count nodes/relationships that would be imported, write nothing",
    )
    args = parser.parse_args(argv)

    try:
        graph = connect_falkor(args.falkor_host, args.falkor_port, args.falkor_graph)
    except BackfillError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    db = SessionLocal()
    try:
        try:
            counts = run_backfill(graph=graph, session=db, dry_run=args.dry_run)
        except BackfillError as exc:
            db.rollback()
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            print(f"ERROR: backfill failed, rolled back: {exc}", file=sys.stderr)
            return 1
    finally:
        db.close()

    print(
        "Backfill complete"
        + (" (dry run — nothing written)" if args.dry_run else "")
        + f": {counts['entities_written']} entities "
        f"({counts['entities_skipped']} skipped) from {counts['nodes_seen']} nodes, "
        f"{counts['relationships_written']} relationships "
        f"({counts['relationships_skipped']} skipped) from {counts['edges_seen']} edges."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
