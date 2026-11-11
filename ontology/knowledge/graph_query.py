"""Knowledge graph queries against Neo4j."""

from __future__ import annotations

import re
from typing import Any

from ontology.graph.protocol import GraphStore
from ontology.projection.provider import get_graph_store

_CYPHER_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_DOC_EDGE_TYPES = ("RELATED_TO", "DOCUMENTED_BY")


def _label(entity_type: str) -> str:
    if not _CYPHER_IDENTIFIER_RE.match(entity_type or ""):
        raise ValueError(f"Invalid entity_type for graph: {entity_type!r}")
    return entity_type


def neighbors(
    entity_type: str,
    entity_key: str,
    *,
    depth: int = 1,
    limit: int = 50,
    store: GraphStore | None = None,
) -> list[dict[str, Any]]:
    store = store or get_graph_store()
    label = _label(entity_type)
    depth = max(1, min(depth, 3))
    limit = max(1, min(limit, 200))
    if depth == 1:
        cypher = (
            f"MATCH (a:{label} {{id: $key}})-[r]-(b) "
            "RETURN a.id AS from_id, type(r) AS rel_type, b.id AS to_id, "
            "labels(b)[0] AS to_label LIMIT $limit"
        )
    else:
        cypher = (
            f"MATCH (a:{label} {{id: $key}})-[r*1..{depth}]-(b) "
            "WITH a, b, r[0] AS rel "
            "RETURN DISTINCT a.id AS from_id, type(rel) AS rel_type, b.id AS to_id, "
            "labels(b)[0] AS to_label LIMIT $limit"
        )
    rows = store.run_query(cypher, {"key": entity_key, "limit": limit})
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "from": row.get("from_id"),
                "type": row.get("rel_type"),
                "to": row.get("to_id"),
                "to_label": row.get("to_label"),
            }
        )
    return out


def linked_documents(
    entity_type: str,
    entity_key: str,
    *,
    store: GraphStore | None = None,
) -> list[dict[str, Any]]:
    store = store or get_graph_store()
    label = _label(entity_type)
    rel_filter = "|".join(_DOC_EDGE_TYPES)
    cypher = (
        f"MATCH (a:{label} {{id: $key}})-[r:{rel_filter}]-(d:Document) "
        "RETURN d.id AS document_id, type(r) AS rel_type"
    )
    rows = store.run_query(cypher, {"key": entity_key})
    return [{"document_id": r.get("document_id"), "rel_type": r.get("rel_type")} for r in rows]


def subgraph(
    root_type: str,
    root_key: str,
    *,
    depth: int = 2,
    limit: int = 100,
    store: GraphStore | None = None,
) -> dict[str, Any]:
    rows = neighbors(root_type, root_key, depth=depth, limit=limit, store=store)
    nodes: list[dict[str, Any]] = [{"id": root_key, "labels": [root_type]}]
    seen = {root_key}
    relationships: list[dict[str, Any]] = []
    for row in rows:
        to_id = row.get("to")
        if to_id and to_id not in seen:
            seen.add(to_id)
            nodes.append(
                {"id": to_id, "labels": [row.get("to_label") or "Entity"]}
            )
        relationships.append(
            {
                "type": row.get("type"),
                "start": row.get("from") or root_key,
                "end": to_id,
            }
        )
    return {
        "root_type": root_type,
        "root_key": root_key,
        "nodes": nodes[:limit],
        "relationships": relationships[:limit],
    }
