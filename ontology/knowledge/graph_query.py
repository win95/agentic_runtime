"""Knowledge graph queries against Neo4j."""

from __future__ import annotations

import re
from typing import Any, Iterable

from ontology.canonical.schema import (
    REL_DOCUMENTED_BY,
    REL_RELATED_TO,
    UNKNOWN_NODE_TYPE,
)
from ontology.graph.protocol import GraphStore
from ontology.projection.provider import get_graph_store

_CYPHER_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Doc edges: the two relations that connect a Document to what it
# documents / relates to. Canonical names from schema.py.
_DOC_EDGE_TYPES = (REL_RELATED_TO, REL_DOCUMENTED_BY)


def _label(entity_type: str) -> str:
    if not _CYPHER_IDENTIFIER_RE.match(entity_type or ""):
        raise ValueError(f"Invalid entity_type for graph: {entity_type!r}")
    return entity_type


def list_entity_keys(
    entity_type: str,
    *,
    limit: int = 200,
    store: GraphStore | None = None,
) -> list[str]:
    """Distinct ``id`` values for one entity label, sorted.

    The Phase 5 dynamic gazetteer reads the live ontology's key
    vocabulary through this (``runtime/retrieval_graph.py::
    refresh_gazetteer_codes``): one call per label (Asset,
    Component, Building, Document). The label goes through the
    same ``_label`` validation as the other read functions, so an
    arbitrary caller string can never inject Cypher; ``limit``
    uses the same 1..200 cap. Rows with a missing / non-string
    ``id`` are skipped; the result is sorted so a refresh is
    deterministic for the same store contents.
    """
    store = store or get_graph_store()
    label = _label(entity_type)
    limit = max(1, min(limit, 200))
    cypher = f"MATCH (n:{label}) RETURN DISTINCT n.id AS id LIMIT $limit"
    rows = store.run_query(cypher, {"limit": limit})
    keys = {row.get("id") for row in rows}
    return sorted(key for key in keys if isinstance(key, str) and key)


def _neighbour_row(row: dict[str, Any]) -> dict[str, Any]:
    """Shape one raw neighbour row into the public neighbour shape."""
    return {
        "from": row.get("from_id"),
        "type": row.get("rel_type"),
        "to": row.get("to_id"),
        "to_label": row.get("to_label"),
    }


def neighbors_many(
    keys: Iterable[tuple[str, str] | str],
    *,
    limit_per_key: int = 200,
    store: GraphStore | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """1-hop neighbours for MANY nodes in ONE Cypher round trip (B9).

    The expansion + live-adjacency builders used to call ``neighbors``
    once per seed/node — up to ~51 sequential Neo4j calls inside a
    single 150 ms budget, so on non-localhost Neo4j the budget
    almost certainly exhausted mid-way and left a partial
    adjacency. This one call serves the whole batch:
    ``MATCH (a)-[r]-(b) WHERE a.id IN [...]``.

    ``keys`` are ``(entity_type, key)`` pairs; a bare key string is
    matched without a label (ids are unique across labels by
    construction). Returns ``{key: [row, ...]}`` with rows shaped
    exactly like ``neighbors`` returns them; rows per key are
    sorted by (to, type) and capped at ``limit_per_key`` — the cap
    is applied deterministically in Python after the fetch, so a
    hub node cannot eat another key's rows, and the same key set
    always yields the same rows. A key with no neighbours (or an
    unknown key) maps to []. Raises on store failure (callers
    degrade to expansion-only / vector-only, as before).
    """
    wanted: list[str] = []
    for item in keys:
        key = item[1] if isinstance(item, tuple) else item
        if key not in wanted:
            wanted.append(key)
    if not wanted:
        return {}
    graph = store or get_graph_store()
    if graph is None:
        raise RuntimeError(
            "Graph store unavailable (is Neo4j up? Seed ontology demo + projection drain)"
        )
    rows = graph.run_query(
        "UNWIND $keys AS key "
        "MATCH (a {id: key})-[r]-(b) "
        "RETURN a.id AS from_id, type(r) AS rel_type, b.id AS to_id, labels(b)[0] AS to_label",
        {"keys": wanted},
    )
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in wanted}
    for row in rows:
        from_id = row.get("from_id")
        if from_id in grouped:
            grouped[from_id].append(_neighbour_row(row))
    for key in grouped:
        group = grouped[key]
        group.sort(key=lambda r: (r.get("to") or "", r.get("type") or ""))
        if len(group) > limit_per_key:
            grouped[key] = group[:limit_per_key]
    return grouped


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
    return [_neighbour_row(row) for row in rows]


def linked_documents(
    entity_type: str,
    entity_key: str,
    *,
    limit: int = 50,
    store: GraphStore | None = None,
) -> list[dict[str, Any]]:
    store = store or get_graph_store()
    label = _label(entity_type)
    limit = max(1, min(limit, 200))
    rel_filter = "|".join(_DOC_EDGE_TYPES)
    cypher = (
        f"MATCH (a:{label} {{id: $key}})-[r:{rel_filter}]-(d:Document) "
        "RETURN d.id AS document_id, type(r) AS rel_type LIMIT $limit"
    )
    rows = store.run_query(cypher, {"key": entity_key, "limit": limit})
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
                {"id": to_id, "labels": [row.get("to_label") or UNKNOWN_NODE_TYPE]}
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
