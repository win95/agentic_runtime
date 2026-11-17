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


# ----------------------------------------------------------------------
# B5 — query API helpers (design §7, Query and Retrieval)
# ----------------------------------------------------------------------

#: Labels that hold ontology metadata, never instances — excluded
#: from entity search so a query never returns a Class node.
_METADATA_LABELS = ("OntologyClass", "ObjectProperty", "DataProperty")

#: First meaningful keyword of a read-only Cypher statement (design
#: §7: "chỉ đọc"). ``CALL`` is allowed as a first word only for
#: non-APOC procedure calls — ``CALL apoc.*`` is banned by the
#: whole-query scan below.
_READ_ONLY_FIRST_KEYWORDS = frozenset(
    {"MATCH", "WITH", "RETURN", "CALL", "UNWIND", "OPTIONAL"}
)

#: Whole-query banned words (matched as standalone words, \b...\b,
#: case-insensitive, after stripping -- and /* */ comments). ``SET``
#: also blocks ``SET`` inside a later clause of a MATCH... query.
_BANNED_KEYWORDS = (
    "MERGE",
    "CREATE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD",
)

_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_FIRST_WORD_RE = re.compile(r"^\s*([A-Za-z]+)", re.IGNORECASE)


def _strip_cypher_comments(cypher: str) -> str:
    """Remove ``--`` line comments and ``/* */`` block comments.

    The read-only scan runs on the stripped text so a banned word
    inside a comment cannot trip (or hide) the guard. String
    literals are NOT stripped: a banned word inside a string literal
    (e.g. a property value ``"deleted"``) is still rejected — a
    deliberate fail-closed choice for a read-only endpoint.
    """
    return _BLOCK_COMMENT_RE.sub(" ", _LINE_COMMENT_RE.sub(" ", cypher))


def _assert_read_only_cypher(cypher: str) -> str:
    """Guard for ``POST /ontology/query`` (design §7).

    Raises ValueError unless (a) the first meaningful keyword is one
    of MATCH / WITH / RETURN / CALL / UNWIND / OPTIONAL, and (b) no
    banned keyword (MERGE, CREATE, DELETE, DETACH, SET, REMOVE, DROP,
    LOAD) appears as a standalone word anywhere in the query, and no
    ``CALL apoc.*`` appears. Returns the original query unchanged.
    """
    if not isinstance(cypher, str) or not cypher.strip():
        raise ValueError("cypher must be a non-empty string")
    stripped = _strip_cypher_comments(cypher)
    match = _FIRST_WORD_RE.match(stripped)
    first = match.group(1).upper() if match else ""
    if first not in _READ_ONLY_FIRST_KEYWORDS:
        raise ValueError(
            f"read-only query must start with one of "
            f"{sorted(_READ_ONLY_FIRST_KEYWORDS)}, got {first!r}"
        )
    for keyword in _BANNED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", stripped, re.IGNORECASE):
            raise ValueError(f"read-only query must not contain {keyword}")
    if re.search(r"\bCALL\s+apoc\.", stripped, re.IGNORECASE):
        raise ValueError("read-only query must not call apoc.* procedures")
    return cypher


def _entity_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Shape one ``(id, label, properties)`` row into an entity dict."""
    entity_key = row.get("id")
    if not isinstance(entity_key, str) or not entity_key:
        return None
    label = row.get("label")
    entity_type = label if isinstance(label, str) and label else UNKNOWN_NODE_TYPE
    props = row.get("properties")
    properties = (
        {k: v for k, v in dict(props).items() if k != "id"}
        if isinstance(props, dict)
        else {}
    )
    return {"entity_type": entity_type, "entity_key": entity_key, "properties": properties}


def entities_by_class(
    store: GraphStore,
    class_id: str,
    *,
    include_subclasses: bool = True,
    limit: int = 100,
) -> list[dict]:
    """Instances of ``class_id`` — plus its subclasses (R1) by default.

    The subclass set comes from
    :func:`ontology.reasoning.subclass_descendants` (BFS down
    ``SUBCLASS_OF``, cycle-safe). One ``MATCH (n:Label)`` per class in
    the set; rows are merged, sorted by entity key and cut to the
    total ``limit`` so the result is deterministic.
    """
    from ontology.reasoning import subclass_descendants

    label = _label(class_id)
    limit = max(1, min(limit, 500))
    classes = [label]
    if include_subclasses:
        classes.extend(
            _label(child) for child in subclass_descendants(store, label)
        )
    found: list[dict] = []
    for class_label in classes:
        rows = store.run_query(
            f"MATCH (n:{class_label}) "
            "RETURN n.id AS id, labels(n)[0] AS label, "
            "properties(n) AS properties ORDER BY n.id LIMIT $limit",
            {"limit": limit},
        )
        for row in rows:
            shaped = _entity_row(row)
            if shaped is not None:
                found.append(shaped)
    found.sort(key=lambda item: item["entity_key"])
    return found[:limit]


def search_entities(
    store: GraphStore,
    query: str,
    *,
    limit: int = 50,
) -> list[dict]:
    """Find entities whose ``id`` or ``name`` contains ``query``.

    Case-insensitive ``CONTAINS`` (``toLower`` on both sides), over
    instance nodes only — ontology metadata nodes (OntologyClass /
    ObjectProperty / DataProperty) are excluded. No Neo4j full-text
    index is used (design §12 open question): this is a label scan
    with a ``CONTAINS`` filter, fine for the demo-scale graph and to
    be revisited (full-text index or gazetteer prefix) if the graph
    grows. An empty/blank query returns [] instead of everything.
    """
    text = (query or "").strip()
    if not text:
        return []
    limit = max(1, min(limit, 500))
    exclusions = " AND ".join(f"NOT n:{label}" for label in _METADATA_LABELS)
    rows = store.run_query(
        "MATCH (n) WHERE " + exclusions + " AND "
        "(toLower(n.id) CONTAINS toLower($query) "
        "OR toLower(coalesce(n.name, '')) CONTAINS toLower($query)) "
        "RETURN n.id AS id, labels(n)[0] AS label, "
        "properties(n) AS properties ORDER BY n.id LIMIT $limit",
        {"query": text, "limit": limit},
    )
    found: list[dict] = []
    for row in rows:
        shaped = _entity_row(row)
        if shaped is not None:
            found.append(shaped)
    return found[:limit]


def traverse(
    store: GraphStore,
    root_type: str,
    root_key: str,
    *,
    depth: int = 2,
    limit: int = 100,
    rel_types: list[str] | None = None,
) -> dict[str, Any]:
    """Multi-hop traversal from a root, optionally filtered by relation type.

    Same shape as :func:`subgraph` (``nodes`` + ``relationships``).
    Without ``rel_types`` this is exactly :func:`subgraph`. With
    ``rel_types`` it runs its own Cypher —
    ``(a)-[r:T1|T2*1..depth]-(b)`` — so the filter applies to EVERY
    hop of the path, not just the first: a node reachable only
    through a filtered-out relationship is dropped with it (the
    root node always stays). Filtering the ``neighbors`` output
    instead would be wrong here: ``neighbors`` reports the FIRST
    hop's type (``r[0]``) for a multi-hop row, so a second hop of a
    filtered-out type would still look like an allowed one.
    """
    if rel_types is None:
        return subgraph(root_type, root_key, depth=depth, limit=limit, store=store)
    allowed = [_label(rel_type) for rel_type in rel_types]
    label = _label(root_type)
    depth = max(1, min(depth, 3))
    limit = max(1, min(limit, 500))
    rel_filter = "|".join(allowed)
    cypher = (
        f"MATCH (a:{label} {{id: $key}})-[r:{rel_filter}*1..{depth}]-(b) "
        "WITH a, b, r[0] AS rel "
        "RETURN DISTINCT a.id AS from_id, type(rel) AS rel_type, b.id AS to_id, "
        "labels(b)[0] AS to_label LIMIT $limit"
    )
    rows = store.run_query(cypher, {"key": root_key, "limit": limit})
    nodes: list[dict[str, Any]] = [{"id": root_key, "labels": [root_type]}]
    seen = {root_key}
    relationships: list[dict[str, Any]] = []
    for row in rows:
        to_id = row.get("to_id")
        if to_id and to_id not in seen:
            seen.add(to_id)
            nodes.append(
                {"id": to_id, "labels": [row.get("to_label") or UNKNOWN_NODE_TYPE]}
            )
        relationships.append(
            {
                "type": row.get("rel_type"),
                "start": row.get("from_id") or root_key,
                "end": to_id,
            }
        )
    return {
        "root_type": root_type,
        "root_key": root_key,
        "nodes": nodes[:limit],
        "relationships": relationships[:limit],
    }


def list_derived_edges(
    store: GraphStore,
    *,
    entity_type: str | None = None,
    entity_key: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Derived edges (``derived=true``, design §4.3).

    Whole graph when no entity is given; when ``entity_key`` is
    given, only edges touching that entity (either endpoint). When
    ``entity_type`` is also given, the entity endpoint must carry
    that label. Each row carries ``rule`` and ``base_ref`` so the
    edge traces back to its base edge.
    """
    limit = max(1, min(limit, 500))
    params: dict[str, Any] = {"limit": limit}
    if entity_key is not None:
        if not isinstance(entity_key, str) or not entity_key:
            raise ValueError("Invalid (empty) entity_key")
        where = "WHERE (a.id = $entity_key OR b.id = $entity_key)"
        params["entity_key"] = entity_key
        if entity_type is not None:
            label = _label(entity_type)
            where += (
                " AND (a.id = $entity_key AND $label = labels(a)[0]"
                " OR b.id = $entity_key AND $label = labels(b)[0])"
            )
            params["label"] = label
        cypher = (
            "MATCH (a)-[r {derived: true}]->(b) " + where + " "
            "RETURN a.id AS source_key, labels(a)[0] AS source_type, "
            "type(r) AS rel_type, b.id AS target_key, "
            "labels(b)[0] AS target_type, r.rule AS rule, "
            "r.base_ref AS base_ref ORDER BY source_key, rel_type, target_key "
            "LIMIT $limit"
        )
    else:
        if entity_type is not None:
            raise ValueError(
                "entity_type without entity_key is not supported for derived edges"
            )
        cypher = (
            "MATCH (a)-[r {derived: true}]->(b) "
            "RETURN a.id AS source_key, labels(a)[0] AS source_type, "
            "type(r) AS rel_type, b.id AS target_key, "
            "labels(b)[0] AS target_type, r.rule AS rule, "
            "r.base_ref AS base_ref ORDER BY source_key, rel_type, target_key "
            "LIMIT $limit"
        )
    rows = store.run_query(cypher, params)
    return [
        {
            "source_type": row.get("source_type"),
            "source_key": row.get("source_key"),
            "rel_type": row.get("rel_type"),
            "target_type": row.get("target_type"),
            "target_key": row.get("target_key"),
            "rule": row.get("rule"),
            "base_ref": row.get("base_ref"),
        }
        for row in rows
    ]
