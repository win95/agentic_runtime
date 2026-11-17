"""Reasoning v1 (design §4.3, R1–R3): materialize derived edges at write time.

All reasoning goes through ``store.run_query(cypher, params)`` only — no
Neo4j driver import, no other ontology module (except ``rules.py`` and the
``GraphStore`` protocol). Every label / relationship type interpolated
into Cypher is validated with ``^[A-Za-z][A-Za-z0-9_]*$``; every dynamic
value travels as a Cypher parameter.

Derived-edge identity (design §4.3/§5): a derived edge is keyed by
``(source, type, target, base_ref)`` — the ``MERGE`` pattern carries
``{derived: true, base_ref: $base_ref}`` so re-running the same write is
a no-op instead of duplicating the edge, and deleting a base edge removes
its derived edges by ``base_ref``.
"""

from __future__ import annotations

import re
from typing import Any

from ontology.graph.protocol import GraphStore
from ontology.reasoning.rules import INVERSE_RELATIONS

_CYPHER_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

_TRANSITIVE_TYPE = "PART_OF"


def _assert_identifier(name: str, field: str) -> str:
    """Validate a label / relationship type before it enters Cypher."""
    if not isinstance(name, str) or not _CYPHER_IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid Cypher identifier for {field}: {name!r}")
    return name


def _assert_key(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid (empty) value for {field}: {value!r}")
    return value


def base_ref_for(
    source_type: str,
    source_key: str,
    relationship_type: str,
    target_type: str,
    target_key: str,
) -> str:
    """Canonical ``base_ref`` identifying one base edge.

    The design (§4.3) says a derived edge carries
    ``base_ref=<subject_ref of the base edge>`` but never defines the
    ``subject_ref`` format. This ``type:key:TYPE:type:key`` form is the
    canonical format v1 uses whenever a base edge does not already carry
    a ``base_ref`` property: recompute derives it from the base edge, and
    the write path should pass the same value into
    ``materialize_for_write`` so a later delete-by-``base_ref`` removes
    exactly the edges this write derived.
    """
    _assert_identifier(source_type, "source_type")
    _assert_identifier(relationship_type, "relationship_type")
    _assert_identifier(target_type, "target_type")
    _assert_key(source_key, "source_key")
    _assert_key(target_key, "target_key")
    return f"{source_type}:{source_key}:{relationship_type}:{target_type}:{target_key}"


def _descriptor(
    *,
    rel_type: str,
    source_type: str,
    source_key: str,
    target_type: str,
    target_key: str,
    rule: str,
) -> dict[str, str]:
    return {
        "type": rel_type,
        "source_type": source_type,
        "source_key": source_key,
        "target_type": target_type,
        "target_key": target_key,
        "rule": rule,
    }


def _merge_derived(
    store: GraphStore,
    *,
    rel_type: str,
    source_type: str,
    source_key: str,
    target_type: str,
    target_key: str,
    rule: str,
    base_ref: str,
) -> dict[str, str]:
    """MERGE one derived edge, idempotently on (source, type, target, base_ref).

    ``MATCH`` (not ``MERGE``) on the endpoints: a missing endpoint yields
    no rows, so no stub node is ever created. The ``MERGE`` pattern
    carries ``{derived: true, base_ref: $base_ref}`` — re-running the
    same base write matches the existing edge instead of creating a
    parallel one. Returns the edge descriptor.
    """
    _assert_identifier(rel_type, "rel_type")
    _assert_identifier(source_type, "source_type")
    _assert_identifier(target_type, "target_type")
    cypher = (
        f"MATCH (a:{source_type} {{id: $source_key}}) "
        f"MATCH (b:{target_type} {{id: $target_key}}) "
        f"MERGE (a)-[r:{rel_type} {{derived: true, base_ref: $base_ref}}]->(b) "
        "SET r.derived = true, r.rule = $rule, r.base_ref = $base_ref"
    )
    store.run_query(
        cypher,
        {"source_key": source_key, "target_key": target_key, "rule": rule, "base_ref": base_ref},
    )
    return _descriptor(
        rel_type=rel_type,
        source_type=source_type,
        source_key=source_key,
        target_type=target_type,
        target_key=target_key,
        rule=rule,
    )


def _dedupe(descriptors: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[dict[str, str]] = []
    for item in descriptors:
        key = (
            item["type"],
            item["source_type"],
            item["source_key"],
            item["target_type"],
            item["target_key"],
            item["rule"],
        )
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _transitive_targets(
    store: GraphStore,
    *,
    source_type: str,
    source_key: str,
    target_type: str,
    target_key: str,
    base_ref: str,
    created: list[dict[str, str]],
) -> None:
    """R3 for a ``PART_OF`` base edge, both directions.

    (a) Every C with ``target -PART_OF-> C`` gives ``source -PART_OF-> C``.
    (b) Every X with ``X -PART_OF-> source`` gives ``X -PART_OF-> target``.
    Skips the degenerate cases where the derived edge would coincide
    with the base edge itself or be a self-loop.
    """
    rows_out = store.run_query(
        f"MATCH (t:{target_type} {{id: $target_key}})-[:PART_OF]->(c) "
        "RETURN c.id AS id, labels(c)[0] AS label",
        {"target_key": target_key},
    )
    for row in rows_out:
        node_key = row.get("id")
        if not isinstance(node_key, str) or not node_key:
            continue
        if node_key in (source_key, target_key):
            continue
        label = row.get("label")
        valid_label = isinstance(label, str) and _CYPHER_IDENTIFIER_RE.match(label)
        node_type = label if valid_label else target_type
        created.append(
            _merge_derived(
                store,
                rel_type=_TRANSITIVE_TYPE,
                source_type=source_type,
                source_key=source_key,
                target_type=node_type,
                target_key=node_key,
                rule="transitive:PART_OF",
                base_ref=base_ref,
            )
        )
    rows_in = store.run_query(
        f"MATCH (x)-[:PART_OF]->(s:{source_type} {{id: $source_key}}) "
        "RETURN x.id AS id, labels(x)[0] AS label",
        {"source_key": source_key},
    )
    for row in rows_in:
        node_key = row.get("id")
        if not isinstance(node_key, str) or not node_key:
            continue
        if node_key in (source_key, target_key):
            continue
        label = row.get("label")
        valid_label = isinstance(label, str) and _CYPHER_IDENTIFIER_RE.match(label)
        node_type = label if valid_label else source_type
        created.append(
            _merge_derived(
                store,
                rel_type=_TRANSITIVE_TYPE,
                source_type=node_type,
                source_key=node_key,
                target_type=target_type,
                target_key=target_key,
                rule="transitive:PART_OF",
                base_ref=base_ref,
            )
        )


def _materialize_one(
    store: GraphStore,
    *,
    relationship_type: str,
    source_type: str,
    source_key: str,
    target_type: str,
    target_key: str,
    base_ref: str,
) -> list[dict[str, str]]:
    """Shared R2/R3 expansion for one base edge. Used by write + recompute."""
    _assert_identifier(relationship_type, "relationship_type")
    _assert_identifier(source_type, "source_type")
    _assert_identifier(target_type, "target_type")
    _assert_key(source_key, "source_key")
    _assert_key(target_key, "target_key")
    _assert_key(base_ref, "base_ref")
    created: list[dict[str, str]] = []
    # R2 inverse: write one direction, derive the other.
    inverse = INVERSE_RELATIONS.get(relationship_type)
    if inverse is not None:
        created.append(
            _merge_derived(
                store,
                rel_type=inverse,
                source_type=target_type,
                source_key=target_key,
                target_type=source_type,
                target_key=source_key,
                rule=f"inverse:{relationship_type}",
                base_ref=base_ref,
            )
        )
    # R3 transitive: PART_OF only, both directions.
    if relationship_type == _TRANSITIVE_TYPE:
        _transitive_targets(
            store,
            source_type=source_type,
            source_key=source_key,
            target_type=target_type,
            target_key=target_key,
            base_ref=base_ref,
            created=created,
        )
    return _dedupe(created)


def materialize_for_write(
    store: GraphStore,
    *,
    relationship_type: str,
    source_type: str,
    source_key: str,
    target_type: str,
    target_key: str,
    base_ref: str,
) -> list[dict]:
    """Derive edges for one freshly written base edge (R2 + R3).

    Returns descriptors of the derived edges — including edges that
    already existed (the ``MERGE`` matched them), so the caller can
    report what this base edge entails.
    """
    return _materialize_one(
        store,
        relationship_type=relationship_type,
        source_type=source_type,
        source_key=source_key,
        target_type=target_type,
        target_key=target_key,
        base_ref=base_ref,
    )


def _deleted_count(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    value = rows[0].get("deleted")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def remove_derived_for_base(store: GraphStore, *, base_ref: str) -> int:
    """Delete every derived edge carrying ``base_ref``; return how many."""
    _assert_key(base_ref, "base_ref")
    rows = store.run_query(
        "MATCH ()-[r {derived: true, base_ref: $base_ref}]->() "
        "DELETE r RETURN count(r) AS deleted",
        {"base_ref": base_ref},
    )
    return _deleted_count(rows)


def _endpoint_label(
    row: dict[str, Any],
    *,
    key: str,
    label_value: Any,
    entity_key: str | None,
    entity_type: str | None,
) -> str | None:
    """Resolve one endpoint label from a base-edge row.

    Falls back to ``entity_type`` for the endpoint that is the entity
    being recomputed; returns ``None`` when the label is unknown (the
    caller skips such a row rather than building Cypher with a bad label).
    """
    if isinstance(label_value, str) and _CYPHER_IDENTIFIER_RE.match(label_value):
        return label_value
    if entity_key is not None and key == entity_key and entity_type is not None:
        return entity_type
    return None


def _rederive_from_base_rows(
    store: GraphStore,
    rows: list[dict[str, Any]],
    *,
    entity_key: str | None = None,
    entity_type: str | None = None,
) -> list[dict[str, str]]:
    """Re-run R2/R3 for each base edge row; skips rows with unusable labels."""
    created: list[dict[str, str]] = []
    for row in rows:
        rel_type = row.get("rel_type")
        source_key = row.get("source_key")
        target_key = row.get("target_key")
        if not isinstance(rel_type, str) or not _CYPHER_IDENTIFIER_RE.match(rel_type):
            continue
        if not isinstance(source_key, str) or not source_key:
            continue
        if not isinstance(target_key, str) or not target_key:
            continue
        source_type = _endpoint_label(
            row, key=source_key, label_value=row.get("source_type"),
            entity_key=entity_key, entity_type=entity_type,
        )
        target_type = _endpoint_label(
            row, key=target_key, label_value=row.get("target_type"),
            entity_key=entity_key, entity_type=entity_type,
        )
        if source_type is None or target_type is None:
            continue
        ref = row.get("base_ref")
        base_ref = ref if isinstance(ref, str) and ref else base_ref_for(
            source_type, source_key, rel_type, target_type, target_key
        )
        created.extend(
            _materialize_one(
                store,
                relationship_type=rel_type,
                source_type=source_type,
                source_key=source_key,
                target_type=target_type,
                target_key=target_key,
                base_ref=base_ref,
            )
        )
    return _dedupe(created)


def recompute_for_entity(
    store: GraphStore,
    *,
    entity_type: str,
    entity_key: str,
) -> dict:
    """Delete derived edges touching one entity, then re-derive from its base edges.

    Returns ``{"removed": n, "created": [...]}``.
    """
    _assert_identifier(entity_type, "entity_type")
    _assert_key(entity_key, "entity_key")
    rows = store.run_query(
        f"MATCH (n:{entity_type} {{id: $entity_key}})-[r {{derived: true}}]-() "
        "DELETE r RETURN count(r) AS deleted",
        {"entity_key": entity_key},
    )
    removed = _deleted_count(rows)
    base_rows = store.run_query(
        f"MATCH (n:{entity_type} {{id: $entity_key}})-[r:PART_OF|HAS_COMPONENT]-(m) "
        "WHERE r.derived IS NULL OR r.derived = false "
        "RETURN startNode(r).id AS source_key, labels(startNode(r))[0] AS source_type, "
        "type(r) AS rel_type, endNode(r).id AS target_key, "
        "labels(endNode(r))[0] AS target_type, r.base_ref AS base_ref",
        {"entity_key": entity_key},
    )
    created = _rederive_from_base_rows(
        store, base_rows, entity_key=entity_key, entity_type=entity_type
    )
    return {"removed": removed, "created": created}


def recompute_all(store: GraphStore) -> dict:
    """Delete ALL derived edges on the graph, then re-derive from every base edge.

    Returns ``{"removed": n, "created": [...]}``. Only base
    ``PART_OF`` / ``HAS_COMPONENT`` edges can derive anything in v1, so
    only those are re-read.
    """
    rows = store.run_query(
        "MATCH ()-[r {derived: true}]->() DELETE r RETURN count(r) AS deleted",
        {},
    )
    removed = _deleted_count(rows)
    base_rows = store.run_query(
        "MATCH (a)-[r:PART_OF|HAS_COMPONENT]->(b) "
        "WHERE r.derived IS NULL OR r.derived = false "
        "RETURN a.id AS source_key, labels(a)[0] AS source_type, "
        "type(r) AS rel_type, b.id AS target_key, "
        "labels(b)[0] AS target_type, r.base_ref AS base_ref",
        {},
    )
    created = _rederive_from_base_rows(store, base_rows)
    return {"removed": removed, "created": created}


def subclass_ancestors(store: GraphStore, class_id: str) -> list[str]:
    """Ancestor class ids of ``class_id`` via ``SUBCLASS_OF`` (R1, query-time).

    Breadth-first walk up ``(:OntologyClass)-[:SUBCLASS_OF]->(:OntologyClass)``.
    A ``visited`` set stops cycles (A -> B -> A terminates). The class
    itself is never included in the result.
    """
    _assert_key(class_id, "class_id")
    visited: set[str] = {class_id}
    ancestors: list[str] = []
    queue: list[str] = [class_id]
    while queue:
        current = queue.pop(0)
        rows = store.run_query(
            "MATCH (c:OntologyClass {id: $class_id})-[:SUBCLASS_OF]->(p:OntologyClass) "
            "RETURN p.id AS id",
            {"class_id": current},
        )
        for row in rows:
            parent = row.get("id")
            if not isinstance(parent, str) or not parent:
                continue
            if parent in visited:
                continue
            visited.add(parent)
            ancestors.append(parent)
            queue.append(parent)
    return ancestors


def subclass_descendants(store: GraphStore, class_id: str) -> list[str]:
    """Subclass ids of ``class_id`` via ``SUBCLASS_OF`` (R1, query-time).

    The downward mirror of :func:`subclass_ancestors`: breadth-first
    walk down ``(:OntologyClass)<-[:SUBCLASS_OF]-(:OntologyClass)``.
    Same cycle guard (``visited`` set) — a subclass cycle terminates
    instead of looping. The class itself is never included. Used by
    ``ontology.knowledge.graph_query.entities_by_class`` so a query
    for a class also returns instances of its subclasses.
    """
    _assert_key(class_id, "class_id")
    visited: set[str] = {class_id}
    descendants: list[str] = []
    queue: list[str] = [class_id]
    while queue:
        current = queue.pop(0)
        rows = store.run_query(
            "MATCH (c:OntologyClass {id: $class_id})<-[:SUBCLASS_OF]-(child:OntologyClass) "
            "RETURN child.id AS id",
            {"class_id": current},
        )
        for row in rows:
            child = row.get("id")
            if not isinstance(child, str) or not child:
                continue
            if child in visited:
                continue
            visited.add(child)
            descendants.append(child)
            queue.append(child)
    return descendants
