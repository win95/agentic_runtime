"""Ontology metadata on Neo4j (design §4.1, B2).

Class, object property and data property definitions live as nodes on
the graph — not as constants in ``ontology/canonical/schema.py``:

- ``(:OntologyClass {id, name, description})``, subclass as
  ``(:OntologyClass)-[:SUBCLASS_OF]->(:OntologyClass)``.
- ``(:ObjectProperty {id, domain, range, inverse_of})`` — ``domain`` /
  ``range`` are class ids, or ``"any"`` for an unconstrained endpoint
  (``RELATED_TO``).
- ``(:DataProperty {id, domain, datatype, required})`` — one node per
  (property key, domain class) pair: the same key (e.g. ``name``) can
  be defined for several classes with different datatypes, so the
  MERGE key is ``(id, domain)``, not ``id`` alone. The design doc
  (§4.1) shows a single ``domain`` per node and never addresses a key
  shared by several classes; the composite key is the resolution
  chosen here (see ``create_data_property``).

All access goes through ``store.run_query(cypher, params)`` only
(the ``GraphStore`` protocol) — no driver import. Every value
interpolated into Cypher (a class id used as a node label, a
relationship type) is validated against
``^[A-Za-z][A-Za-z0-9_]*$`` first; everything else travels as a
Cypher parameter.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ontology.graph.protocol import GraphStore
from ontology.reasoning import subclass_ancestors

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

#: Domain/range value meaning "any class" (used by RELATED_TO).
ANY_CLASS = "any"

#: Datatype names accepted on a DataProperty node (validated at
#: create/update time; ``validate.py`` maps them to Python types).
DATATYPE_NAMES = frozenset(
    {"str", "int", "float", "bool", "string", "integer", "number", "boolean"}
)

_CACHE_TTL_SECONDS = 60.0

# id(store) -> (store, expires_at, payload). The strong reference to
# the store is deliberate: it stops a garbage-collected store's id
# from being reused by a different store while its entry is cached,
# which would leak one store's metadata into another's reads.
# Entries are evicted on insert once expired; ``invalidate`` /
# ``clear`` drop the whole cache.
_CACHE: dict[int, tuple[GraphStore, float, dict[str, list[dict[str, Any]]]]] = {}


def _assert_identifier(name: Any, field: str) -> str:
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid Cypher identifier for {field}: {name!r}")
    return name


def _assert_class_id(name: Any, field: str) -> str:
    """A class id (used as a label for instances, so it must be a
    valid Cypher identifier) — or ``"any"`` for a domain/range."""
    if name == ANY_CLASS:
        return name
    return _assert_identifier(name, field)


def _assert_datatype(datatype: Any) -> str:
    if not isinstance(datatype, str) or datatype not in DATATYPE_NAMES:
        raise ValueError(
            f"Invalid datatype {datatype!r}: expected one of "
            f"{sorted(DATATYPE_NAMES)}"
        )
    return datatype


def _row(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: row.get(key) for key in keys}


def _data_row(row: dict[str, Any]) -> dict[str, Any]:
    """A DataProperty row with ``required`` coerced to bool (a node
    written before the field existed returns None for it, which
    Pydantic would reject as a bool)."""
    mapped = _row(row, _DATA_PROPERTY_KEYS)
    mapped["required"] = bool(mapped.get("required"))
    return mapped


def _first_or_synth(
    rows: list[dict[str, Any]], keys: list[str], synth: dict[str, Any]
) -> dict[str, Any]:
    """First returned row mapped to ``keys``, or ``synth`` when the
    store returned nothing (a fake store records the Cypher but
    returns no rows — the call still succeeded)."""
    if rows:
        return _row(rows[0], keys)
    return dict(synth)


def _count(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    try:
        return int(rows[0].get("n") or 0)
    except (TypeError, ValueError):
        return 0


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------


def get_metadata(
    store: GraphStore, *, force_refresh: bool = False
) -> dict[str, list[dict[str, Any]]]:
    """All ontology metadata, cached in-process for 60s.

    The cache is keyed by the store instance (``id(store)``) and holds
    a strong reference to it, so two different stores — e.g. two fake
    stores in two tests — never share an entry. Every mutation in
    this module calls :func:`invalidate_metadata_cache`.
    """
    now = time.monotonic()
    key = id(store)
    entry = _CACHE.get(key)
    if not force_refresh and entry is not None:
        cached_store, expires_at, payload = entry
        if cached_store is store and now < expires_at:
            return {name: [dict(row) for row in rows] for name, rows in payload.items()}
    payload = {
        "classes": list_classes(store),
        "object_properties": list_object_properties(store),
        "data_properties": list_data_properties(store),
    }
    for cached_key in [
        k for k, (_, expires_at, _) in _CACHE.items() if now >= expires_at
    ]:
        del _CACHE[cached_key]
    _CACHE[key] = (store, now + _CACHE_TTL_SECONDS, payload)
    return {name: [dict(row) for row in rows] for name, rows in payload.items()}


def invalidate_metadata_cache() -> None:
    """Drop all cached metadata (called by every mutation below)."""
    _CACHE.clear()


def clear_metadata_cache() -> None:
    """Test helper — same as :func:`invalidate_metadata_cache`."""
    _CACHE.clear()


# ----------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------

_CLASS_KEYS = ["id", "name", "description"]


def list_classes(store: GraphStore) -> list[dict[str, Any]]:
    rows = store.run_query(
        "MATCH (c:OntologyClass) "
        "RETURN c.id AS id, c.name AS name, c.description AS description "
        "ORDER BY c.id",
    )
    return [_row(row, _CLASS_KEYS) for row in rows]


def get_class(store: GraphStore, class_id: str) -> dict[str, Any] | None:
    _assert_identifier(class_id, "class_id")
    rows = store.run_query(
        "MATCH (c:OntologyClass {id: $id}) "
        "RETURN c.id AS id, c.name AS name, c.description AS description",
        {"id": class_id},
    )
    if not rows:
        return None
    return _row(rows[0], _CLASS_KEYS)


def create_class(
    store: GraphStore,
    *,
    class_id: str,
    name: str,
    description: str | None = None,
) -> dict[str, Any]:
    """MERGE the class node by id (idempotent — re-creating updates
    name/description instead of duplicating the node)."""
    _assert_identifier(class_id, "class_id")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("class name must be a non-empty string")
    rows = store.run_query(
        "MERGE (c:OntologyClass {id: $id}) "
        "SET c.name = $name, c.description = $description "
        "RETURN c.id AS id, c.name AS name, c.description AS description",
        {"id": class_id, "name": name, "description": description},
    )
    invalidate_metadata_cache()
    return _first_or_synth(
        rows, _CLASS_KEYS, {"id": class_id, "name": name, "description": description}
    )


def update_class(
    store: GraphStore,
    *,
    class_id: str,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """PATCH a class: only the fields passed (non-None) are changed.
    Raises ValueError when the class does not exist."""
    _assert_identifier(class_id, "class_id")
    current = get_class(store, class_id)
    if current is None:
        raise ValueError(f"class {class_id!r} not found")
    assignments = []
    params: dict[str, Any] = {"id": class_id}
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("class name must be a non-empty string")
        assignments.append("c.name = $name")
        params["name"] = name
    if description is not None:
        assignments.append("c.description = $description")
        params["description"] = description
    if assignments:
        store.run_query(
            "MATCH (c:OntologyClass {id: $id}) SET " + ", ".join(assignments) + " "
            "RETURN c.id AS id, c.name AS name, c.description AS description",
            params,
        )
    invalidate_metadata_cache()
    updated = dict(current)
    if name is not None:
        updated["name"] = name
    if description is not None:
        updated["description"] = description
    return updated


def delete_class(store: GraphStore, class_id: str) -> dict[str, Any]:
    """Delete a class node (DETACH DELETE its SUBCLASS_OF edges).

    Refused (ValueError) while a subclass still points at it or an
    instance still carries the class as its label — deleting either
    would silently orphan graph data the metadata used to describe.
    Raises ValueError when the class does not exist.
    """
    _assert_identifier(class_id, "class_id")
    current = get_class(store, class_id)
    if current is None:
        raise ValueError(f"class {class_id!r} not found")
    subclass_rows = store.run_query(
        "MATCH (c:OntologyClass {id: $id})<-[:SUBCLASS_OF]-(child:OntologyClass) "
        "RETURN count(child) AS n",
        {"id": class_id},
    )
    n_subclasses = _count(subclass_rows)
    if n_subclasses:
        raise ValueError(
            f"class {class_id!r} still has {n_subclasses} subclass(es); "
            "remove them first"
        )
    instance_rows = store.run_query(
        f"MATCH (n:{class_id}) RETURN count(n) AS n",
    )
    n_instances = _count(instance_rows)
    if n_instances:
        raise ValueError(
            f"class {class_id!r} still has {n_instances} instance(s); "
            "delete them first"
        )
    store.run_query(
        "MATCH (c:OntologyClass {id: $id}) DETACH DELETE c",
        {"id": class_id},
    )
    invalidate_metadata_cache()
    return current


def add_subclass(store: GraphStore, *, child_id: str, parent_id: str) -> dict[str, Any]:
    """Add ``child -SUBCLASS_OF-> parent``.

    Raises ValueError when either class is missing, or when the edge
    would create a cycle: the parent must not be the child itself or
    a descendant of the child (checked via ``subclass_ancestors`` on
    the parent). Re-adding an existing edge is a noop.
    """
    _assert_identifier(child_id, "child_id")
    _assert_identifier(parent_id, "parent_id")
    if get_class(store, child_id) is None:
        raise ValueError(f"class {child_id!r} not found")
    if get_class(store, parent_id) is None:
        raise ValueError(f"class {parent_id!r} not found")
    if child_id == parent_id:
        raise ValueError(
            f"class {child_id!r} cannot be a subclass of itself"
        )
    if child_id in subclass_ancestors(store, parent_id):
        raise ValueError(
            f"adding {child_id!r} -SUBCLASS_OF-> {parent_id!r} would "
            "create a cycle"
        )
    store.run_query(
        "MATCH (c:OntologyClass {id: $child_id}), "
        "(p:OntologyClass {id: $parent_id}) "
        "MERGE (c)-[:SUBCLASS_OF]->(p)",
        {"child_id": child_id, "parent_id": parent_id},
    )
    invalidate_metadata_cache()
    return {"child_id": child_id, "parent_id": parent_id}


def remove_subclass(store: GraphStore, *, child_id: str, parent_id: str) -> dict[str, Any]:
    """Delete ``child -SUBCLASS_OF-> parent`` (noop when absent)."""
    _assert_identifier(child_id, "child_id")
    _assert_identifier(parent_id, "parent_id")
    store.run_query(
        "MATCH (c:OntologyClass {id: $child_id})-[r:SUBCLASS_OF]->"
        "(p:OntologyClass {id: $parent_id}) DELETE r",
        {"child_id": child_id, "parent_id": parent_id},
    )
    invalidate_metadata_cache()
    return {"child_id": child_id, "parent_id": parent_id}


# ----------------------------------------------------------------------
# Object properties
# ----------------------------------------------------------------------

_OBJECT_PROPERTY_KEYS = ["id", "domain", "range", "inverse_of"]


def list_object_properties(store: GraphStore) -> list[dict[str, Any]]:
    rows = store.run_query(
        "MATCH (p:ObjectProperty) "
        "RETURN p.id AS id, p.domain AS domain, p.range AS range, "
        "p.inverse_of AS inverse_of ORDER BY p.id",
    )
    return [_row(row, _OBJECT_PROPERTY_KEYS) for row in rows]


def get_object_property(store: GraphStore, property_id: str) -> dict[str, Any] | None:
    _assert_identifier(property_id, "property_id")
    rows = store.run_query(
        "MATCH (p:ObjectProperty {id: $id}) "
        "RETURN p.id AS id, p.domain AS domain, p.range AS range, "
        "p.inverse_of AS inverse_of",
        {"id": property_id},
    )
    if not rows:
        return None
    return _row(rows[0], _OBJECT_PROPERTY_KEYS)


def create_object_property(
    store: GraphStore,
    *,
    property_id: str,
    domain: str,
    range: str,
    inverse_of: str | None = None,
) -> dict[str, Any]:
    """MERGE the object-property node by id (idempotent)."""
    _assert_identifier(property_id, "property_id")
    _assert_class_id(domain, "domain")
    _assert_class_id(range, "range")
    if inverse_of is not None:
        _assert_identifier(inverse_of, "inverse_of")
    rows = store.run_query(
        "MERGE (p:ObjectProperty {id: $id}) "
        "SET p.domain = $domain, p.range = $range, p.inverse_of = $inverse_of "
        "RETURN p.id AS id, p.domain AS domain, p.range AS range, "
        "p.inverse_of AS inverse_of",
        {
            "id": property_id,
            "domain": domain,
            "range": range,
            "inverse_of": inverse_of,
        },
    )
    invalidate_metadata_cache()
    return _first_or_synth(
        rows,
        _OBJECT_PROPERTY_KEYS,
        {
            "id": property_id,
            "domain": domain,
            "range": range,
            "inverse_of": inverse_of,
        },
    )


def update_object_property(
    store: GraphStore,
    *,
    property_id: str,
    domain: str | None = None,
    range: str | None = None,
    inverse_of: str | None = None,
) -> dict[str, Any]:
    """PATCH an object property (only non-None fields change).
    Raises ValueError when the property does not exist."""
    _assert_identifier(property_id, "property_id")
    current = get_object_property(store, property_id)
    if current is None:
        raise ValueError(f"object property {property_id!r} not found")
    assignments = []
    params: dict[str, Any] = {"id": property_id}
    if domain is not None:
        _assert_class_id(domain, "domain")
        assignments.append("p.domain = $domain")
        params["domain"] = domain
    if range is not None:
        _assert_class_id(range, "range")
        assignments.append("p.range = $range")
        params["range"] = range
    if inverse_of is not None:
        _assert_identifier(inverse_of, "inverse_of")
        assignments.append("p.inverse_of = $inverse_of")
        params["inverse_of"] = inverse_of
    if assignments:
        store.run_query(
            "MATCH (p:ObjectProperty {id: $id}) SET " + ", ".join(assignments) + " "
            "RETURN p.id AS id, p.domain AS domain, p.range AS range, "
            "p.inverse_of AS inverse_of",
            params,
        )
    invalidate_metadata_cache()
    updated = dict(current)
    for key, value in (
        ("domain", domain),
        ("range", range),
        ("inverse_of", inverse_of),
    ):
        if value is not None:
            updated[key] = value
    return updated


def delete_object_property(store: GraphStore, property_id: str) -> dict[str, Any]:
    """Delete an object-property node.

    Refused (ValueError) while any edge on the graph still uses the
    type — including derived edges. Raises ValueError when the
    property does not exist.
    """
    _assert_identifier(property_id, "property_id")
    current = get_object_property(store, property_id)
    if current is None:
        raise ValueError(f"object property {property_id!r} not found")
    in_use_rows = store.run_query(
        f"MATCH ()-[r:{property_id}]->() RETURN count(r) AS n",
    )
    n_in_use = _count(in_use_rows)
    if n_in_use:
        raise ValueError(
            f"object property {property_id!r} is still used by "
            f"{n_in_use} edge(s); delete them first"
        )
    store.run_query(
        "MATCH (p:ObjectProperty {id: $id}) DELETE p",
        {"id": property_id},
    )
    invalidate_metadata_cache()
    return current


# ----------------------------------------------------------------------
# Data properties
# ----------------------------------------------------------------------

_DATA_PROPERTY_KEYS = ["id", "domain", "datatype", "required"]


def list_data_properties(store: GraphStore) -> list[dict[str, Any]]:
    rows = store.run_query(
        "MATCH (d:DataProperty) "
        "RETURN d.id AS id, d.domain AS domain, d.datatype AS datatype, "
        "d.required AS required ORDER BY d.domain, d.id",
    )
    return [_data_row(row) for row in rows]


def get_data_property(store: GraphStore, property_id: str) -> dict[str, Any] | None:
    """First definition of ``property_id`` (lowest domain first).
    A key defined for several classes has several definitions —
    see ``create_data_property``."""
    _assert_identifier(property_id, "property_id")
    rows = store.run_query(
        "MATCH (d:DataProperty {id: $id}) "
        "RETURN d.id AS id, d.domain AS domain, d.datatype AS datatype, "
        "d.required AS required",
        {"id": property_id},
    )
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: str(row.get("domain") or ""))
    return _data_row(rows[0])


def create_data_property(
    store: GraphStore,
    *,
    property_id: str,
    domain: str,
    datatype: str,
    required: bool = False,
) -> dict[str, Any]:
    """MERGE the data-property node by (id, domain) (idempotent).

    The composite key is deliberate: the same property key (``name``)
    is defined for Asset, Building and Component with the same
    datatype today, but nothing stops two classes from typing one key
    differently, and MERGE-by-id alone would collapse those
    definitions into one row.
    """
    _assert_identifier(property_id, "property_id")
    _assert_identifier(domain, "domain")
    _assert_datatype(datatype)
    rows = store.run_query(
        "MERGE (d:DataProperty {id: $id, domain: $domain}) "
        "SET d.datatype = $datatype, d.required = $required "
        "RETURN d.id AS id, d.domain AS domain, d.datatype AS datatype, "
        "d.required AS required",
        {
            "id": property_id,
            "domain": domain,
            "datatype": datatype,
            "required": bool(required),
        },
    )
    invalidate_metadata_cache()
    created = _first_or_synth(
        rows,
        _DATA_PROPERTY_KEYS,
        {
            "id": property_id,
            "domain": domain,
            "datatype": datatype,
            "required": bool(required),
        },
    )
    created["required"] = bool(created.get("required"))
    return created


def update_data_property(
    store: GraphStore,
    *,
    property_id: str,
    domain: str | None = None,
    datatype: str | None = None,
    required: bool | None = None,
) -> dict[str, Any]:
    """PATCH a data property.

    ``domain`` here is a *filter*, not a new value: it narrows the
    update to the definition for that class (a key defined for
    several classes updates every definition when ``domain`` is
    omitted). Only ``datatype`` / ``required`` change. Raises
    ValueError when no definition matches.
    """
    _assert_identifier(property_id, "property_id")
    if domain is not None:
        _assert_identifier(domain, "domain")
    if datatype is not None:
        _assert_datatype(datatype)
    current = get_data_property(store, property_id)
    if current is None:
        raise ValueError(f"data property {property_id!r} not found")
    if domain is not None and current.get("domain") != domain:
        # The (id, domain) definition does not exist — but another
        # domain's definition of the same key might. Check directly.
        rows = store.run_query(
            "MATCH (d:DataProperty {id: $id, domain: $domain}) "
            "RETURN d.id AS id, d.domain AS domain, d.datatype AS datatype, "
            "d.required AS required",
            {"id": property_id, "domain": domain},
        )
        if not rows:
            raise ValueError(
                f"data property {property_id!r} for domain {domain!r} not found"
            )
        current = _data_row(rows[0])
    assignments = []
    params: dict[str, Any] = {"id": property_id}
    if domain is not None:
        params["domain"] = domain
    if datatype is not None:
        assignments.append("d.datatype = $datatype")
        params["datatype"] = datatype
    if required is not None:
        assignments.append("d.required = $required")
        params["required"] = bool(required)
    if assignments:
        where = " WHERE d.domain = $domain" if domain is not None else ""
        store.run_query(
            "MATCH (d:DataProperty {id: $id})" + where + " SET "
            + ", ".join(assignments) + " "
            "RETURN d.id AS id, d.domain AS domain, d.datatype AS datatype, "
            "d.required AS required",
            params,
        )
    invalidate_metadata_cache()
    updated = dict(current)
    if datatype is not None:
        updated["datatype"] = datatype
    if required is not None:
        updated["required"] = bool(required)
    return updated


def delete_data_property(store: GraphStore, property_id: str) -> dict[str, Any]:
    """Delete every definition of ``property_id`` (all domains).
    Raises ValueError when the property does not exist."""
    _assert_identifier(property_id, "property_id")
    current = get_data_property(store, property_id)
    if current is None:
        raise ValueError(f"data property {property_id!r} not found")
    store.run_query(
        "MATCH (d:DataProperty {id: $id}) DELETE d",
        {"id": property_id},
    )
    invalidate_metadata_cache()
    return current
