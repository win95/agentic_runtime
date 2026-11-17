"""Write-time validation against ontology metadata (design §4.1, B2).

Deliberate coordinator decision — the **v1 fallback**: when the
store holds *no* ``OntologyClass`` node at all, metadata has not
been seeded yet (bootstrap, and every fake store in the older tests),
so validation falls back to the v1 allowlists in
``ontology/canonical/schema.py`` (``ENTITY_TYPES_V1`` /
``RELATIONSHIP_TYPES_V1``). The moment at least one class node
exists, the metadata is authoritative and the allowlists stop
applying. This is what keeps the old tests — whose fake stores never
seed metadata — passing without any change to them.

Rules (metadata mode):

- Entity: the class must be a defined ``OntologyClass``. Data
  properties defined for the class — including ones whose domain is
  an ancestor of the class (R1 subclass, via ``subclass_ancestors``)
  — are checked: a present value must match the declared datatype
  (``str`` / ``int`` / ``float`` / ``bool`` mapped to Python types;
  ``bool`` is never accepted as ``int``/``float`` even though it
  subclasses ``int``; an ``int`` is accepted for ``float``), and a
  ``required`` property that is missing is rejected. A property key
  that is neither a data property defined for this class nor a system
  key is rejected — but **only when the class already has at least
  one data property defined**, so legacy data written before its
  class got metadata is not rejected for keys nobody has defined
  yet.
- Relationship: the type must be a defined ``ObjectProperty``.
  Source/target domain/range is checked against the endpoint's class
  (the node's ``class`` property when present, else its label) —
  including subclasses of the domain/range class — **but only when
  the endpoint node exists on Neo4j**. An endpoint that does not
  exist yet cannot be verified, so its side of the check is skipped
  (documented, not silently passed as "valid": the caller still gets
  the type check). ``RELATED_TO`` has domain/range ``any`` and
  always passes.

System keys (never treated as unknown data properties, never
datatype-checked): ``id`` / ``status`` / ``version`` / ``class`` /
``base_ref`` / ``derived`` / ``trace_id``. ``status`` is in the list
because demo data carries a user ``status`` value (``motor-M0``:
``"status": "retired"``) that must not be mistaken for the store's
own status property.
"""

from __future__ import annotations

from typing import Any

from ontology.canonical.schema import (
    ENTITY_TYPES_V1,
    RELATIONSHIP_TYPES_V1,
)
from ontology.graph.protocol import GraphStore
from ontology.metadata.store import ANY_CLASS, get_metadata
from ontology.reasoning import subclass_ancestors

#: Keys owned by the store layer (or the write path), never user data
#: properties. Exempt from the unknown-key rule and datatype checks.
SYSTEM_PROPERTY_KEYS = frozenset(
    {"id", "status", "version", "class", "base_ref", "derived", "trace_id"}
)

#: Declared datatype name -> acceptable Python type(s).
_DATATYPE_TYPES: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": (int, float),
    "number": (int, float),
    "bool": bool,
    "boolean": bool,
}


def _has_metadata(store: GraphStore) -> bool:
    """True once at least one OntologyClass node exists."""
    return len(get_metadata(store)["classes"]) > 0


def _applicable_data_properties(
    store: GraphStore, *, entity_type: str
) -> list[dict[str, Any]]:
    """Data properties defined for ``entity_type`` — including ones
    whose domain is an ancestor of the class (R1)."""
    meta = get_metadata(store)
    domains = {entity_type}
    domains.update(subclass_ancestors(store, entity_type))
    return [
        prop
        for prop in meta["data_properties"]
        if prop.get("domain") in domains
    ]


def _check_datatype(datatype: str, key: str, value: Any, *, entity_type: str) -> None:
    expected = _DATATYPE_TYPES.get(str(datatype).lower())
    if expected is None:
        # Unknown datatype name in metadata: do not invent a rule.
        return
    if isinstance(value, bool) and expected is not bool:
        raise ValueError(
            f"property {key!r} on {entity_type} must be {datatype}, "
            f"got bool {value!r}"
        )
    if not isinstance(value, expected):
        raise ValueError(
            f"property {key!r} on {entity_type} must be {datatype}, "
            f"got {type(value).__name__} {value!r}"
        )


def validate_entity(
    store: GraphStore, *, entity_type: str, properties: dict[str, Any] | None
) -> None:
    """Validate one entity write. Raises ValueError on violation.

    ``properties=None`` means "no payload" (a relation-stub write):
    only the class check runs — there is nothing to datatype-check.
    """
    if _has_metadata(store):
        class_ids = {row["id"] for row in get_metadata(store)["classes"]}
        if entity_type not in class_ids:
            raise ValueError(
                f"entity_type {entity_type!r} is not a defined class"
            )
    elif entity_type not in ENTITY_TYPES_V1:
        raise ValueError(f"entity_type {entity_type!r} not allowed in v1")
    if properties is None:
        return
    if not _has_metadata(store):
        # Fallback mode: no data-property definitions exist, so there
        # is nothing to datatype/required/unknown-key check against.
        return
    defined = _applicable_data_properties(store, entity_type=entity_type)
    defined_keys = {prop["id"] for prop in defined}
    for prop in defined:
        key = prop["id"]
        if prop.get("required") and key not in properties:
            raise ValueError(
                f"required data property {key!r} is missing for {entity_type}"
            )
        if key in properties:
            _check_datatype(
                str(prop.get("datatype") or "str"),
                key,
                properties[key],
                entity_type=entity_type,
            )
    if defined:
        unknown = [
            key
            for key in properties
            if key not in defined_keys and key not in SYSTEM_PROPERTY_KEYS
        ]
        if unknown:
            raise ValueError(
                f"unknown data propert{'y' if len(unknown) == 1 else 'ies'} "
                f"{sorted(unknown)} for {entity_type}"
            )


def validate_relationship(
    store: GraphStore,
    *,
    relationship_type: str,
    source_type: str,
    source_key: str,
    target_type: str,
    target_key: str,
) -> None:
    """Validate one relationship write. Raises ValueError on violation.

    Domain/range is checked only for endpoints that exist on Neo4j;
    a missing endpoint's side is skipped (it cannot be verified).
    """
    if _has_metadata(store):
        meta = get_metadata(store)
        prop = next(
            (
                row
                for row in meta["object_properties"]
                if row["id"] == relationship_type
            ),
            None,
        )
        if prop is None:
            raise ValueError(
                f"relationship_type {relationship_type!r} is not a "
                "defined object property"
            )
        class_ids = {row["id"] for row in meta["classes"]}
        for endpoint_type in (source_type, target_type):
            if endpoint_type not in class_ids:
                raise ValueError(
                    f"entity_type {endpoint_type!r} is not a defined class"
                )
        domain = prop.get("domain")
        range_ = prop.get("range")
    else:
        if relationship_type not in RELATIONSHIP_TYPES_V1:
            raise ValueError(
                f"relationship_type {relationship_type!r} not allowed in v1"
            )
        for endpoint_type in (source_type, target_type):
            if endpoint_type not in ENTITY_TYPES_V1:
                raise ValueError(
                    f"entity_type {endpoint_type!r} not allowed in v1"
                )
        # Fallback mode has no domain/range definitions to check.
        return
    for endpoint_type, endpoint_key, expected in (
        (source_type, source_key, domain),
        (target_type, target_key, range_),
    ):
        if expected is None or expected == ANY_CLASS:
            continue
        node = store.get_node(endpoint_type, endpoint_key)
        if node is None:
            # Endpoint does not exist yet: domain/range cannot be
            # verified, so this side is skipped (not assumed valid).
            continue
        actual = node.get("class") or endpoint_type
        if actual != expected and expected not in subclass_ancestors(store, actual):
            raise ValueError(
                f"{endpoint_type}:{endpoint_key} is a {actual}, not a "
                f"{expected} (domain/range of {relationship_type})"
            )


def default_entity_validator(
    store: GraphStore, *, entity_type: str, properties: dict[str, Any] | None
) -> None:
    """Entity half of the validator the write service calls: same
    signature as ``validate_entity`` (store first, the rest keyword)."""
    validate_entity(store, entity_type=entity_type, properties=properties)


def default_relationship_validator(
    store: GraphStore,
    *,
    relationship_type: str,
    source_type: str,
    source_key: str,
    target_type: str,
    target_key: str,
) -> None:
    """Relationship half of the validator the write service calls."""
    validate_relationship(
        store,
        relationship_type=relationship_type,
        source_type=source_type,
        source_key=source_key,
        target_type=target_type,
        target_key=target_key,
    )
