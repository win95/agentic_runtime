"""Ontology schema allowlists (v1 — slim)."""

from __future__ import annotations

import re

ENTITY_TYPES_V1: frozenset[str] = frozenset(
    {
        "Asset",
        "Building",
        "Component",
        "Document",
    }
)

RELATIONSHIP_TYPES_V1: frozenset[str] = frozenset(
    {
        "PART_OF",
        "HAS_COMPONENT",
        "LOCATED_IN",
        "RELATED_TO",
        "DOCUMENTED_BY",
    }
)

_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def normalize_entity_label(raw: str) -> str:
    value = (raw or "").strip()
    if not _LABEL_RE.match(value):
        raise ValueError(f"Invalid entity_type identifier: {raw!r}")
    if value not in ENTITY_TYPES_V1:
        raise ValueError(f"entity_type {raw!r} not allowed in v1")
    return value


def assert_relationship_type(relationship_type: str) -> str:
    value = (relationship_type or "").strip().upper()
    if value not in RELATIONSHIP_TYPES_V1:
        raise ValueError(f"relationship_type {relationship_type!r} not allowed in v1")
    return value
