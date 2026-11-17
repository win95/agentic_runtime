"""Ontology schema allowlists (v1 — slim).

Single source of truth for entity/relation names: every other
module imports these names instead of re-listing them (hardcode
cleanup 2026-11-17). Renaming a label or relation touches this
file only. The same file owns the canonical status / change-type
/ outcome enums, so the write service, repository, read helpers
and projection engine all compare the same values.
"""

from __future__ import annotations

import re
from enum import Enum

# --- Named labels / relations (the canonical spellings) ---------------
ENTITY_ASSET = "Asset"
ENTITY_BUILDING = "Building"
ENTITY_COMPONENT = "Component"
ENTITY_DOCUMENT = "Document"

REL_PART_OF = "PART_OF"
REL_HAS_COMPONENT = "HAS_COMPONENT"
REL_LOCATED_IN = "LOCATED_IN"
REL_RELATED_TO = "RELATED_TO"
REL_DOCUMENTED_BY = "DOCUMENTED_BY"

ENTITY_TYPES_V1: frozenset[str] = frozenset(
    {
        ENTITY_ASSET,
        ENTITY_BUILDING,
        ENTITY_COMPONENT,
        ENTITY_DOCUMENT,
    }
)

RELATIONSHIP_TYPES_V1: frozenset[str] = frozenset(
    {
        REL_PART_OF,
        REL_HAS_COMPONENT,
        REL_LOCATED_IN,
        REL_RELATED_TO,
        REL_DOCUMENTED_BY,
    }
)

# Fallbacks shared by the retrieval graph arm and the hooks: a node
# whose type the graph never reported, and a relation whose type a
# row never named. Both are canonical spellings (RELATED_TO is a
# v1 relation; "RELATED" is not — the old hook fallback used it).
UNKNOWN_NODE_TYPE = "Entity"
DEFAULT_RELATION_TYPE = REL_RELATED_TO

MAX_LABEL_LEN = 63
_LABEL_RE = re.compile(rf"^[A-Za-z_][A-Za-z0-9_]{{0,{MAX_LABEL_LEN}}}$")


class EntityStatus(str, Enum):
    """Canonical entity/relationship row status (L1)."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

    @classmethod
    def inactive(cls) -> frozenset["EntityStatus"]:
        """Statuses that hide a row from reads and from the graph."""
        return frozenset({cls.ARCHIVED, cls.DELETED})


class ChangeType(str, Enum):
    """What a canonical write did (``OntologyVersion.change_type``).

    ``ARCHIVED``/``DELETED`` double as change types: a write that
    moves a row to an inactive status reports that status as its
    change. There is no ``"delete"`` value — a delete is an
    ``ARCHIVED``/``DELETED`` change (see ``write_service._change_type``).
    """

    CREATE = "create"
    UPDATE = "update"
    ARCHIVED = "archived"
    DELETED = "deleted"


class WriteOutcome(str, Enum):
    """A write that changed nothing (``CanonicalWriteResult.change_type``)."""

    NOOP = "noop"
    REPLAY = "replay"


class ProjectionStatus(str, Enum):
    """Projection queue row status (``ontology_projection_state``)."""

    PENDING = "pending"
    PROJECTING = "projecting"
    SYNCED = "synced"
    DEAD = "dead"


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
