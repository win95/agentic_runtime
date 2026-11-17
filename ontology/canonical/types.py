"""Canonical ontology value types."""

from __future__ import annotations

from dataclasses import dataclass


def entity_ref(entity_type: str, entity_key: str) -> str:
    return f"entity:{entity_type}:{entity_key}"


def relationship_ref(
    relationship_type: str,
    source_type: str,
    source_key: str,
    target_type: str,
    target_key: str,
) -> str:
    return (
        f"rel:{relationship_type}:{source_type}:{source_key}:"
        f"{target_type}:{target_key}"
    )


@dataclass(frozen=True)
class CanonicalWriteResult:
    subject_ref: str
    version: int
    change_type: str
    applied: bool
    replayed: bool = False


@dataclass
class EntityRecord:
    """L1 state row shape (state only — no object properties and no
    state_hash anymore; the object lives on Neo4j, design §3.1)."""

    entity_type: str
    entity_key: str
    status: str
    version: int


@dataclass
class RelationshipRecord:
    """L1 state row shape for a relationship (as EntityRecord)."""

    relationship_type: str
    source_type: str
    source_key: str
    target_type: str
    target_key: str
    status: str
    version: int
