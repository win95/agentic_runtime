"""Canonical ontology value types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    entity_type: str
    entity_key: str
    properties: dict[str, Any] | None
    status: str
    version: int
    state_hash: str


@dataclass
class RelationshipRecord:
    relationship_type: str
    source_type: str
    source_key: str
    target_type: str
    target_key: str
    properties: dict[str, Any] | None
    status: str
    version: int
    state_hash: str
