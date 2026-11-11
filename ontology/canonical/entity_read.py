"""Read helpers for canonical entities."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from ontology.canonical.repository import CanonicalOntologyRepository


def get_entity(session: Session, entity_type: str, entity_key: str) -> dict[str, Any] | None:
    row = CanonicalOntologyRepository(session).get_entity(entity_type, entity_key)
    if row is None:
        return None
    return {
        "entity_type": row.entity_type,
        "entity_key": row.entity_key,
        "properties": row.properties or {},
        "status": row.status,
        "version": row.version,
        "state_hash": row.state_hash,
    }


def list_entities(session: Session, entity_type: str, *, limit: int = 200) -> list[dict[str, Any]]:
    rows = CanonicalOntologyRepository(session).list_entities_by_type(entity_type, limit=limit)
    return [
        {
            "entity_type": r.entity_type,
            "entity_key": r.entity_key,
            "properties": r.properties or {},
            "status": r.status,
            "version": r.version,
        }
        for r in rows
    ]
