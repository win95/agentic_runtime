"""ProjectionEngine — drain pending rows from ontology_projection_state."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.models import OntologyProjectionState
from ontology.projection.handlers.entity import project_entity
from ontology.projection.handlers.relationship import project_relationship
from ontology.projection.provider import get_graph_store

logger = logging.getLogger(__name__)


def _parse_entity_ref(subject_ref: str) -> tuple[str, str]:
    # entity:Asset:O21
    parts = subject_ref.split(":", 2)
    if len(parts) != 3 or parts[0] != "entity":
        raise ValueError(f"Bad entity subject_ref: {subject_ref}")
    return parts[1], parts[2]


def _parse_relationship_ref(subject_ref: str) -> tuple[str, str, str, str, str]:
    # rel:PART_OF:Asset:O21:Building:building-B1
    parts = subject_ref.split(":")
    if len(parts) != 6 or parts[0] != "rel":
        raise ValueError(f"Bad relationship subject_ref: {subject_ref}")
    return parts[1], parts[2], parts[3], parts[4], parts[5]


class ProjectionEngine:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._repo = CanonicalOntologyRepository(session)
        self._store = get_graph_store()

    def project_one(self, row: OntologyProjectionState) -> None:
        change_type = row.change_type or "update"
        if row.subject_kind == "entity":
            etype, ekey = _parse_entity_ref(row.subject_ref)
            entity = self._repo.get_entity(etype, ekey)
            if entity is None:
                self._store.delete_node(etype, ekey)
                self._repo.mark_projection_done(row, row.canonical_version)
                return
            project_entity(
                self._store,
                entity_type=entity.entity_type,
                entity_key=entity.entity_key,
                properties=entity.properties,
                status=entity.status,
                change_type=change_type,
            )
            self._repo.mark_projection_done(row, entity.version)
            return

        rtype, st, sk, tt, tk = _parse_relationship_ref(row.subject_ref)
        rel = self._repo.get_relationship(rtype, st, sk, tt, tk)
        if rel is None:
            self._store.delete_relationship(st, sk, rtype, tt, tk)
            self._repo.mark_projection_done(row, row.canonical_version)
            return
        project_relationship(
            self._store,
            relationship_type=rel.relationship_type,
            source_type=rel.source_type,
            source_key=rel.source_key,
            target_type=rel.target_type,
            target_key=rel.target_key,
            properties=rel.properties,
            status=rel.status,
            change_type=change_type,
        )
        self._repo.mark_projection_done(row, rel.version)

    def drain_batch(self, limit: int = 32) -> dict[str, Any]:
        if not self._store.is_healthy():
            return {"processed": 0, "errors": 1, "detail": "neo4j_unhealthy"}
        rows = self._repo.claim_pending_projections(limit=limit)
        processed = 0
        errors = 0
        for row in rows:
            try:
                self.project_one(row)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                logger.exception("Projection failed for %s", row.subject_ref)
                self._repo.mark_projection_failed(row, str(exc))
        self._session.commit()
        return {"processed": processed, "errors": errors, "claimed": len(rows)}


def drain_pending_projections(session: Session, *, limit: int = 32) -> dict[str, Any]:
    return ProjectionEngine(session).drain_batch(limit=limit)
