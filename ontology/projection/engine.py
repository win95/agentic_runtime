"""ProjectionEngine — drain pending rows from ontology_projection_state."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.graph.protocol import GraphStore
from ontology.models import OntologyProjectionState
from ontology.projection.handlers.entity import project_entity
from ontology.projection.handlers.relationship import project_relationship

logger = logging.getLogger(__name__)

# Fallback when settings carry no value (e.g. a stale settings object).
MAX_PROJECTION_ATTEMPTS = 5


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
    def __init__(
        self,
        session: Session,
        *,
        store: GraphStore | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self._session = session
        self._repo = CanonicalOntologyRepository(session)
        # Built lazily via _get_store() so drain_batch can turn a
        # store-construction failure into the unhealthy result instead
        # of crashing the beat task / CLI.
        self._store = store
        if max_attempts is not None:
            self._max_attempts = max_attempts
        else:
            self._max_attempts = int(
                getattr(settings, "projection_max_attempts", MAX_PROJECTION_ATTEMPTS)
            )

    def _get_store(self) -> GraphStore:
        if self._store is None:
            # Imported here (not at module import) so tests can patch
            # ``ontology.projection.provider.get_graph_store`` and so a
            # construction failure is raised inside drain_batch's guard.
            from ontology.projection.provider import get_graph_store

            self._store = get_graph_store()
        return self._store

    def _mark_done(self, row: OntologyProjectionState, projected_version: int) -> None:
        # The mark is conditional on canonical_version still equalling
        # the projected version; False means a newer canonical write
        # landed mid-projection and the row stays pending for re-drain.
        if not self._repo.mark_projection_done(row, projected_version):
            logger.info(
                "Projection mark skipped (stale version) for %s; left pending",
                row.subject_ref,
            )

    def project_one(self, row: OntologyProjectionState) -> None:
        store = self._get_store()
        change_type = row.change_type or "update"
        if row.subject_kind == "entity":
            etype, ekey = _parse_entity_ref(row.subject_ref)
            entity = self._repo.get_entity(etype, ekey)
            if entity is None:
                store.delete_node(etype, ekey)
                self._mark_done(row, row.canonical_version)
                return
            props = dict(entity.properties or {})
            props.update(
                {
                    "status": entity.status,
                    "trace_id": entity.trace_id,
                    "canonical_version": entity.version,
                }
            )
            project_entity(
                store,
                entity_type=entity.entity_type,
                entity_key=entity.entity_key,
                properties=props,
                status=entity.status,
                change_type=change_type,
            )
            self._mark_done(row, entity.version)
            return

        rtype, st, sk, tt, tk = _parse_relationship_ref(row.subject_ref)
        rel = self._repo.get_relationship(rtype, st, sk, tt, tk)
        if rel is None:
            store.delete_relationship(st, sk, rtype, tt, tk)
            self._mark_done(row, row.canonical_version)
            return
        if rel.status in ("deleted", "archived") or change_type == "delete":
            project_relationship(
                store,
                relationship_type=rel.relationship_type,
                source_type=rel.source_type,
                source_key=rel.source_key,
                target_type=rel.target_type,
                target_key=rel.target_key,
                properties=rel.properties,
                status=rel.status,
                change_type=change_type,
            )
            self._mark_done(row, rel.version)
            return
        # Active edge: both endpoints must already exist in the graph —
        # merge_relationship never creates endpoint stubs.
        missing = [
            (etype, ekey)
            for (etype, ekey) in ((st, sk), (tt, tk))
            if not store.node_exists(etype, ekey)
        ]
        if missing:
            gone = []
            for endpoint_type, endpoint_key in missing:
                endpoint = self._repo.get_entity(endpoint_type, endpoint_key)
                if endpoint is None or endpoint.status != "active":
                    gone.append((endpoint_type, endpoint_key))
            if gone:
                # Endpoint legitimately gone from canonical: the edge is
                # invalid — drop it from the graph and mark the row done.
                logger.warning(
                    "Projection dropping relationship %s: endpoint(s) gone from canonical: %s",
                    row.subject_ref,
                    ", ".join(f"{t}:{k}" for t, k in gone),
                )
                store.delete_relationship(st, sk, rtype, tt, tk)
                self._mark_done(row, rel.version)
                return
            # Endpoint active in canonical but not yet projected: fail
            # the row so the drain retries it (its own pending row will
            # project the endpoint first).
            raise RuntimeError(
                "Relationship endpoint(s) not yet projected for "
                f"{row.subject_ref}: " + ", ".join(f"{t}:{k}" for t, k in missing)
            )
        props = dict(rel.properties or {})
        props.update(
            {
                "status": rel.status,
                "trace_id": rel.trace_id,
                "canonical_version": rel.version,
            }
        )
        project_relationship(
            store,
            relationship_type=rel.relationship_type,
            source_type=rel.source_type,
            source_key=rel.source_key,
            target_type=rel.target_type,
            target_key=rel.target_key,
            properties=props,
            status=rel.status,
            change_type=change_type,
        )
        self._mark_done(row, rel.version)

    def drain_batch(self, limit: int = 32) -> dict[str, Any]:
        try:
            store = self._get_store()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Projection drain aborted: graph store unavailable: %s", exc)
            return {
                "processed": 0,
                "errors": 1,
                "dead": 0,
                "claimed": 0,
                "detail": "neo4j_unhealthy",
            }
        if not store.is_healthy():
            return {
                "processed": 0,
                "errors": 1,
                "dead": 0,
                "claimed": 0,
                "detail": "neo4j_unhealthy",
            }
        rows = self._repo.claim_pending_projections(limit=limit)
        processed = 0
        errors = 0
        dead = 0
        for row in rows:
            try:
                self.project_one(row)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                logger.exception("Projection failed for %s", row.subject_ref)
                if (row.attempts or 0) >= self._max_attempts:
                    dead += 1
                    logger.warning(
                        "Projection poisoned %s after %d attempt(s); marking dead, last error: %s",
                        row.subject_ref,
                        row.attempts,
                        exc,
                    )
                    self._repo.mark_projection_dead(row, str(exc))
                else:
                    self._repo.mark_projection_failed(row, str(exc))
        self._session.commit()
        return {"processed": processed, "errors": errors, "dead": dead, "claimed": len(rows)}


def drain_pending_projections(
    session: Session, *, limit: int = 32, max_attempts: int | None = None
) -> dict[str, Any]:
    return ProjectionEngine(session, max_attempts=max_attempts).drain_batch(limit=limit)
