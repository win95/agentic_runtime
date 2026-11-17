"""ProjectionEngine — drain pending rows from ontology_projection_state.

The drain is a STATE RECONCILER now (design §5.4), not a projection:
L1 holds no object data, so no row carries a payload to copy. Each
outbox row is a state-sync intent — "subject (type, key) is at
version V, status S" — and the drain converges the Neo4j object to
that state:

- L1 active + object present with drifted status/version → SET them.
- L1 archived/deleted + object present → delete the object.
- L1 active + object missing → UNRECOVERABLE: L1 cannot rebuild the
  object, so the row goes dead at once (no retry) with last_error +
  a warning (operator restores from the Neo4j backup).
- A missing L1 state row (orphaned intent) → drop the object, mark done.
- An active edge whose endpoint is gone from L1 is invalid, not
  drifted: drop the edge and mark done (the old "endpoint gone"
  rule, kept).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.schema import ChangeType, EntityStatus
from ontology.graph.protocol import GraphStore
from ontology.models import OntologyProjectionState
from ontology.projection.handlers import UnrecoverableDriftError
from ontology.projection.handlers.entity import project_entity
from ontology.projection.handlers.relationship import project_relationship

logger = logging.getLogger(__name__)

# Fallback when settings carry no value (e.g. a stale settings object).
# The value itself lives in app/config.py (projection_max_attempts).
MAX_PROJECTION_ATTEMPTS = settings.projection_max_attempts

# Drain result detail when the graph store is unavailable: the same
# string app/worker/tasks.py and scripts/project_ontology_queue.py
# report, so callers can match on one name.
NEO4J_UNHEALTHY_DETAIL = "neo4j_unhealthy"

# Queue depth when counting fails (observability only — never fails
# the drain). Returned as a copy so callers cannot mutate the shared
# constant.
EMPTY_QUEUE_DEPTH: dict[str, int] = {"pending": 0, "dead_total": 0}


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
        change_type = row.change_type or ChangeType.UPDATE
        if row.subject_kind == "entity":
            etype, ekey = _parse_entity_ref(row.subject_ref)
            entity = self._repo.get_entity(etype, ekey)
            if entity is None:
                # No L1 state at all: orphaned intent — drop the node.
                store.delete_node(etype, ekey)
                self._mark_done(row, row.canonical_version)
                return
            project_entity(
                store,
                entity_type=entity.entity_type,
                entity_key=entity.entity_key,
                status=entity.status,
                version=entity.version,
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
        if rel.status in EntityStatus.inactive() or change_type in (
            ChangeType.ARCHIVED,
            ChangeType.DELETED,
        ):
            project_relationship(
                store,
                relationship_type=rel.relationship_type,
                source_type=rel.source_type,
                source_key=rel.source_key,
                target_type=rel.target_type,
                target_key=rel.target_key,
                status=rel.status,
                version=rel.version,
                change_type=change_type,
            )
            self._mark_done(row, rel.version)
            return
        # Active edge: is it on the graph?
        edge = store.get_relationship(st, sk, rtype, tt, tk)
        if edge is None:
            gone = []
            for endpoint_type, endpoint_key in ((st, sk), (tt, tk)):
                endpoint = self._repo.get_entity(endpoint_type, endpoint_key)
                if endpoint is None or endpoint.status != EntityStatus.ACTIVE:
                    gone.append((endpoint_type, endpoint_key))
            if gone:
                # Endpoint legitimately gone from canonical: the edge
                # is invalid — drop it from the graph and mark the row
                # done. (The write service never creates such an edge
                # on Neo4j; this covers edges the old projection left
                # behind and edges whose endpoint was deleted after
                # the edge was written.)
                logger.warning(
                    "Reconciler dropping relationship %s: endpoint(s) gone "
                    "from canonical: %s",
                    row.subject_ref,
                    ", ".join(f"{t}:{k}" for t, k in gone),
                )
                store.delete_relationship(st, sk, rtype, tt, tk)
                self._mark_done(row, rel.version)
                return
            # Endpoints are fine — the edge itself is lost. L1 cannot
            # rebuild it (no object data), so this is unrecoverable.
            raise UnrecoverableDriftError(
                f"Relationship {row.subject_ref} is active in L1 "
                f"(version {rel.version}) but missing on Neo4j; it "
                f"cannot be recreated from L1 state — restore from "
                f"the Neo4j backup"
            )
        project_relationship(
            store,
            relationship_type=rel.relationship_type,
            source_type=rel.source_type,
            source_key=rel.source_key,
            target_type=rel.target_type,
            target_key=rel.target_key,
            status=rel.status,
            version=rel.version,
            change_type=change_type,
        )
        self._mark_done(row, rel.version)

    def drain_batch(self, limit: int | None = None) -> dict[str, Any]:
        limit = settings.projection_drain_batch_size if limit is None else limit
        try:
            store = self._get_store()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Projection drain aborted: graph store unavailable: %s", exc)
            return {
                "processed": 0,
                "errors": 1,
                "dead": 0,
                "claimed": 0,
                "detail": NEO4J_UNHEALTHY_DETAIL,
                **self._queue_depth(),
            }
        if not store.is_healthy():
            return {
                "processed": 0,
                "errors": 1,
                "dead": 0,
                "claimed": 0,
                "detail": NEO4J_UNHEALTHY_DETAIL,
                **self._queue_depth(),
            }
        rows = self._repo.claim_pending_projections(limit=limit)
        processed = 0
        errors = 0
        dead = 0
        for row in rows:
            try:
                self.project_one(row)
                processed += 1
            except UnrecoverableDriftError as exc:
                # Missing object, L1 cannot rebuild it: dead at once
                # (retrying cannot heal this), with last_error + alert.
                errors += 1
                dead += 1
                logger.warning(
                    "Projection unrecoverable drift for %s: %s — marking dead",
                    row.subject_ref,
                    exc,
                )
                self._repo.mark_projection_dead(row, str(exc))
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
        depth = self._queue_depth()
        if depth["dead_total"] > 0:
            logger.warning(
                "Projection drain: %d dead row(s) in ontology_projection_state "
                "need a manual fix (upsert_projection_pending re-queues one)",
                depth["dead_total"],
            )
        if processed > 0:
            # B7: the ontology changed — refresh the retrieval
            # gazetteer once, so the next retriever build links the
            # new keys without re-fetching on every build. Best
            # effort only: a refresh failure must never fail the
            # drain (the gazetteer fails closed to linking nothing).
            try:
                from runtime.retrieval_graph import refresh_gazetteer_codes

                refresh_gazetteer_codes(force=True)
            except Exception as exc:  # noqa: BLE001 — refresh never fails a drain
                logger.debug("Post-drain gazetteer refresh failed: %s", exc)
        return {
            "processed": processed,
            "errors": errors,
            "dead": dead,
            "claimed": len(rows),
            **depth,
        }

    def _queue_depth(self) -> dict[str, int]:
        """Remaining queue depth (best-effort — a counting failure
        must not fail the drain)."""
        try:
            counts = self._repo.count_projection_states()
            return {"pending": counts["pending"], "dead_total": counts["dead"]}
        except Exception as exc:  # noqa: BLE001 — observability only
            logger.debug("Projection queue depth unavailable: %s", exc)
            return dict(EMPTY_QUEUE_DEPTH)


def drain_pending_projections(
    session: Session, *, limit: int | None = None, max_attempts: int | None = None
) -> dict[str, Any]:
    return ProjectionEngine(session, max_attempts=max_attempts).drain_batch(limit=limit)
