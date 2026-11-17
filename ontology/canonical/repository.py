"""Sync canonical ontology repository — state only (design §3).

L1 keeps no object data: this repository only reads/writes state
rows (status, version, trace/provenance) and the projection-state
(outbox) rows. Object reads/writes go through the Neo4j store.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.config import settings
from ontology.canonical.schema import ChangeType, ProjectionStatus
from ontology.models import (
    OntologyEntity,
    OntologyProjectionState,
    OntologyRelationship,
)

logger = logging.getLogger(__name__)

# Projection error text kept on a dead/failed row (operator inspection).
MAX_ERROR_LEN = 4000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CanonicalOntologyRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def session(self) -> Session:
        """Underlying session (for SAVEPOINTs / expunge in the write service)."""
        return self._session

    def get_entity(self, entity_type: str, entity_key: str) -> OntologyEntity | None:
        stmt = select(OntologyEntity).where(
            OntologyEntity.entity_type == entity_type,
            OntologyEntity.entity_key == entity_key,
        )
        return self._session.scalars(stmt).first()

    def get_entity_by_source_event_id(
        self, entity_type: str, entity_key: str, source_event_id: str
    ) -> OntologyEntity | None:
        """The state row for THIS entity whose last write carried
        ``source_event_id`` — the replay lookup (design §5 step 2).

        Scoped to the subject on purpose: callers share one event id
        across several subjects (ingest stamps one id on both endpoints
        and the edge), so a global lookup could return the wrong row.
        """
        stmt = select(OntologyEntity).where(
            OntologyEntity.entity_type == entity_type,
            OntologyEntity.entity_key == entity_key,
            OntologyEntity.source_event_id == source_event_id,
        )
        return self._session.scalars(stmt).first()

    def get_relationship(
        self,
        relationship_type: str,
        source_type: str,
        source_key: str,
        target_type: str,
        target_key: str,
    ) -> OntologyRelationship | None:
        stmt = select(OntologyRelationship).where(
            OntologyRelationship.relationship_type == relationship_type,
            OntologyRelationship.source_type == source_type,
            OntologyRelationship.source_key == source_key,
            OntologyRelationship.target_type == target_type,
            OntologyRelationship.target_key == target_key,
        )
        return self._session.scalars(stmt).first()

    def get_relationship_by_source_event_id(
        self,
        relationship_type: str,
        source_type: str,
        source_key: str,
        target_type: str,
        target_key: str,
        source_event_id: str,
    ) -> OntologyRelationship | None:
        """The state row for THIS edge whose last write carried
        ``source_event_id`` — the replay lookup (design §5 step 2),
        scoped to the subject like ``get_entity_by_source_event_id``."""
        stmt = select(OntologyRelationship).where(
            OntologyRelationship.relationship_type == relationship_type,
            OntologyRelationship.source_type == source_type,
            OntologyRelationship.source_key == source_key,
            OntologyRelationship.target_type == target_type,
            OntologyRelationship.target_key == target_key,
            OntologyRelationship.source_event_id == source_event_id,
        )
        return self._session.scalars(stmt).first()

    def save_entity(self, row: OntologyEntity) -> OntologyEntity:
        self._session.add(row)
        self._session.flush()
        return row

    def save_relationship(self, row: OntologyRelationship) -> OntologyRelationship:
        self._session.add(row)
        self._session.flush()
        return row

    def upsert_projection_pending(
        self,
        *,
        subject_kind: str,
        subject_ref: str,
        canonical_version: int,
        change_type: str | ChangeType,
        trace_id: str | None,
    ) -> None:
        stmt = select(OntologyProjectionState).where(
            OntologyProjectionState.subject_ref == subject_ref
        )
        row = self._session.scalars(stmt).first()
        now = _utcnow()
        if row is None:
            self._session.add(
                OntologyProjectionState(
                    subject_kind=subject_kind,
                    subject_ref=subject_ref,
                    canonical_version=canonical_version,
                    projected_version=None,
                    status=ProjectionStatus.PENDING,
                    change_type=change_type,
                    attempts=0,
                    last_enqueued_at=now,
                    trace_id=trace_id,
                )
            )
        else:
            # Re-queueing a dead row must reset its attempt count: the
            # poison policy marks a row dead once attempts reach the
            # limit, and attempts are only incremented on claim, so
            # without this reset a re-queued row would go dead again on
            # its very next failure and could never recover.
            if row.status == ProjectionStatus.DEAD:
                row.attempts = 0
            row.canonical_version = canonical_version
            row.status = ProjectionStatus.PENDING
            row.change_type = change_type
            row.last_enqueued_at = now
            row.trace_id = trace_id
            row.last_error = None
        self._session.flush()

    def count_projection_states(self) -> dict[str, int]:
        """Queue depth for observability: remaining ``pending`` rows
        and terminal ``dead`` rows (poison rows need a manual fix —
        upsert_projection_pending re-queues one)."""
        pending = self._session.scalar(
            select(func.count())
            .select_from(OntologyProjectionState)
            .where(OntologyProjectionState.status == ProjectionStatus.PENDING)
        )
        dead = self._session.scalar(
            select(func.count())
            .select_from(OntologyProjectionState)
            .where(OntologyProjectionState.status == ProjectionStatus.DEAD)
        )
        return {"pending": int(pending or 0), "dead": int(dead or 0)}

    def claim_pending_projections(
        self, limit: int | None = None
    ) -> list[OntologyProjectionState]:
        limit = settings.projection_drain_batch_size if limit is None else limit
        stmt = (
            select(OntologyProjectionState)
            .where(OntologyProjectionState.status == ProjectionStatus.PENDING)
            .order_by(
                OntologyProjectionState.last_enqueued_at.asc(),
                OntologyProjectionState.id.asc(),
            )
            .limit(limit)
        )
        try:
            stmt = stmt.with_for_update(skip_locked=True)
        except Exception:
            logger.exception(
                "FOR UPDATE not supported by this dialect; "
                "claiming pending projections without row locks"
            )
        rows = list(self._session.scalars(stmt).all())
        for row in rows:
            row.status = ProjectionStatus.PROJECTING
            row.attempts = (row.attempts or 0) + 1
        self._session.flush()
        return rows

    def mark_projection_done(
        self, row: OntologyProjectionState, projected_version: int
    ) -> bool:
        """Mark a projection row synced — but only if nothing newer landed.

        The UPDATE is conditional on ``canonical_version`` still equalling
        the version that was actually read and projected
        (``projected_version``): a canonical write landing between the
        drain's read and this mark leaves the DB row with a newer
        ``canonical_version``, so the UPDATE matches 0 rows and the row
        is left ``pending`` for the next drain instead of silently
        dropping the newer write. Returns True when the row was marked.
        """
        result = self._session.execute(
            update(OntologyProjectionState)
            .where(
                OntologyProjectionState.id == row.id,
                OntologyProjectionState.canonical_version == projected_version,
            )
            .values(
                projected_version=projected_version,
                status=ProjectionStatus.SYNCED,
                last_projected_at=_utcnow(),
                last_error=None,
            )
        )
        if result.rowcount == 0:
            # A newer canonical write arrived mid-projection; re-queue so
            # the next drain re-projects it instead of stranding the row
            # in "projecting" (claim only picks up "pending" rows).
            row.status = ProjectionStatus.PENDING
            self._session.flush()
            self._session.expire(row)
            return False
        self._session.expire(row)
        return True

    def mark_projection_failed(self, row: OntologyProjectionState, error: str) -> None:
        row.status = ProjectionStatus.PENDING
        row.last_error = (error or "")[:MAX_ERROR_LEN]
        self._session.flush()

    def mark_projection_dead(self, row: OntologyProjectionState, error: str) -> None:
        """Poison-row terminal state: never re-claimed (claim is pending-only).

        The row keeps its last_error so an operator can inspect why it
        died; re-queueing is a manual fix (upsert_projection_pending
        resets status to "pending"), never automatic.
        """
        row.status = ProjectionStatus.DEAD
        row.last_error = (error or "")[:MAX_ERROR_LEN]
        self._session.flush()
