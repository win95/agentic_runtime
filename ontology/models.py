"""Canonical ontology ORM — shares app.models.Base for Alembic."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text, UniqueConstraint

from app.models import Base, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_id() -> str:
    return str(uuid.uuid4())


class OntologyEntity(Base):
    __tablename__ = "ontology_entities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_key: Mapped[str] = mapped_column(String(200), nullable=False)
    properties: Mapped[dict | None] = mapped_column(JSON, default=None)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    trace_id: Mapped[str] = mapped_column(String(100), nullable=False)
    source_event_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("entity_type", "entity_key", name="uq_ontology_entities_key"),
        Index("ix_ontology_entities_type", "entity_type"),
    )


class OntologyRelationship(Base):
    __tablename__ = "ontology_relationships"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    relationship_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_key: Mapped[str] = mapped_column(String(200), nullable=False)
    target_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_key: Mapped[str] = mapped_column(String(200), nullable=False)
    properties: Mapped[dict | None] = mapped_column(JSON, default=None)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    trace_id: Mapped[str] = mapped_column(String(100), nullable=False)
    source_event_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "relationship_type",
            "source_type",
            "source_key",
            "target_type",
            "target_key",
            name="uq_ontology_relationships_edge",
        ),
        Index("ix_ontology_relationships_source", "source_type", "source_key"),
    )


class OntologyVersion(Base):
    __tablename__ = "ontology_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    subject_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    subject_ref: Mapped[str] = mapped_column(String(300), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    change_type: Mapped[str] = mapped_column(String(16), nullable=False)
    snapshot: Mapped[dict | None] = mapped_column(JSON, default=None)
    trace_id: Mapped[str] = mapped_column(String(100), nullable=False)
    source_event_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    transaction_time: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        UniqueConstraint(
            "subject_kind",
            "subject_ref",
            "version",
            name="uq_ontology_versions_ref_version",
        ),
    )


class OntologyProjectionState(Base):
    __tablename__ = "ontology_projection_state"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    subject_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    subject_ref: Mapped[str] = mapped_column(String(300), nullable=False)
    canonical_version: Mapped[int] = mapped_column(Integer, nullable=False)
    projected_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    change_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_enqueued_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_projected_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    __table_args__ = (
        UniqueConstraint("subject_kind", "subject_ref", name="uq_ontology_projection_state_ref"),
        Index("ix_ontology_projection_state_status", "status"),
    )
