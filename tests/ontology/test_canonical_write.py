"""Canonical ontology write/read on SQLite."""

from __future__ import annotations

import pytest

from app.core.db import SessionLocal, init_db
from ontology.canonical.entity_read import get_entity
from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.write_service import OntologyWriteService
from ontology.models import OntologyProjectionState, OntologyVersion


@pytest.fixture()
def db():
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_write_entity_idempotent(db):
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-trace", source_event_id="evt-1")
    r1 = writer.write_entity(
        entity_type="Asset",
        entity_key="O21",
        properties={"name": "Elevator"},
        provenance=prov,
    )
    db.commit()
    r2 = writer.write_entity(
        entity_type="Asset",
        entity_key="O21",
        properties={"name": "Elevator"},
        provenance=prov,
    )
    assert r1.applied
    assert not r2.applied
    row = get_entity(db, "Asset", "O21")
    assert row is not None
    assert row["properties"]["name"] == "Elevator"


def test_write_relationship(db):
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-rel")
    writer.write_entity(
        entity_type="Asset", entity_key="O21", properties={}, provenance=prov
    )
    writer.write_entity(
        entity_type="Building",
        entity_key="building-B1",
        properties={},
        provenance=prov,
    )
    writer.write_relationship(
        relationship_type="PART_OF",
        source_type="Asset",
        source_key="O21",
        target_type="Building",
        target_key="building-B1",
        properties=None,
        provenance=prov,
    )
    db.commit()
    rel = repo.get_relationship("PART_OF", "Asset", "O21", "Building", "building-B1")
    assert rel is not None
    assert rel.status == "active"


def _pending_row(db, subject_ref: str) -> OntologyProjectionState | None:
    return (
        db.query(OntologyProjectionState)
        .filter_by(subject_ref=subject_ref)
        .first()
    )


def test_write_entity_noop_without_event_id(db):
    """Same state, no source_event_id: second write is a noop, not a replay."""
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-noop")
    r1 = writer.write_entity(
        entity_type="Asset",
        entity_key="O21-noop",
        properties={"name": "Elevator"},
        provenance=prov,
    )
    db.commit()
    versions_before = (
        db.query(OntologyVersion)
        .filter_by(subject_ref="entity:Asset:O21-noop")
        .count()
    )
    r2 = writer.write_entity(
        entity_type="Asset",
        entity_key="O21-noop",
        properties={"name": "Elevator"},
        provenance=prov,
    )
    assert r1.applied
    assert not r2.applied
    assert r2.change_type == "noop"
    assert not r2.replayed
    assert r2.version == r1.version
    assert (
        db.query(OntologyVersion)
        .filter_by(subject_ref="entity:Asset:O21-noop")
        .count()
        == versions_before
    )


def test_pending_row_enqueued_on_write(db):
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-pending")
    result = writer.write_entity(
        entity_type="Asset",
        entity_key="O21-pending",
        properties={"name": "Elevator"},
        provenance=prov,
    )
    row = _pending_row(db, "entity:Asset:O21-pending")
    assert row is not None
    assert row.status == "pending"
    assert row.canonical_version == result.version
    assert row.subject_kind == "entity"


def test_write_rejects_bad_entity_type(db):
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-bad-type")
    with pytest.raises(ValueError, match="not allowed in v1"):
        writer.write_entity(
            entity_type="Spaceship",
            entity_key="S1",
            properties={},
            provenance=prov,
        )
    with pytest.raises(ValueError, match="not allowed in v1"):
        writer.write_relationship(
            relationship_type="PART_OF",
            source_type="Asset",
            source_key="O21",
            target_type="Spaceship",
            target_key="S1",
            properties=None,
            provenance=prov,
        )


def test_status_transitions_active_archived_deleted(db):
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-status-flow")
    r1 = writer.write_entity(
        entity_type="Asset",
        entity_key="O21-status-flow",
        properties={"name": "Elevator"},
        provenance=prov,
    )
    assert r1.change_type == "create"
    r2 = writer.write_entity(
        entity_type="Asset",
        entity_key="O21-status-flow",
        properties={"name": "Elevator"},
        provenance=prov,
        status="archived",
    )
    assert r2.applied and r2.change_type == "archived"
    assert r2.version == r1.version + 1
    r3 = writer.write_entity(
        entity_type="Asset",
        entity_key="O21-status-flow",
        properties={"name": "Elevator"},
        provenance=prov,
        status="deleted",
    )
    assert r3.applied and r3.change_type == "deleted"
    assert r3.version == r2.version + 1
    row = repo.get_entity("Asset", "O21-status-flow")
    assert row is not None and row.status == "deleted"


def test_mark_projection_done_is_conditional(db):
    """A write landing between claim and mark must not be dropped."""
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-mark")
    writer.write_entity(
        entity_type="Asset",
        entity_key="O21-mark",
        properties={"n": 1},
        provenance=prov,
    )
    db.commit()
    claimed = repo.claim_pending_projections(limit=32)
    row = next(
        r for r in claimed if r.subject_ref == "entity:Asset:O21-mark"
    )
    # Concurrent write lands after the claim, before the mark.
    writer.write_entity(
        entity_type="Asset",
        entity_key="O21-mark",
        properties={"n": 2},
        provenance=prov,
    )
    assert repo.mark_projection_done(row, 1) is False
    db.commit()
    pending = _pending_row(db, "entity:Asset:O21-mark")
    assert pending is not None
    assert pending.status == "pending"
    assert pending.canonical_version == 2
    assert pending.projected_version is None
    # Next drain re-projects the newer version and the mark succeeds.
    claimed = repo.claim_pending_projections(limit=32)
    row = next(
        r for r in claimed if r.subject_ref == "entity:Asset:O21-mark"
    )
    assert repo.mark_projection_done(row, 2) is True
    db.commit()
    synced = _pending_row(db, "entity:Asset:O21-mark")
    assert synced is not None
    assert synced.status == "synced"
    assert synced.projected_version == 2


class _FakeGraphStore:
    """Minimal graph store for the seed re-run test (no Neo4j)."""

    def __init__(self) -> None:
        self.nodes: dict[tuple[str, str], dict] = {}
        self.edges: list[tuple] = []

    def is_healthy(self) -> bool:
        return True

    def merge_node(self, label, node_id, properties) -> None:
        self.nodes[(label, node_id)] = properties

    def node_exists(self, label, node_id) -> bool:
        return (label, node_id) in self.nodes

    def delete_node(self, label, node_id) -> None:
        self.nodes.pop((label, node_id), None)

    def merge_relationship(
        self, source_label, source_id, rel_type, target_label, target_id, properties
    ) -> None:
        self.edges.append((source_label, source_id, rel_type, target_label, target_id))

    def delete_relationship(
        self, source_label, source_id, rel_type, target_label, target_id
    ) -> None:
        pass

    def close(self) -> None:
        return None


