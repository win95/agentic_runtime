"""Canonical ontology write/read on SQLite."""

from __future__ import annotations

import pytest

from app.core.db import SessionLocal, init_db
from ontology.canonical.entity_read import get_entity
from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.write_service import OntologyWriteService
from ontology.models import OntologyEntity, OntologyProjectionState, OntologyVersion


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


def test_write_key_rejects_colon_and_control_chars(db):
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-bad-key")
    with pytest.raises(ValueError, match="must not contain ':'"):
        writer.write_entity(
            entity_type="Asset",
            entity_key="O:21",
            properties={},
            provenance=prov,
        )
    with pytest.raises(ValueError, match="control characters"):
        writer.write_entity(
            entity_type="Asset",
            entity_key="O21\x01",
            properties={},
            provenance=prov,
        )
    with pytest.raises(ValueError, match="must not contain ':'"):
        writer.write_relationship(
            relationship_type="PART_OF",
            source_type="Asset",
            source_key="O21",
            target_type="Building",
            target_key="B:1",
            properties=None,
            provenance=prov,
        )


def test_write_key_strips_surrounding_whitespace(db):
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-strip-key")
    result = writer.write_entity(
        entity_type="Asset",
        entity_key="  O21-stripped  ",
        properties={"name": "Elevator"},
        provenance=prov,
    )
    assert result.subject_ref == "entity:Asset:O21-stripped"
    assert repo.get_entity("Asset", "O21-stripped") is not None
    assert repo.get_entity("Asset", "  O21-stripped  ") is None


def test_write_rejects_bad_status(db):
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-bad-status")
    with pytest.raises(ValueError, match="must be one of"):
        writer.write_entity(
            entity_type="Asset",
            entity_key="O21-status",
            properties={},
            provenance=prov,
            status="retired",
        )
    with pytest.raises(ValueError, match="must be one of"):
        writer.write_relationship(
            relationship_type="PART_OF",
            source_type="Asset",
            source_key="O21",
            target_type="Building",
            target_key="building-B1",
            properties=None,
            provenance=prov,
            status="hidden",
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


def test_concurrent_duplicate_entity_insert_retries_as_update(db, monkeypatch):
    """Simulated race: get_entity says 'missing' but the INSERT collides."""
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-race-insert")
    writer.write_entity(
        entity_type="Asset",
        entity_key="O21-race",
        properties={"n": 1},
        provenance=prov,
    )
    db.commit()

    orig_get = repo.get_entity
    calls = {"n": 0}

    def _flaky(entity_type, entity_key):
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # stale read: the row is already there
        return orig_get(entity_type, entity_key)

    monkeypatch.setattr(repo, "get_entity", _flaky)
    result = writer.write_entity(
        entity_type="Asset",
        entity_key="O21-race",
        properties={"n": 2},
        provenance=prov,
    )
    db.commit()
    assert result.applied
    assert result.version == 2
    assert result.change_type == "update"
    rows = (
        db.query(OntologyEntity)
        .filter_by(entity_type="Asset", entity_key="O21-race")
        .all()
    )
    assert len(rows) == 1
    assert rows[0].properties == {"n": 2}
    assert rows[0].version == 2


def test_concurrent_duplicate_version_row_retries(db):
    """Simulated race: the version row already exists when we append it."""
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-race-version")
    writer.write_entity(
        entity_type="Asset",
        entity_key="O21-ver-race",
        properties={"n": 1},
        provenance=prov,
    )
    db.commit()
    # A concurrent writer's version row for v2 is already in the table.
    db.add(
        OntologyVersion(
            subject_kind="entity",
            subject_ref="entity:Asset:O21-ver-race",
            version=2,
            change_type="update",
            snapshot={},
            trace_id="race-2",
            source_event_id=None,
        )
    )
    db.commit()
    result = writer.write_entity(
        entity_type="Asset",
        entity_key="O21-ver-race",
        properties={"n": 2},
        provenance=prov,
    )
    db.commit()
    assert result.applied
    assert result.version == 3
    row = repo.get_entity("Asset", "O21-ver-race")
    assert row is not None
    assert row.version == 3
    assert row.properties == {"n": 2}


def test_ensure_entity_preserves_existing_properties(db):
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="test-ensure")
    writer.write_entity(
        entity_type="Asset",
        entity_key="O21-ensure",
        properties={"name": "Elevator"},
        provenance=prov,
    )
    db.commit()
    versions_before = (
        db.query(OntologyVersion)
        .filter_by(subject_ref="entity:Asset:O21-ensure")
        .count()
    )
    result = writer.ensure_entity(
        entity_type="Asset",
        entity_key="O21-ensure",
        properties=None,
        provenance=prov,
    )
    assert not result.applied
    assert result.change_type == "noop"
    row = repo.get_entity("Asset", "O21-ensure")
    assert row is not None and row.properties == {"name": "Elevator"}
    assert (
        db.query(OntologyVersion)
        .filter_by(subject_ref="entity:Asset:O21-ensure")
        .count()
        == versions_before
    )
    created = writer.ensure_entity(
        entity_type="Building",
        entity_key="building-ensure",
        properties=None,
        provenance=prov,
    )
    assert created.applied
    assert repo.get_entity("Building", "building-ensure") is not None


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


def test_seed_rerun_is_noop(monkeypatch):
    """Re-running the demo seeder creates no new versions (S3)."""
    from ontology.projection import provider as provider_module
    from seeds.seed_ontology_demo import O21_PROPERTIES, seed_ontology_demo

    monkeypatch.setattr(provider_module, "_store", _FakeGraphStore())
    seed_ontology_demo()

    db = SessionLocal()
    try:
        versions_after_first = db.query(OntologyVersion).count()
        entity = CanonicalOntologyRepository(db).get_entity("Asset", "O21")
        assert entity is not None
        version_after_first = entity.version
        assert entity.properties == O21_PROPERTIES
    finally:
        db.close()

    seed_ontology_demo()

    db = SessionLocal()
    try:
        assert db.query(OntologyVersion).count() == versions_after_first
        entity = CanonicalOntologyRepository(db).get_entity("Asset", "O21")
        assert entity is not None
        assert entity.version == version_after_first
        assert entity.properties == O21_PROPERTIES
    finally:
        db.close()
