"""Projection engine with in-memory fake graph store."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.db import SessionLocal, init_db
from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.types import relationship_ref
from ontology.canonical.write_service import OntologyWriteService
from ontology.models import OntologyProjectionState
from ontology.projection import provider as provider_module
from ontology.projection.engine import (
    MAX_PROJECTION_ATTEMPTS,
    ProjectionEngine,
    drain_pending_projections,
)


class FakeGraphStore:
    def __init__(self, *, healthy: bool = True) -> None:
        self.nodes: dict[tuple[str, str], dict] = {}
        # Edges keyed by (source_label, source_id, rel_type, target_label,
        # target_id) so a re-drain overwrites instead of duplicating.
        self.edges: dict[tuple, dict | None] = {}
        self.deleted_nodes: list[tuple[str, str]] = []
        self.deleted_relationships: list[tuple] = []
        self.healthy = healthy

    def is_healthy(self) -> bool:
        return self.healthy

    def run_query(self, cypher: str, params: dict[str, Any] | None = None) -> list:
        return []

    def merge_node(self, label: str, node_id: str, properties: dict[str, Any]) -> None:
        self.nodes[(label, node_id)] = dict(properties)

    def node_exists(self, label: str, node_id: str) -> bool:
        return (label, node_id) in self.nodes

    def delete_node(self, label: str, node_id: str) -> None:
        self.nodes.pop((label, node_id), None)
        self.deleted_nodes.append((label, node_id))

    def merge_relationship(
        self,
        source_label: str,
        source_id: str,
        rel_type: str,
        target_label: str,
        target_id: str,
        properties: dict[str, Any] | None,
    ) -> None:
        key = (source_label, source_id, rel_type, target_label, target_id)
        self.edges[key] = dict(properties) if properties is not None else None

    def delete_relationship(
        self,
        source_label: str,
        source_id: str,
        rel_type: str,
        target_label: str,
        target_id: str,
    ) -> None:
        key = (source_label, source_id, rel_type, target_label, target_id)
        self.edges.pop(key, None)
        self.deleted_relationships.append(key)

    def close(self) -> None:
        return None


class FailingGraphStore(FakeGraphStore):
    """Every merge fails — for the poison-row test."""

    def merge_node(self, label: str, node_id: str, properties: dict[str, Any]) -> None:
        raise RuntimeError("boom")


class _RecordingSession:
    """Minimal neo4j session stand-in: records Cypher, returns canned rows."""

    def __init__(self, calls: list, rows: list[dict]) -> None:
        self._calls = calls
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher: str, params: dict[str, Any] | None = None):
        self._calls.append((cypher, params))
        return list(self._rows)


class _RecordingDriver:
    """Minimal neo4j Driver stand-in for Neo4jGraphStore tests."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.calls: list[tuple[str, dict | None]] = []
        self.rows = rows or []

    def session(self, database=None):
        return _RecordingSession(self.calls, self.rows)


@pytest.fixture()
def db():
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _writer(db):
    return OntologyWriteService(CanonicalOntologyRepository(db))


def _row_status(db, subject_ref: str) -> str | None:
    db.expire_all()
    row = db.query(OntologyProjectionState).filter_by(subject_ref=subject_ref).first()
    return row.status if row is not None else None


def test_projection_drain_merges_nodes(db, monkeypatch):
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)
    repo = CanonicalOntologyRepository(db)
    writer = OntologyWriteService(repo)
    prov = WriteProvenance(trace_id="proj-test")
    writer.write_entity(
        entity_type="Asset",
        entity_key="O99-proj",
        properties={"name": "lift"},
        provenance=prov,
    )
    db.commit()
    result = drain_pending_projections(db)
    assert result["processed"] >= 1
    assert ("Asset", "O99-proj") in fake.nodes


def test_projection_drain_merges_relationship_with_provenance_props(db, monkeypatch):
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)
    writer = _writer(db)
    prov = WriteProvenance(trace_id="proj-rel")
    writer.write_entity(
        entity_type="Asset",
        entity_key="O-rel-a",
        properties={"name": "lift"},
        provenance=prov,
    )
    writer.write_entity(
        entity_type="Building",
        entity_key="B-rel-1",
        properties={"name": "tower"},
        provenance=prov,
    )
    writer.write_relationship(
        relationship_type="PART_OF",
        source_type="Asset",
        source_key="O-rel-a",
        target_type="Building",
        target_key="B-rel-1",
        properties={"since": "2020"},
        provenance=prov,
    )
    db.commit()
    result = drain_pending_projections(db)
    assert result["processed"] == 3
    assert result["errors"] == 0
    edge = ("Asset", "O-rel-a", "PART_OF", "Building", "B-rel-1")
    assert edge in fake.edges
    # S4: projected props carry provenance + canonical version.
    assert fake.edges[edge]["status"] == "active"
    assert fake.edges[edge]["trace_id"] == "proj-rel"
    assert fake.edges[edge]["canonical_version"] == 1
    node_props = fake.nodes[("Asset", "O-rel-a")]
    assert node_props["status"] == "active"
    assert node_props["trace_id"] == "proj-rel"
    assert node_props["canonical_version"] == 1


def test_projection_deletes_missing_entity_row(db, monkeypatch):
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)
    repo = CanonicalOntologyRepository(db)
    repo.upsert_projection_pending(
        subject_kind="entity",
        subject_ref="entity:Asset:O-ghost",
        canonical_version=1,
        change_type="update",
        trace_id="ghost-test",
    )
    db.commit()
    result = drain_pending_projections(db)
    assert result["processed"] == 1
    assert ("Asset", "O-ghost") in fake.deleted_nodes
    assert _row_status(db, "entity:Asset:O-ghost") == "synced"


def test_projection_deletes_soft_deleted_entity(db, monkeypatch):
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)
    writer = _writer(db)
    prov = WriteProvenance(trace_id="proj-del")
    writer.write_entity(
        entity_type="Asset",
        entity_key="O-del-1",
        properties={"name": "lift"},
        provenance=prov,
    )
    db.commit()
    drain_pending_projections(db)
    assert ("Asset", "O-del-1") in fake.nodes
    writer.write_entity(
        entity_type="Asset",
        entity_key="O-del-1",
        properties={"name": "lift"},
        provenance=prov,
        status="deleted",
    )
    db.commit()
    drain_pending_projections(db)
    assert ("Asset", "O-del-1") not in fake.nodes
    assert ("Asset", "O-del-1") in fake.deleted_nodes


def test_projection_deletes_missing_relationship_row(db, monkeypatch):
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)
    repo = CanonicalOntologyRepository(db)
    ref = relationship_ref("PART_OF", "Asset", "O-ghost-rel", "Building", "B-ghost")
    repo.upsert_projection_pending(
        subject_kind="relationship",
        subject_ref=ref,
        canonical_version=1,
        change_type="update",
        trace_id="ghost-rel-test",
    )
    db.commit()
    result = drain_pending_projections(db)
    assert result["processed"] == 1
    edge = ("Asset", "O-ghost-rel", "PART_OF", "Building", "B-ghost")
    assert edge in fake.deleted_relationships
    assert _row_status(db, ref) == "synced"


def test_projection_redrain_is_idempotent(db, monkeypatch):
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)
    writer = _writer(db)
    prov = WriteProvenance(trace_id="proj-idem")
    writer.write_entity(
        entity_type="Asset",
        entity_key="O-idem",
        properties={"name": "lift"},
        provenance=prov,
    )
    writer.write_entity(
        entity_type="Building",
        entity_key="B-idem",
        properties={"name": "tower"},
        provenance=prov,
    )
    writer.write_relationship(
        relationship_type="PART_OF",
        source_type="Asset",
        source_key="O-idem",
        target_type="Building",
        target_key="B-idem",
        properties={},
        provenance=prov,
    )
    db.commit()
    first = drain_pending_projections(db)
    assert first["processed"] == 3
    assert len(fake.edges) == 1
    second = drain_pending_projections(db)
    assert second["processed"] == 0
    assert len(fake.edges) == 1
    assert ("Asset", "O-idem") in fake.nodes
    assert ("Building", "B-idem") in fake.nodes


def test_projection_unhealthy_store_short_circuits(db, monkeypatch):
    fake = FakeGraphStore(healthy=False)
    monkeypatch.setattr(provider_module, "_store", fake)
    writer = _writer(db)
    writer.write_entity(
        entity_type="Asset",
        entity_key="O-unhealthy",
        properties={"name": "lift"},
        provenance=WriteProvenance(trace_id="proj-unhealthy"),
    )
    db.commit()
    result = drain_pending_projections(db)
    assert result == {
        "processed": 0,
        "errors": 1,
        "dead": 0,
        "claimed": 0,
        "detail": "neo4j_unhealthy",
    }
    # Row was never claimed — still pending.
    assert _row_status(db, "entity:Asset:O-unhealthy") == "pending"
    # Once healthy again the same row drains normally.
    fake.healthy = True
    result = drain_pending_projections(db)
    assert result["processed"] == 1
    assert ("Asset", "O-unhealthy") in fake.nodes


def test_projection_store_construction_failure_is_unhealthy(db, monkeypatch):
    def _boom():
        raise RuntimeError("Neo4j driver unavailable")

    monkeypatch.setattr(provider_module, "get_graph_store", _boom)
    result = drain_pending_projections(db)
    assert result["processed"] == 0
    assert result["errors"] == 1
    assert result["detail"] == "neo4j_unhealthy"


def test_poison_row_goes_dead_after_max_attempts(db):
    from app.config import settings

    assert MAX_PROJECTION_ATTEMPTS == 5
    assert settings.projection_max_attempts == 5
    fake = FailingGraphStore()
    engine = ProjectionEngine(db, store=fake, max_attempts=2)
    writer = _writer(db)
    writer.write_entity(
        entity_type="Asset",
        entity_key="O-poison",
        properties={"name": "lift"},
        provenance=WriteProvenance(trace_id="proj-poison"),
    )
    db.commit()
    first = engine.drain_batch()
    assert first["errors"] == 1
    assert first["dead"] == 0
    assert _row_status(db, "entity:Asset:O-poison") == "pending"
    second = engine.drain_batch()
    assert second["errors"] == 1
    assert second["dead"] == 1
    assert _row_status(db, "entity:Asset:O-poison") == "dead"
    # Dead rows are never re-claimed.
    third = engine.drain_batch()
    assert third["processed"] == 0
    assert third["claimed"] == 0
    assert third["dead"] == 0


def test_requeue_dead_row_resets_attempts(db):
    repo = CanonicalOntologyRepository(db)
    repo.upsert_projection_pending(
        subject_kind="entity",
        subject_ref="entity:Asset:O-dead",
        canonical_version=1,
        change_type="create",
        trace_id="dead-test",
    )
    db.commit()
    row = (
        db.query(OntologyProjectionState)
        .filter_by(subject_ref="entity:Asset:O-dead")
        .first()
    )
    row.status = "dead"
    row.attempts = 5
    db.commit()
    repo.upsert_projection_pending(
        subject_kind="entity",
        subject_ref="entity:Asset:O-dead",
        canonical_version=2,
        change_type="update",
        trace_id="dead-test-requeue",
    )
    db.commit()
    db.expire_all()
    row = (
        db.query(OntologyProjectionState)
        .filter_by(subject_ref="entity:Asset:O-dead")
        .first()
    )
    assert row.status == "pending"
    assert row.attempts == 0
    # Leave no pending row behind: test_projection_batch_limit counts
    # claimed rows in this shared in-memory DB.
    db.query(OntologyProjectionState).filter_by(
        subject_ref="entity:Asset:O-dead"
    ).delete()
    db.commit()


def test_requeue_pending_row_keeps_attempts(db):
    repo = CanonicalOntologyRepository(db)
    repo.upsert_projection_pending(
        subject_kind="entity",
        subject_ref="entity:Asset:O-pending",
        canonical_version=1,
        change_type="create",
        trace_id="pending-test",
    )
    db.commit()
    row = (
        db.query(OntologyProjectionState)
        .filter_by(subject_ref="entity:Asset:O-pending")
        .first()
    )
    row.attempts = 3
    db.commit()
    repo.upsert_projection_pending(
        subject_kind="entity",
        subject_ref="entity:Asset:O-pending",
        canonical_version=2,
        change_type="update",
        trace_id="pending-test-requeue",
    )
    db.commit()
    db.expire_all()
    row = (
        db.query(OntologyProjectionState)
        .filter_by(subject_ref="entity:Asset:O-pending")
        .first()
    )
    assert row.status == "pending"
    assert row.attempts == 3
    # Leave no pending row behind: test_projection_batch_limit counts
    # claimed rows in this shared in-memory DB.
    db.query(OntologyProjectionState).filter_by(
        subject_ref="entity:Asset:O-pending"
    ).delete()
    db.commit()


def test_projection_batch_limit(db, monkeypatch):
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)
    writer = _writer(db)
    prov = WriteProvenance(trace_id="proj-limit")
    for key in ("O-lim-1", "O-lim-2", "O-lim-3"):
        writer.write_entity(
            entity_type="Asset",
            entity_key=key,
            properties={"name": key},
            provenance=prov,
        )
    db.commit()
    first = drain_pending_projections(db, limit=2)
    assert first["claimed"] == 2
    assert first["processed"] == 2
    second = drain_pending_projections(db, limit=2)
    assert second["claimed"] == 1
    assert second["processed"] == 1


def test_projection_relationship_retries_when_endpoint_unprojected(db, monkeypatch):
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)
    writer = _writer(db)
    prov = WriteProvenance(trace_id="proj-retry")
    writer.write_entity(
        entity_type="Asset",
        entity_key="O-retry",
        properties={"name": "lift"},
        provenance=prov,
    )
    writer.write_entity(
        entity_type="Building",
        entity_key="B-retry",
        properties={"name": "tower"},
        provenance=prov,
    )
    writer.write_relationship(
        relationship_type="PART_OF",
        source_type="Asset",
        source_key="O-retry",
        target_type="Building",
        target_key="B-retry",
        properties={},
        provenance=prov,
    )
    db.commit()
    drain_pending_projections(db)
    edge = ("Asset", "O-retry", "PART_OF", "Building", "B-retry")
    assert edge in fake.edges
    # Simulate the endpoint not being in the graph yet (and the edge
    # never projected), with the edge re-enqueued (e.g. a re-projection
    # trigger).
    fake.nodes.pop(("Building", "B-retry"))
    fake.edges.pop(edge)
    repo = CanonicalOntologyRepository(db)
    rel = repo.get_relationship("PART_OF", "Asset", "O-retry", "Building", "B-retry")
    ref = relationship_ref("PART_OF", "Asset", "O-retry", "Building", "B-retry")
    repo.upsert_projection_pending(
        subject_kind="relationship",
        subject_ref=ref,
        canonical_version=rel.version,
        change_type="update",
        trace_id="proj-retry",
    )
    db.commit()
    result = drain_pending_projections(db)
    assert result["errors"] == 1
    assert edge not in fake.edges
    # Endpoint is active in canonical, so the row retries, not dies.
    assert _row_status(db, ref) == "pending"
    # Once the endpoint is projected, the edge drains.
    fake.nodes[("Building", "B-retry")] = {"id": "B-retry"}
    result = drain_pending_projections(db)
    assert result["processed"] == 1
    assert edge in fake.edges


def test_projection_relationship_dropped_when_endpoint_gone(db, monkeypatch):
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)
    writer = _writer(db)
    prov = WriteProvenance(trace_id="proj-gone")
    writer.write_entity(
        entity_type="Asset",
        entity_key="O-gone",
        properties={"name": "lift"},
        provenance=prov,
    )
    writer.write_entity(
        entity_type="Building",
        entity_key="B-gone",
        properties={"name": "tower"},
        provenance=prov,
    )
    writer.write_relationship(
        relationship_type="PART_OF",
        source_type="Asset",
        source_key="O-gone",
        target_type="Building",
        target_key="B-gone",
        properties={},
        provenance=prov,
    )
    db.commit()
    drain_pending_projections(db)
    edge = ("Asset", "O-gone", "PART_OF", "Building", "B-gone")
    assert edge in fake.edges
    # Endpoint deleted in canonical; re-enqueue the edge row.
    writer.write_entity(
        entity_type="Building",
        entity_key="B-gone",
        properties={"name": "tower"},
        provenance=prov,
        status="deleted",
    )
    repo = CanonicalOntologyRepository(db)
    rel = repo.get_relationship("PART_OF", "Asset", "O-gone", "Building", "B-gone")
    ref = relationship_ref("PART_OF", "Asset", "O-gone", "Building", "B-gone")
    repo.upsert_projection_pending(
        subject_kind="relationship",
        subject_ref=ref,
        canonical_version=rel.version,
        change_type="update",
        trace_id="proj-gone",
    )
    db.commit()
    result = drain_pending_projections(db)
    assert result["errors"] == 0
    assert edge not in fake.edges
    assert edge in fake.deleted_relationships
    assert _row_status(db, ref) == "synced"


def test_projection_relationship_dropped_when_endpoint_never_existed(db, monkeypatch):
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)
    writer = _writer(db)
    prov = WriteProvenance(trace_id="proj-never")
    writer.write_entity(
        entity_type="Building",
        entity_key="B-never",
        properties={"name": "tower"},
        provenance=prov,
    )
    # Relationship references an Asset that was never written.
    writer.write_relationship(
        relationship_type="PART_OF",
        source_type="Asset",
        source_key="O-never",
        target_type="Building",
        target_key="B-never",
        properties={},
        provenance=prov,
    )
    db.commit()
    result = drain_pending_projections(db)
    assert result["errors"] == 0
    edge = ("Asset", "O-never", "PART_OF", "Building", "B-never")
    assert edge not in fake.edges
    assert edge in fake.deleted_relationships
    # No stub node was created for the missing endpoint.
    assert ("Asset", "O-never") not in fake.nodes
    ref = relationship_ref("PART_OF", "Asset", "O-never", "Building", "B-never")
    assert _row_status(db, ref) == "synced"


def test_neo4j_store_ensure_constraints_issues_uniqueness_cypher():
    from ontology.canonical.schema import ENTITY_TYPES_V1
    from ontology.graph.neo4j_store import Neo4jGraphStore

    driver = _RecordingDriver()
    Neo4jGraphStore(driver=driver)
    constraint_calls = [
        cypher for cypher, _ in driver.calls if cypher.startswith("CREATE CONSTRAINT")
    ]
    assert len(constraint_calls) == len(ENTITY_TYPES_V1)
    for label in sorted(ENTITY_TYPES_V1):
        assert any(
            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE" in cypher for cypher in constraint_calls
        )


def test_neo4j_store_merge_uses_full_replace_and_match_endpoints():
    from ontology.graph.neo4j_store import Neo4jGraphStore

    driver = _RecordingDriver()
    store = Neo4jGraphStore(driver=driver)
    driver.calls.clear()
    store.merge_node("Asset", "O1", {"name": "x"})
    cypher, params = driver.calls[-1]
    assert "SET n = $props" in cypher
    assert "n.updated_at = timestamp()" in cypher
    assert params["props"]["id"] == "O1"
    driver.calls.clear()
    store.merge_relationship("Asset", "O1", "PART_OF", "Building", "B1", {"since": "2020"})
    cypher, _ = driver.calls[-1]
    assert "MATCH (a:Asset {id: $source_id})" in cypher
    assert "MATCH (b:Building {id: $target_id})" in cypher
    assert "MERGE (a)-[r:PART_OF]->(b)" in cypher
    assert "SET r = $props" in cypher
    assert "MERGE (a:Asset" not in cypher
    assert "MERGE (b:Building" not in cypher


def test_neo4j_store_node_exists():
    from ontology.graph.neo4j_store import Neo4jGraphStore

    assert (
        Neo4jGraphStore(driver=_RecordingDriver(rows=[{"id": "O1"}])).node_exists("Asset", "O1")
        is True
    )
    assert Neo4jGraphStore(driver=_RecordingDriver(rows=[])).node_exists("Asset", "O1") is False


def test_build_graph_store_rejects_unknown_backend(monkeypatch):
    from app.config import settings
    from ontology.graph.factory import build_graph_store

    monkeypatch.setattr(settings, "graph_database_backend", "falkordb")
    with pytest.raises(ValueError, match="allowed values"):
        build_graph_store()
