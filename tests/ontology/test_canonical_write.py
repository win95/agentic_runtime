"""Canonical ontology write/read on SQLite."""

from __future__ import annotations

import pytest

from app.core.db import SessionLocal, init_db
from ontology.canonical.entity_read import get_entity
from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.write_service import OntologyWriteService


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
