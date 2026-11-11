"""Ontology HTTP routes (canonical read; graph mocked)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.db import SessionLocal, init_db
from app.main import app
from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.write_service import OntologyWriteService

client = TestClient(app)


def _seed_o21() -> None:
    init_db()
    db = SessionLocal()
    try:
        writer = OntologyWriteService(CanonicalOntologyRepository(db))
        writer.write_entity(
            entity_type="Asset",
            entity_key="O21",
            properties={"name": "demo"},
            provenance=WriteProvenance(trace_id="api-test"),
        )
        db.commit()
    finally:
        db.close()


def test_get_entity_not_found():
    resp = client.get("/ontology/entities/Asset/missing-key")
    assert resp.status_code == 404


def test_get_entity_ok(monkeypatch):
    _seed_o21()
    resp = client.get("/ontology/entities/Asset/O21")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entity_key"] == "O21"
    assert body["properties"]["name"] == "demo"


def test_get_relationships_mocked(monkeypatch):
    monkeypatch.setattr(
        "app.routers.ontology.graph_query.neighbors",
        lambda *a, **k: [
            {"from": "O21", "type": "PART_OF", "to": "building-B1", "to_label": "Building"}
        ],
    )
    resp = client.get("/ontology/entities/Asset/O21/relationships")
    assert resp.status_code == 200
    assert resp.json()[0]["rel_type"] == "PART_OF"
