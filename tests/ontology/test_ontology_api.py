"""Ontology HTTP routes (canonical read; graph mocked)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
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


def _set_o21_status(status: str) -> None:
    init_db()
    db = SessionLocal()
    try:
        writer = OntologyWriteService(CanonicalOntologyRepository(db))
        writer.write_entity(
            entity_type="Asset",
            entity_key="O21",
            properties={"name": "demo"},
            provenance=WriteProvenance(trace_id="api-test"),
            status=status,
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


def test_get_entity_deleted_returns_404():
    _seed_o21()
    _set_o21_status("deleted")
    resp = client.get("/ontology/entities/Asset/O21")
    assert resp.status_code == 404


def test_get_entity_archived_returns_404():
    _seed_o21()
    _set_o21_status("archived")
    resp = client.get("/ontology/entities/Asset/O21")
    assert resp.status_code == 404


def test_get_entity_store_unavailable_returns_503(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("postgres down")

    monkeypatch.setattr("app.routers.ontology.read_entity", _boom)
    resp = client.get("/ontology/entities/Asset/O21")
    assert resp.status_code == 503


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


def test_get_relationships_graph_down_returns_503(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr("app.routers.ontology.graph_query.neighbors", _boom)
    resp = client.get("/ontology/entities/Asset/O21/relationships")
    assert resp.status_code == 503


def test_get_knowledge_links_mocked(monkeypatch):
    seen: dict = {}

    def _fake(entity_type, entity_key, *, limit=50, **k):
        seen["limit"] = limit
        return [{"document_id": "DOC-1", "rel_type": "DOCUMENTED_BY"}]

    monkeypatch.setattr(
        "app.routers.ontology.graph_query.linked_documents", _fake
    )
    resp = client.get("/ontology/entities/Asset/O21/knowledge")
    assert resp.status_code == 200
    assert seen["limit"] == 50  # router default
    body = resp.json()
    assert body["entity_key"] == "O21"
    assert body["documents"][0]["document_id"] == "DOC-1"

    resp = client.get("/ontology/entities/Asset/O21/knowledge?limit=5")
    assert resp.status_code == 200
    assert seen["limit"] == 5


def test_get_knowledge_links_graph_down_returns_503(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr("app.routers.ontology.graph_query.linked_documents", _boom)
    resp = client.get("/ontology/entities/Asset/O21/knowledge")
    assert resp.status_code == 503


def test_get_subgraph_mocked(monkeypatch):
    def _fake(root_type, root_key, *, depth=2, limit=100, **k):
        return {
            "root_type": root_type,
            "root_key": root_key,
            "nodes": [{"id": root_key, "labels": [root_type]}],
            "relationships": [],
        }

    monkeypatch.setattr("app.routers.ontology.graph_query.subgraph", _fake)
    resp = client.get("/ontology/graph?root_type=Asset&root_key=O21")
    assert resp.status_code == 200
    body = resp.json()
    assert body["root_key"] == "O21"
    assert body["nodes"][0]["id"] == "O21"


def test_invalid_entity_type_returns_400():
    resp = client.get("/ontology/entities/Widget/W1")
    assert resp.status_code == 400


def test_invalid_entity_key_returns_400():
    # Asset keys must match ^[A-Za-z0-9_-]{1,64}$ — a colon is invalid.
    resp = client.get("/ontology/entities/Asset/O:21")
    assert resp.status_code == 400


def test_limit_and_depth_validation_422(monkeypatch):
    monkeypatch.setattr(
        "app.routers.ontology.graph_query.linked_documents",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "app.routers.ontology.graph_query.subgraph",
        lambda *a, **k: {
            "root_type": "Asset",
            "root_key": "O21",
            "nodes": [],
            "relationships": [],
        },
    )
    assert client.get("/ontology/entities/Asset/O21/knowledge?limit=0").status_code == 422
    assert client.get("/ontology/entities/Asset/O21/knowledge?limit=201").status_code == 422
    assert client.get("/ontology/graph?root_type=Asset&root_key=O21&depth=0").status_code == 422
    assert client.get("/ontology/graph?root_type=Asset&root_key=O21&depth=4").status_code == 422
    assert client.get("/ontology/graph?root_type=Asset&root_key=O21&limit=501").status_code == 422


def test_wrong_api_key_returns_401(monkeypatch):
    # Tests normally run with API_KEY="" (auth disabled); enable it here.
    monkeypatch.setattr(settings, "api_key", "secret")
    resp = client.get("/ontology/entities/Asset/missing-key")
    assert resp.status_code == 401
    resp = client.get(
        "/ontology/entities/Asset/missing-key", headers={"X-API-Key": "wrong"}
    )
    assert resp.status_code == 401
    resp = client.get(
        "/ontology/entities/Asset/missing-key", headers={"X-API-Key": "secret"}
    )
    assert resp.status_code == 404  # auth passed; the entity just does not exist


def test_trigger_projection_drain_mocked(monkeypatch):
    seen: dict = {}

    def _fake(db, *, limit=32, **k):
        seen["limit"] = limit
        return {"processed": 2, "errors": 0, "dead": 0, "claimed": 2}

    monkeypatch.setattr("app.routers.ontology.drain_pending_projections", _fake)
    resp = client.post("/ontology/internal/project-drain")
    assert resp.status_code == 200
    # Same batch size as the beat task (settings.projection_drain_batch_size).
    assert seen["limit"] == settings.projection_drain_batch_size
    body = resp.json()
    assert body["processed"] == 2
    assert body["errors"] == 0
    assert body["claimed"] == 2
