"""Projection engine with in-memory fake graph store."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.db import SessionLocal, init_db
from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.write_service import OntologyWriteService
from ontology.projection import provider as provider_module
from ontology.projection.engine import drain_pending_projections


class FakeGraphStore:
    def __init__(self) -> None:
        self.nodes: dict[tuple[str, str], dict] = {}
        self.edges: list[tuple] = []

    def is_healthy(self) -> bool:
        return True

    def run_query(self, cypher: str, params: dict[str, Any] | None = None) -> list:
        return []

    def merge_node(self, label: str, node_id: str, properties: dict[str, Any]) -> None:
        self.nodes[(label, node_id)] = properties

    def delete_node(self, label: str, node_id: str) -> None:
        self.nodes.pop((label, node_id), None)

    def merge_relationship(
        self,
        source_label: str,
        source_id: str,
        rel_type: str,
        target_label: str,
        target_id: str,
        properties: dict[str, Any] | None,
    ) -> None:
        self.edges.append((source_label, source_id, rel_type, target_label, target_id))

    def delete_relationship(
        self,
        source_label: str,
        source_id: str,
        rel_type: str,
        target_label: str,
        target_id: str,
    ) -> None:
        pass

    def close(self) -> None:
        return None


@pytest.fixture()
def db():
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


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
