"""Backfill FalkorDB -> canonical: mapping logic with a fake driver.

No live FalkorDB here: the fake graph returns canned ``result_set``
rows shaped like ``falkordb`` Node/Edge objects (duck-typed: labels /
properties / relation / src_node / dest_node).
"""

from __future__ import annotations

import pytest

from app.core.db import SessionLocal, init_db
from ontology.canonical.repository import CanonicalOntologyRepository
from scripts.backfill_falkor_to_canonical import (
    BackfillError,
    connect_falkor,
    main,
    map_edge,
    map_node,
    run_backfill,
)


class FakeNode:
    def __init__(self, labels, properties):
        self.labels = labels
        self.properties = properties


class FakeEdge:
    def __init__(self, src_node, relation, dest_node, properties=None):
        self.src_node = src_node
        self.relation = relation
        self.dest_node = dest_node
        self.properties = properties or {}


class FakeResult:
    def __init__(self, rows):
        self.result_set = rows


class FakeGraph:
    """Returns canned nodes, then canned edges, per query text."""

    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges
        self.queries: list[str] = []

    def query(self, q, params=None, timeout=None):
        self.queries.append(q)
        if q.startswith("MATCH (n)"):
            return FakeResult([[n] for n in self._nodes])
        if q.startswith("MATCH (a)-[r]"):
            return FakeResult([[e.src_node, e, e.dest_node] for e in self._edges])
        raise AssertionError(f"unexpected query: {q!r}")


@pytest.fixture()
def db():
    init_db()
    session = SessionLocal()
    # The in-memory DB is shared across tests in this process; wipe the
    # ontology tables so each backfill test starts from an empty graph.
    from ontology.models import (
        OntologyEntity,
        OntologyProjectionState,
        OntologyRelationship,
        OntologyVersion,
    )

    for model in (
        OntologyProjectionState,
        OntologyVersion,
        OntologyRelationship,
        OntologyEntity,
    ):
        session.query(model).delete()
    session.commit()
    try:
        yield session
    finally:
        session.close()


def _graph():
    asset = FakeNode(["Asset"], {"id": "O21", "name": "lift"})
    building = FakeNode(["Building"], {"id": "building-B1"})
    widget = FakeNode(["Widget"], {"id": "W1"})  # not in ENTITY_TYPES_V1
    nokey = FakeNode(["Asset"], {"name": "no id property"})
    edge = FakeEdge(asset, "PART_OF", building, {"since": "2020"})
    bad_edge = FakeEdge(asset, "NOT_A_REL", building)
    return FakeGraph(
        nodes=[asset, building, widget, nokey],
        edges=[edge, bad_edge],
    )


def test_map_node_ok_and_skips():
    mapped = map_node(FakeNode(["Asset"], {"id": "O21", "name": "lift"}))
    assert mapped == {
        "entity_type": "Asset",
        "entity_key": "O21",
        "properties": {"name": "lift"},  # "id" is the key, not a property
    }
    assert map_node(FakeNode(["Widget"], {"id": "W1"})) is None
    assert map_node(FakeNode(["Asset"], {"name": "no id"})) is None
    assert map_node(FakeNode([], {"id": "X"})) is None


def test_map_edge_ok_and_skips():
    asset = FakeNode(["Asset"], {"id": "O21"})
    building = FakeNode(["Building"], {"id": "building-B1"})
    mapped = map_edge(FakeEdge(asset, "part_of", building, {"since": "2020"}))
    assert mapped is not None
    assert mapped["relationship_type"] == "PART_OF"
    assert mapped["source_key"] == "O21"
    assert mapped["target_key"] == "building-B1"
    assert map_edge(FakeEdge(asset, "NOT_A_REL", building)) is None
    assert map_edge(FakeEdge(asset, "PART_OF", FakeNode(["Widget"], {"id": "W1"}))) is None


def test_run_backfill_writes_entities_then_relationships(db):
    counts = run_backfill(graph=_graph(), session=db)
    assert counts == {
        "nodes_seen": 4,
        "entities_written": 2,
        "entities_skipped": 2,
        "edges_seen": 2,
        "relationships_written": 1,
        "relationships_skipped": 1,
    }
    repo = CanonicalOntologyRepository(db)
    entity = repo.get_entity("Asset", "O21")
    assert entity is not None
    assert entity.properties == {"name": "lift"}
    assert entity.source_event_id == "falkor:Asset:O21"
    assert repo.get_entity("Widget", "W1") is None
    rel = repo.get_relationship("PART_OF", "Asset", "O21", "Building", "building-B1")
    assert rel is not None
    assert rel.source_event_id == "falkor:rel:O21:PART_OF:building-B1"
    assert rel.properties == {"since": "2020"}


def test_run_backfill_is_idempotent(db):
    graph = _graph()
    first = run_backfill(graph=graph, session=db)
    assert first["entities_written"] == 2
    assert first["relationships_written"] == 1
    second = run_backfill(graph=graph, session=db)
    assert second["entities_written"] == 0  # replayed via source_event_id
    assert second["relationships_written"] == 0
    repo = CanonicalOntologyRepository(db)
    assert repo.get_entity("Asset", "O21").version == 1


def test_run_backfill_dry_run_writes_nothing(db):
    counts = run_backfill(graph=_graph(), session=db, dry_run=True)
    assert counts["entities_written"] == 2  # would-be count
    assert counts["relationships_written"] == 1
    repo = CanonicalOntologyRepository(db)
    assert repo.get_entity("Asset", "O21") is None
    assert repo.get_relationship("PART_OF", "Asset", "O21", "Building", "building-B1") is None


def test_main_unreachable_falkor_exits_nonzero_and_writes_nothing(db, monkeypatch):
    def _unreachable(*a, **k):
        raise BackfillError("FalkorDB unreachable at localhost:1")

    monkeypatch.setattr(
        "scripts.backfill_falkor_to_canonical.connect_falkor",
        _unreachable,
    )
    # main() builds its own session; the db fixture session shares the
    # same in-memory DB, so "writes nothing" is observable here.
    assert main(["--falkor-host", "localhost", "--falkor-port", "1"]) == 1
    repo = CanonicalOntologyRepository(db)
    assert repo.get_entity("Asset", "O21") is None


def test_connect_falkor_missing_driver(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "falkordb" or name.startswith("falkordb."):
            raise ImportError("No module named 'falkordb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(BackfillError, match="pip install falkordb==1.2.0"):
        connect_falkor("localhost", 6379, "knowledge_graph")
