"""Edge-case seed tests — poison projection row + backfill fixture.

No live Neo4j/FalkorDB: the drain runs against the in-memory fake
graph store, and the fixture loader runs against a recording fake.
"""

import json
from pathlib import Path

from app.core.db import SessionLocal
from ontology.models import OntologyProjectionState
from ontology.projection import provider as provider_module
from ontology.projection.engine import drain_pending_projections
from scripts.load_backfill_fixture import load_fixture
from seeds.seed_edge_cases import POISON_SUBJECT_REF, plant_poison_row
from seeds.seed_edge_cases import main as seed_edge_main
from tests.ontology.test_projection_fake_store import FakeGraphStore

FIXTURE = Path(__file__).resolve().parent.parent / "seeds" / "backfill_fixture.json"


def test_plant_poison_row_idempotent(db_session):
    db = SessionLocal()
    try:
        first = plant_poison_row(db)
        assert first == {"planted": True, "status": "pending"}
        second = plant_poison_row(db)
        assert second == {"planted": False, "status": "pending"}
        rows = (
            db.query(OntologyProjectionState)
            .filter_by(subject_ref=POISON_SUBJECT_REF)
            .all()
        )
        assert len(rows) == 1
        row = rows[0]
        assert row.subject_kind == "relationship"
        assert row.canonical_version == 1
        assert row.attempts == 0
    finally:
        db.close()


def test_poison_row_goes_dead_after_max_attempts(db_session, monkeypatch):
    monkeypatch.setattr(provider_module, "_store", FakeGraphStore())
    db = SessionLocal()
    try:
        plant_poison_row(db)
        # One attempt is enough when max_attempts=1: the drain cannot
        # parse the ref, records the failure, and marks the row dead.
        result = drain_pending_projections(db, max_attempts=1)
        assert result["dead"] == 1
        assert result["errors"] == 1
        assert result["processed"] == 0
        row = (
            db.query(OntologyProjectionState)
            .filter_by(subject_ref=POISON_SUBJECT_REF)
            .one()
        )
        assert row.status == "dead"
        assert row.attempts == 1
        assert "rel:seed-edge-poison" in (row.last_error or "")
    finally:
        db.close()


def test_seed_edge_cases_dry_run(capsys, db_session):
    assert seed_edge_main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Dry run OK" in out
    assert "would plant 1 row" in out
    db = SessionLocal()
    try:
        assert db.query(OntologyProjectionState).count() == 0
    finally:
        db.close()


class _RecordingGraph:
    def __init__(self):
        self.queries = []

    def query(self, cypher, params=None):
        self.queries.append((cypher, params or {}))
        return []


def test_backfill_fixture_loads_with_merge(db_session):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["graph"] == "backfill_fixture"
    graph = _RecordingGraph()
    counts = load_fixture(graph, fixture)
    assert counts == {"nodes": 4, "edges": 3}
    node_queries = [q for q, _ in graph.queries if q.startswith("MERGE (n:")]
    edge_queries = [q for q, _ in graph.queries if "MERGE (a)-[r:" in q]
    assert len(node_queries) == 4
    assert len(edge_queries) == 3
    # The legacy asset O19 is NOT in canonical — that is the point of
    # the fixture (the backfill script must create it).
    assert "O19" in [p["id"] for _, p in graph.queries if "id" in p]
    assert any("PART_OF" in q for q in edge_queries)
    assert any("HAS_COMPONENT" in q for q in edge_queries)
    assert any("DOCUMENTED_BY" in q for q in edge_queries)
    # Re-running issues the same MERGE queries — idempotent by construction.
    again = _RecordingGraph()
    assert load_fixture(again, fixture) == counts
    assert again.queries == graph.queries


def test_backfill_fixture_rejects_unknown_types():
    graph = _RecordingGraph()
    try:
        load_fixture(graph, {
            "nodes": [{"labels": ["Gadget"], "properties": {"id": "g1"}}],
            "edges": [],
        })
    except ValueError as exc:
        assert "unknown labels" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown label")
    try:
        load_fixture(graph, {"nodes": [], "edges": [
            {"type": "OWNS", "from": {"labels": ["Asset"], "properties": {"id": "O19"}},
             "to": {"labels": ["Building"], "properties": {"id": "building-B1"}}},
        ]})
    except ValueError as exc:
        assert "unknown type" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown relation type")
