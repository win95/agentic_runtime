"""Backfill L1 -> Neo4j tests (design 2026-11-17, section 8 step 1).

The new L1 model has no ``properties`` column, so the legacy tables are
created here with raw DDL (including ``properties``) and seeded with
sample rows; the backfill itself only ever reads L1 through raw SQL,
never through the ORM models. Neo4j is a fake store that records every
MERGE and answers the count / property lookups ``--verify`` uses.
"""

from __future__ import annotations

import json
import re

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from scripts.backfill_l1_to_neo4j import (
    run_backfill,
    verify_backfill,
)

LEGACY_ENTITIES_DDL = """
CREATE TABLE ontology_entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    properties TEXT,
    state_hash TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    source_event_id TEXT,
    created_at TEXT,
    updated_at TEXT,
    UNIQUE (entity_type, entity_key)
)
"""

LEGACY_RELATIONSHIPS_DDL = """
CREATE TABLE ontology_relationships (
    id TEXT PRIMARY KEY,
    relationship_type TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_key TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_key TEXT NOT NULL,
    properties TEXT,
    state_hash TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    source_event_id TEXT,
    created_at TEXT,
    updated_at TEXT,
    UNIQUE (relationship_type, source_type, source_key, target_type, target_key)
)
"""


class FakeGraphStore:
    """Records MERGE calls and answers --verify lookups from them."""

    def __init__(self) -> None:
        self.nodes: dict[tuple[str, str], dict] = {}
        self.edges: list[dict] = []
        self.merge_node_calls: list[tuple[str, str, dict]] = []
        self.merge_relationship_calls: list[tuple] = []

    def merge_node(self, label, node_id, properties):
        props = dict(properties or {})
        props["id"] = node_id
        self.nodes[(label, node_id)] = props
        self.merge_node_calls.append((label, node_id, dict(properties or {})))

    def merge_relationship(
        self, source_label, source_id, rel_type, target_label, target_id, properties
    ):
        record = {
            "source_label": source_label,
            "source_id": source_id,
            "rel_type": rel_type,
            "target_label": target_label,
            "target_id": target_id,
            "properties": dict(properties or {}),
        }
        self.edges.append(record)
        self.merge_relationship_calls.append(
            (source_label, source_id, rel_type, target_label, target_id, dict(properties or {}))
        )

    def run_query(self, cypher, params=None):
        params = params or {}
        if "count(n)" in cypher:
            total = sum(
                1 for props in self.nodes.values() if props.get("status") == "active"
            )
            return [{"total": total}]
        if "count(r)" in cypher:
            total = sum(
                1 for edge in self.edges if edge["properties"].get("status") == "active"
            )
            return [{"total": total}]
        if "properties(n)" in cypher:
            match = re.search(r"MATCH \(n:(\w+)", cypher)
            label = match.group(1) if match else ""
            props = self.nodes.get((label, params.get("id")))
            if props is None:
                return []
            return [{"props": dict(props)}]
        raise AssertionError(f"unexpected query in fake store: {cypher!r}")

    def close(self):
        pass


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text(LEGACY_ENTITIES_DDL))
        conn.execute(text(LEGACY_RELATIONSHIPS_DDL))
        conn.execute(
            text(
                "INSERT INTO ontology_entities "
                "(id, entity_type, entity_key, properties, state_hash, "
                " version, status, trace_id) "
                "VALUES (:id, :type, :key, :props, 'hash', :version, :status, 't1')"
            ),
            [
                {
                    "id": "e1",
                    "type": "Asset",
                    "key": "room-101",
                    "props": json.dumps({"name": "Room 101", "floor": 1}),
                    "version": 2,
                    "status": "active",
                },
                {
                    "id": "e2",
                    "type": "Building",
                    "key": "hotel-a",
                    "props": json.dumps({"name": "Hotel A"}),
                    "version": 1,
                    "status": "active",
                },
                {
                    "id": "e3",
                    "type": "Component",
                    "key": "ac-01",
                    "props": json.dumps({"name": "AC unit", "temperature": 22.5}),
                    "version": 3,
                    "status": "archived",
                },
            ],
        )
        conn.execute(
            text(
                "INSERT INTO ontology_relationships "
                "(id, relationship_type, source_type, source_key, "
                " target_type, target_key, properties, state_hash, "
                " version, status, trace_id) "
                "VALUES (:id, :rel, :stype, :skey, :ttype, :tkey, "
                " :props, 'hash', :version, :status, 't1')"
            ),
            [
                {
                    "id": "r1",
                    "rel": "PART_OF",
                    "stype": "Asset",
                    "skey": "room-101",
                    "ttype": "Building",
                    "tkey": "hotel-a",
                    "props": json.dumps({"since": "2024-01-01"}),
                    "version": 1,
                    "status": "active",
                },
            ],
        )
    sess = Session(bind=engine)
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


def test_backfill_merges_entities_and_relationships(session):
    store = FakeGraphStore()
    counts = run_backfill(session=session, store=store)

    assert counts == {
        "entities_seen": 3,
        "entities_written": 3,
        "relationships_seen": 1,
        "relationships_written": 1,
    }

    room = store.nodes[("Asset", "room-101")]
    assert room == {
        "id": "room-101",
        "name": "Room 101",
        "floor": 1,
        "status": "active",
        "version": 2,
    }
    hotel = store.nodes[("Building", "hotel-a")]
    assert hotel["name"] == "Hotel A"
    assert hotel["status"] == "active"
    assert hotel["version"] == 1

    # Archived rows are backfilled too, carrying their status, so the
    # drain/reconciler can deal with them afterwards.
    ac = store.nodes[("Component", "ac-01")]
    assert ac["name"] == "AC unit"
    assert ac["temperature"] == 22.5
    assert ac["status"] == "archived"
    assert ac["version"] == 3

    assert len(store.edges) == 1
    edge = store.edges[0]
    assert edge["source_label"] == "Asset"
    assert edge["source_id"] == "room-101"
    assert edge["rel_type"] == "PART_OF"
    assert edge["target_label"] == "Building"
    assert edge["target_id"] == "hotel-a"
    assert edge["properties"] == {
        "since": "2024-01-01",
        "status": "active",
        "version": 1,
    }


def test_backfill_dry_run_writes_nothing(session):
    store = FakeGraphStore()
    counts = run_backfill(session=session, store=store, dry_run=True)

    assert counts["entities_seen"] == 3
    assert counts["entities_written"] == 3  # would-be count
    assert counts["relationships_seen"] == 1
    assert counts["relationships_written"] == 1  # would-be count
    assert store.merge_node_calls == []
    assert store.merge_relationship_calls == []
    assert store.nodes == {}
    assert store.edges == []


def test_backfill_dry_run_needs_no_store(session):
    counts = run_backfill(session=session, store=None, dry_run=True)
    assert counts["entities_written"] == 3
    assert counts["relationships_written"] == 1


def test_backfill_real_run_requires_store(session):
    with pytest.raises(Exception, match="Neo4j store is required"):
        run_backfill(session=session, store=None)


def test_verify_matches_after_backfill(session):
    store = FakeGraphStore()
    run_backfill(session=session, store=store)

    result = verify_backfill(session=session, store=store)

    assert result["l1_active_entities"] == 2
    assert result["neo4j_active_nodes"] == 2
    assert result["l1_active_relationships"] == 1
    assert result["neo4j_active_edges"] == 1
    assert result["l1_total_entities"] == 3
    assert result["match"] is True
    assert len(result["spot_checks"]) == 2  # only active entities sampled
    assert all(check["ok"] for check in result["spot_checks"])
    assert {c["entity_key"] for c in result["spot_checks"]} == {
        "room-101",
        "hotel-a",
    }


def test_verify_detects_missing_node(session):
    store = FakeGraphStore()
    run_backfill(session=session, store=store)
    del store.nodes[("Asset", "room-101")]

    result = verify_backfill(session=session, store=store)

    assert result["match"] is False
    assert result["neo4j_active_nodes"] == 1
    room_check = next(
        c for c in result["spot_checks"] if c["entity_key"] == "room-101"
    )
    assert room_check["ok"] is False
    assert "missing" in room_check["detail"]


def test_verify_detects_property_mismatch(session):
    store = FakeGraphStore()
    run_backfill(session=session, store=store)
    store.nodes[("Building", "hotel-a")]["name"] = "Wrong Name"

    result = verify_backfill(session=session, store=store)

    assert result["match"] is False
    hotel_check = next(
        c for c in result["spot_checks"] if c["entity_key"] == "hotel-a"
    )
    assert hotel_check["ok"] is False
    assert "mismatched" in hotel_check["detail"]
