"""B5 query API (Cypher read-only, search, entities-by-class,
traverse, schema/stats, reasoning) — TestClient + fake store.

The fake below extends ``MetadataFakeGraphStore`` with a small
interpreter for the Cypher the new ``graph_query`` helpers emit
(neighbours incl. multi-hop BFS, per-label instance lists, id/name
CONTAINS search, derived-edge rows, count queries). Anything it
does not recognise returns ``query_rows`` — the canned rows a
``POST /ontology/query`` call sees.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from ontology.knowledge import graph_query
from ontology.metadata import store as metadata_store
from ontology.projection import provider as provider_module
from ontology.reasoning import subclass_descendants
from tests.ontology.test_projection_fake_store import _RecordingDriver
from tests.test_ontology_metadata import MetadataFakeGraphStore

client = TestClient(app)


class QueryFakeGraphStore(MetadataFakeGraphStore):
    """Metadata fake + in-memory instance graph for B5 queries."""

    def __init__(self, *, healthy: bool = True) -> None:
        super().__init__(healthy=healthy)
        self.query_rows: list[dict[str, Any]] = [{"id": "O21", "name": "lift"}]
        self.constraint_rows: list[dict[str, Any]] | None = None
        self.last_timeout: Any = "unset"

    # -- helpers ------------------------------------------------------
    def _adjacent(self, label: str, key: str) -> list[dict[str, Any]]:
        """1-hop neighbour rows for one node (either edge direction)."""
        rows: list[dict[str, Any]] = []
        for (s_label, s_key, rel, t_label, t_key) in sorted(self.edges):
            if (s_label, s_key) == (label, key):
                rows.append(
                    {
                        "from_id": key,
                        "rel_type": rel,
                        "to_id": t_key,
                        "to_label": t_label,
                    }
                )
            elif (t_label, t_key) == (label, key):
                rows.append(
                    {
                        "from_id": key,
                        "rel_type": rel,
                        "to_id": s_key,
                        "to_label": s_label,
                    }
                )
        return rows

    def _bfs_neighbours(
        self, label: str, key: str, depth: int, allowed: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """Multi-hop rows: one row per (reached node, first-hop type)
        — mirroring the real ``neighbors`` Cypher, which RETURNs
        DISTINCT (from, type(r[0]), to, label), so the same node
        reached via two different first hops yields two rows. ``seen``
        is keyed by (node, first-hop type) for the same reason. When
        ``allowed`` is given, only hops of those types are walked
        (the traverse-with-rel_types Cypher)."""
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = {(key, "")}
        frontier: list[tuple[str, str, str]] = [(label, key, "")]
        for _ in range(depth):
            nxt: list[tuple[str, str, str]] = []
            for node_label, node_key, first_rel in frontier:
                for row in self._adjacent(node_label, node_key):
                    if allowed is not None and row["rel_type"] not in allowed:
                        continue
                    to_id = row["to_id"]
                    rel = first_rel or row["rel_type"]
                    if (to_id, rel) in seen:
                        continue
                    seen.add((to_id, rel))
                    rows.append(
                        {
                            "from_id": key,
                            "rel_type": rel,
                            "to_id": to_id,
                            "to_label": row["to_label"],
                        }
                    )
                    nxt.append((row["to_label"], to_id, rel))
            frontier = nxt
        return rows

    # -- run_query interpreter ----------------------------------------
    def run_query(
        self, cypher: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> list:
        self.last_timeout = kwargs.get("timeout", "unset")
        self.queries.append((cypher, params))
        params = params or {}

        if cypher.strip() == "SHOW CONSTRAINTS":
            if self.constraint_rows is not None:
                return list(self.constraint_rows)
            return []

        if "derived: true" in cypher and "count(r)" in cypher:
            return [
                {
                    "n": sum(
                        1
                        for props in self.edges.values()
                        if isinstance(props, dict) and props.get("derived") is True
                    )
                }
            ]

        if "derived: true" in cypher and "source_key" in cypher:
            rows = []
            for (s_label, s_key, rel, t_label, t_key), props in sorted(
                self.edges.items()
            ):
                if not (isinstance(props, dict) and props.get("derived") is True):
                    continue
                if params.get("entity_key") is not None:
                    if s_key != params["entity_key"] and t_key != params["entity_key"]:
                        continue
                    if params.get("label") is not None:
                        if s_key == params["entity_key"] and s_label != params["label"]:
                            continue
                        if t_key == params["entity_key"] and t_label != params["label"]:
                            continue
                rows.append(
                    {
                        "source_key": s_key,
                        "source_type": s_label,
                        "rel_type": rel,
                        "target_key": t_key,
                        "target_type": t_label,
                        "rule": props.get("rule"),
                        "base_ref": props.get("base_ref"),
                    }
                )
            return rows[: params.get("limit", 100)]

        if "<-[:SUBCLASS_OF]-(child" in cypher:
            # subclass_descendants: children of params["class_id"].
            return [
                {"id": child}
                for child, parent in sorted(self.subclasses)
                if parent == params.get("class_id")
            ]

        if "MATCH (n:" in cypher and "properties(n) AS properties" in cypher:
            match = re.search(r"MATCH \(n:([A-Za-z][A-Za-z0-9_]*)\)", cypher)
            label = match.group(1) if match else ""
            rows = [
                {
                    "id": key,
                    "label": node_label,
                    "properties": dict(props),
                }
                for (node_label, key), props in sorted(self.nodes.items())
                if node_label == label
            ]
            return rows[: params.get("limit", 100)]

        if "MATCH (n) WHERE" in cypher and "CONTAINS" in cypher:
            needle = str(params.get("query") or "").lower()
            rows = []
            for (label, key), props in sorted(self.nodes.items()):
                if label in ("OntologyClass", "ObjectProperty", "DataProperty"):
                    continue
                name = str(props.get("name") or "")
                if needle in key.lower() or needle in name.lower():
                    rows.append(
                        {"id": key, "label": label, "properties": dict(props)}
                    )
            return rows[: params.get("limit", 50)]

        if "MATCH (a:" in cypher and "to_label" in cypher:
            match = re.search(r"MATCH \(a:([A-Za-z][A-Za-z0-9_]*)", cypher)
            label = match.group(1) if match else ""
            depth_match = re.search(r"\*1\.\.(\d+)", cypher)
            rel_match = re.search(r"\[r:([A-Za-z][A-Za-z0-9_|]*)\*", cypher)
            allowed = (
                set(rel_match.group(1).split("|")) if rel_match else None
            )
            if depth_match:
                rows = self._bfs_neighbours(
                    label, params.get("key"), int(depth_match.group(1)), allowed
                )
            else:
                rows = self._adjacent(label, params.get("key"))
            return rows[: params.get("limit", 50)]

        if "MATCH (n:" in cypher and "count(n)" in cypher:
            match = re.search(r"MATCH \(n:([A-Za-z][A-Za-z0-9_]*)\)", cypher)
            label = match.group(1) if match else ""
            return [
                {"n": sum(1 for (node_label, _) in self.nodes if node_label == label)}
            ]

        if cypher.startswith("MATCH ()-[r:") and "count(r)" in cypher:
            match = re.search(r"\[r:([A-Za-z][A-Za-z0-9_]*)\]", cypher)
            rel_type = match.group(1) if match else ""
            return [{"n": sum(1 for key in self.edges if key[2] == rel_type)}]

        # Metadata queries (classes / properties / SUBCLASS_OF up-walk /
        # counts) are handled by the parent interpreter.
        result = super().run_query(cypher, params)
        if result:
            return result
        # A generic read query (POST /ontology/query) that no branch
        # above recognised: the canned rows for the test. Metadata
        # queries are excluded — a legitimately empty metadata lookup
        # must stay empty (e.g. get_class of a missing class).
        stripped = cypher.strip().upper()
        is_user_read = stripped.startswith(
            ("MATCH", "WITH", "RETURN", "UNWIND", "CALL", "OPTIONAL")
        )
        is_metadata = any(
            token in cypher
            for token in (
                "OntologyClass",
                "ObjectProperty",
                "DataProperty",
                "SUBCLASS_OF",
                "SHOW CONSTRAINTS",
                "count(",
            )
        )
        if is_user_read and not is_metadata:
            return list(self.query_rows)
        return result


@pytest.fixture(autouse=True)
def _fake_graph_store(monkeypatch):
    store = QueryFakeGraphStore()
    # A small instance graph: O21 -PART_OF-> B1 -HAS_COMPONENT-> M1,
    # plus a derived edge B1 -HAS_COMPONENT-> O21 (inverse of PART_OF).
    store.merge_node("Asset", "O21", {"name": "lift", "status": "active", "version": 1})
    store.merge_node("Building", "B1", {"name": "Tower A", "status": "active", "version": 1})
    store.merge_node("Component", "M1", {"name": "motor", "status": "active", "version": 1})
    store.merge_node("Asset", "O22", {"name": "escalator", "status": "active", "version": 1})
    store.merge_relationship("Asset", "O21", "PART_OF", "Building", "B1", {})
    store.merge_relationship("Building", "B1", "HAS_COMPONENT", "Component", "M1", {})
    store.merge_relationship(
        "Building",
        "B1",
        "HAS_COMPONENT",
        "Asset",
        "O21",
        {"derived": True, "rule": "inverse:PART_OF", "base_ref": "base-1"},
    )
    monkeypatch.setattr(provider_module, "_store", store)
    metadata_store.clear_metadata_cache()
    yield store
    metadata_store.clear_metadata_cache()


def _seed_v1_metadata(store: QueryFakeGraphStore) -> None:
    for class_id in ("Asset", "Building", "Component", "Document"):
        metadata_store.create_class(store, class_id=class_id, name=class_id)
    metadata_store.create_object_property(
        store, property_id="PART_OF", domain="Asset", range="Building",
        inverse_of="HAS_COMPONENT",
    )
    metadata_store.create_object_property(
        store, property_id="HAS_COMPONENT", domain="Building", range="Component",
        inverse_of="PART_OF",
    )


# ----------------------------------------------------------------------
# Read-only Cypher guard (unit)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n:Asset) RETURN n",
        "match (n) return n",
        "WITH 1 AS x RETURN x",
        "RETURN 1",
        "UNWIND [1,2] AS x RETURN x",
        "OPTIONAL MATCH (n) RETURN n",
        "CALL db.labels()",
        "MATCH (n) RETURN n -- MERGE is only in a comment",
        "/* CREATE */ MATCH (n) RETURN n",
    ],
)
def test_read_only_guard_accepts_reads(cypher):
    assert graph_query._assert_read_only_cypher(cypher) == cypher


@pytest.mark.parametrize(
    "cypher",
    [
        "MERGE (n:Asset {id: 'X'})",
        "CREATE (n:Asset)",
        "MATCH (n) DELETE n",
        "MATCH (n) DETACH DELETE n",
        "MATCH (n) SET n.x = 1 RETURN n",
        "MATCH (n) REMOVE n.x RETURN n",
        "DROP CONSTRAINT foo",
        "LOAD CSV FROM 'file:///x.csv' AS row RETURN row",
        "CALL apoc.util.sleep(1)",
        "MATCH (n) CALL apoc.do.when(true, 'RETURN 1', '') YIELD value RETURN value",
        "DELETE (n)",
        "  ",
        "",
    ],
)
def test_read_only_guard_rejects_writes(cypher):
    with pytest.raises(ValueError):
        graph_query._assert_read_only_cypher(cypher)


# ----------------------------------------------------------------------
# POST /ontology/query
# ----------------------------------------------------------------------


def test_query_read_returns_columns_and_rows(_fake_graph_store):
    _fake_graph_store.query_rows = [
        {"id": "O21", "name": "lift"},
        {"id": "O22", "name": "escalator"},
    ]
    resp = client.post(
        "/ontology/query",
        json={"cypher": "MATCH (n:Asset) RETURN n.id AS id, n.name AS name"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["columns"] == ["id", "name"]
    assert len(body["rows"]) == 2
    # The timeout reaches the store (default 30 s).
    assert _fake_graph_store.last_timeout == 30.0


def test_query_with_clause_and_params(_fake_graph_store):
    resp = client.post(
        "/ontology/query",
        json={
            "cypher": "WITH $key AS k MATCH (n {id: k}) RETURN n",
            "params": {"key": "O21"},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["columns"] == ["id", "name"]


def test_query_empty_result_has_empty_columns(_fake_graph_store):
    _fake_graph_store.query_rows = []
    resp = client.post("/ontology/query", json={"cypher": "MATCH (n:Nope) RETURN n"})
    assert resp.status_code == 200
    assert resp.json() == {"columns": [], "rows": []}


@pytest.mark.parametrize(
    "cypher",
    [
        "MERGE (n:Asset {id: 'X'})",
        "CREATE (n:Asset)",
        "MATCH (n) DELETE n",
        "MATCH (n) DETACH DELETE n",
        "MATCH (n) SET n.x = 1 RETURN n",
        "MATCH (n) REMOVE n.x RETURN n",
        "DROP CONSTRAINT foo",
        "CALL apoc.util.sleep(1)",
    ],
)
def test_query_write_keyword_returns_400(cypher):
    resp = client.post("/ontology/query", json={"cypher": cypher})
    assert resp.status_code == 400


def test_query_limit_is_clamped(_fake_graph_store):
    _fake_graph_store.query_rows = [{"id": f"O{i}"} for i in range(10)]
    resp = client.post(
        "/ontology/query",
        json={"cypher": "MATCH (n) RETURN n", "limit": 3},
    )
    assert resp.status_code == 200
    assert len(resp.json()["rows"]) == 3
    resp = client.post(
        "/ontology/query",
        json={"cypher": "MATCH (n) RETURN n", "limit": 9999},
    )
    assert resp.status_code == 200
    assert len(resp.json()["rows"]) == 10  # clamped to 500, only 10 rows exist


def test_query_store_down_returns_503(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr("app.routers.ontology.get_graph_store", _boom)
    resp = client.post("/ontology/query", json={"cypher": "MATCH (n) RETURN n"})
    assert resp.status_code == 503


def test_neo4j_store_run_query_timeout_falls_back_without_keyword():
    """Neo4jGraphStore.run_query(timeout=...) retries without the
    keyword when the driver session does not accept it (the
    recording stub's run() takes only (cypher, params))."""
    from ontology.graph.neo4j_store import Neo4jGraphStore

    driver = _RecordingDriver(rows=[{"n": 1}])
    store = Neo4jGraphStore(driver=driver)
    driver.calls.clear()
    rows = store.run_query("RETURN 1 AS n", timeout=5.0)
    assert rows == [{"n": 1}]
    assert driver.calls == [("RETURN 1 AS n", {})]
    # No timeout: the old call shape is unchanged.
    driver.calls.clear()
    assert store.run_query("RETURN 1 AS n") == [{"n": 1}]
    assert driver.calls == [("RETURN 1 AS n", {})]


# ----------------------------------------------------------------------
# GET /ontology/search
# ----------------------------------------------------------------------


def test_search_by_id_substring_case_insensitive():
    resp = client.get("/ontology/search?q=o21")
    assert resp.status_code == 200
    keys = [row["entity_key"] for row in resp.json()]
    assert keys == ["O21"]


def test_search_by_name_substring():
    resp = client.get("/ontology/search?q=TOWER")
    assert resp.status_code == 200
    assert [row["entity_key"] for row in resp.json()] == ["B1"]
    resp = client.get("/ontology/search?q=motor")
    assert [row["entity_key"] for row in resp.json()] == ["M1"]


def test_search_no_match_returns_empty():
    assert client.get("/ontology/search?q=zzz-no-such").json() == []


def test_search_empty_query_is_422():
    assert client.get("/ontology/search?q=").status_code == 422


def test_search_limit():
    resp = client.get("/ontology/search?q=O&limit=1")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


# ----------------------------------------------------------------------
# GET /ontology/entities?class= (list — must not clash with
# GET /ontology/entities/{type}/{key})
# ----------------------------------------------------------------------


def test_entities_by_class_list():
    resp = client.get("/ontology/entities?class=Asset")
    assert resp.status_code == 200
    assert [row["entity_key"] for row in resp.json()] == ["O21", "O22"]


def test_entities_by_class_includes_subclasses(_fake_graph_store):
    _seed_v1_metadata(_fake_graph_store)
    metadata_store.create_class(_fake_graph_store, class_id="SubAsset", name="SubAsset")
    metadata_store.add_subclass(_fake_graph_store, child_id="SubAsset", parent_id="Asset")
    _fake_graph_store.merge_node("SubAsset", "SA1", {"name": "sub lift"})
    # Subclass instance is included by default (R1)...
    resp = client.get("/ontology/entities?class=Asset")
    assert [row["entity_key"] for row in resp.json()] == ["O21", "O22", "SA1"]
    # ... and excluded when include_subclasses=false.
    resp = client.get("/ontology/entities?class=Asset&include_subclasses=false")
    assert [row["entity_key"] for row in resp.json()] == ["O21", "O22"]


def test_entities_by_class_unknown_class_returns_400():
    assert client.get("/ontology/entities?class=Widget").status_code == 400


def test_entities_list_does_not_shadow_single_entity_get():
    # The 3-segment route still answers single-entity reads.
    resp = client.get("/ontology/entities/Asset/O21")
    assert resp.status_code == 200
    assert resp.json()["entity_key"] == "O21"
    assert resp.json()["properties"]["name"] == "lift"


# ----------------------------------------------------------------------
# GET /ontology/traverse
# ----------------------------------------------------------------------


def test_traverse_unfiltered():
    resp = client.get("/ontology/traverse?root_type=Asset&root_key=O21&depth=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["root_key"] == "O21"
    assert {node["id"] for node in body["nodes"]} == {"O21", "B1", "M1"}
    assert {rel["type"] for rel in body["relationships"]} == {
        "PART_OF",
        "HAS_COMPONENT",
    }


def test_traverse_filters_rel_types():
    resp = client.get(
        "/ontology/traverse?root_type=Asset&root_key=O21&depth=2&rel_types=PART_OF"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert {rel["type"] for rel in body["relationships"]} == {"PART_OF"}
    # M1 is reachable only through HAS_COMPONENT, so it drops out.
    assert {node["id"] for node in body["nodes"]} == {"O21", "B1"}


def test_traverse_rel_types_comma_separated():
    resp = client.get(
        "/ontology/traverse?root_type=Asset&root_key=O21&depth=2"
        "&rel_types=PART_OF,HAS_COMPONENT"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert {rel["type"] for rel in body["relationships"]} == {
        "PART_OF",
        "HAS_COMPONENT",
    }
    assert {node["id"] for node in body["nodes"]} == {"O21", "B1", "M1"}


def test_traverse_unknown_rel_type_returns_400():
    resp = client.get(
        "/ontology/traverse?root_type=Asset&root_key=O21&rel_types=FOO"
    )
    assert resp.status_code == 400


# ----------------------------------------------------------------------
# GET /ontology/schema + GET /ontology/graph/stats
# ----------------------------------------------------------------------


def test_schema_fallback_v1_when_no_metadata():
    resp = client.get("/ontology/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert [c["id"] for c in body["classes"]] == [
        "Asset",
        "Building",
        "Component",
        "Document",
    ]
    assert {p["id"] for p in body["object_properties"]} == {
        "PART_OF",
        "HAS_COMPONENT",
        "LOCATED_IN",
        "RELATED_TO",
        "DOCUMENTED_BY",
    }
    # No SHOW CONSTRAINTS rows from the fake: default UNIQUE (label, id).
    assert body["constraints"] == [
        {"name": f"unique_{label}_id", "labels": [label], "properties": ["id"]}
        for label in ["Asset", "Building", "Component", "Document"]
    ]


def test_schema_uses_metadata_when_seeded(_fake_graph_store):
    _seed_v1_metadata(_fake_graph_store)
    resp = client.get("/ontology/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert [c["id"] for c in body["classes"]] == [
        "Asset",
        "Building",
        "Component",
        "Document",
    ]
    part_of = next(p for p in body["object_properties"] if p["id"] == "PART_OF")
    assert part_of["domain"] == "Asset"
    assert part_of["range"] == "Building"
    assert part_of["inverse_of"] == "HAS_COMPONENT"


def test_schema_parses_show_constraints_rows(_fake_graph_store):
    _fake_graph_store.constraint_rows = [
        {
            "name": "c1",
            "labelsOrTypes": ["Asset"],
            "properties": ["id"],
        }
    ]
    resp = client.get("/ontology/schema")
    assert resp.status_code == 200
    assert resp.json()["constraints"] == [
        {"name": "c1", "labels": ["Asset"], "properties": ["id"]}
    ]


def test_graph_stats_counts(_fake_graph_store):
    resp = client.get("/ontology/graph/stats")
    assert resp.status_code == 200
    body = resp.json()
    # v1 fallback labels/relations (no metadata seeded).
    assert body["nodes_by_label"] == {
        "Asset": 2,
        "Building": 1,
        "Component": 1,
        "Document": 0,
    }
    assert body["edges_by_type"]["PART_OF"] == 1
    assert body["edges_by_type"]["HAS_COMPONENT"] == 2
    assert body["derived_edges"] == 1
    assert body["total_nodes"] == 4
    assert body["total_edges"] == 3


def test_graph_stats_uses_metadata_labels(_fake_graph_store):
    _seed_v1_metadata(_fake_graph_store)
    metadata_store.create_class(_fake_graph_store, class_id="Sensor", name="Sensor")
    _fake_graph_store.merge_node("Sensor", "S1", {"name": "temp"})
    resp = client.get("/ontology/graph/stats")
    assert resp.status_code == 200
    assert resp.json()["nodes_by_label"]["Sensor"] == 1


# ----------------------------------------------------------------------
# Reasoning endpoints
# ----------------------------------------------------------------------


def test_reasoning_rules():
    resp = client.get("/ontology/reasoning/rules")
    assert resp.status_code == 200
    assert [rule["id"] for rule in resp.json()] == ["R1", "R2", "R3"]


def test_reasoning_derived_all():
    resp = client.get("/ontology/reasoning/derived")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["rule"] == "inverse:PART_OF"
    assert body[0]["base_ref"] == "base-1"
    assert body[0]["source_key"] == "B1"
    assert body[0]["target_key"] == "O21"


def test_reasoning_derived_for_entity():
    resp = client.get("/ontology/reasoning/derived?entity_key=O21")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    resp = client.get("/ontology/reasoning/derived?entity_key=M1")
    assert resp.status_code == 200
    assert resp.json() == []
    # entity_type narrows to edges where that endpoint has the label.
    resp = client.get(
        "/ontology/reasoning/derived?entity_type=Building&entity_key=B1"
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    resp = client.get(
        "/ontology/reasoning/derived?entity_type=Component&entity_key=B1"
    )
    assert resp.status_code == 200
    assert resp.json() == []


def test_reasoning_derived_type_without_key_returns_400():
    assert client.get("/ontology/reasoning/derived?entity_type=Asset").status_code == 400


def test_reasoning_recompute_for_entity(monkeypatch):
    seen: dict = {}

    def _fake(store, *, entity_type, entity_key, **k):
        seen["entity"] = (entity_type, entity_key)
        return {"removed": 2, "created": [{"type": "HAS_COMPONENT"}]}

    monkeypatch.setattr(
        "app.routers.ontology.recompute_for_entity", _fake
    )
    resp = client.post(
        "/ontology/reasoning/recompute",
        json={"entity_type": "Asset", "entity_key": "O21"},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "removed": 2,
        "created": [{"type": "HAS_COMPONENT"}],
        "scope": "Asset:O21",
    }
    assert seen["entity"] == ("Asset", "O21")


def test_reasoning_recompute_all(monkeypatch):
    def _fake(store, **k):
        return {"removed": 5, "created": []}

    monkeypatch.setattr("app.routers.ontology.recompute_all", _fake)
    resp = client.post("/ontology/reasoning/recompute", json={"all": True})
    assert resp.status_code == 200
    assert resp.json()["scope"] == "all"
    assert resp.json()["removed"] == 5


def test_reasoning_recompute_needs_a_scope():
    assert client.post("/ontology/reasoning/recompute", json={}).status_code == 400
    assert client.post(
        "/ontology/reasoning/recompute", json={"entity_type": "Asset"}
    ).status_code == 400
    assert client.post(
        "/ontology/reasoning/recompute",
        json={"all": True, "entity_type": "Asset", "entity_key": "O21"},
    ).status_code == 400


# ----------------------------------------------------------------------
# graph_query unit tests (fake store directly)
# ----------------------------------------------------------------------


def test_entities_by_class_unit(_fake_graph_store):
    rows = graph_query.entities_by_class(_fake_graph_store, "Asset")
    assert [row["entity_key"] for row in rows] == ["O21", "O22"]
    assert rows[0]["properties"]["name"] == "lift"


def test_search_entities_unit_excludes_metadata_nodes(_fake_graph_store):
    _fake_graph_store.merge_node("OntologyClass", "Asset", {"name": "Asset"})
    rows = graph_query.search_entities(_fake_graph_store, "asset")
    # The OntologyClass node named "Asset" must not come back.
    assert [row["entity_key"] for row in rows] == []
    assert graph_query.search_entities(_fake_graph_store, "  ") == []


def test_list_derived_edges_unit(_fake_graph_store):
    rows = graph_query.list_derived_edges(_fake_graph_store)
    assert len(rows) == 1
    assert rows[0]["rel_type"] == "HAS_COMPONENT"


def test_traverse_unit_filters_and_keeps_root(_fake_graph_store):
    payload = graph_query.traverse(
        _fake_graph_store, "Asset", "O21", depth=2, rel_types=["HAS_COMPONENT"]
    )
    assert {rel["type"] for rel in payload["relationships"]} == {"HAS_COMPONENT"}
    assert {node["id"] for node in payload["nodes"]} == {"O21", "B1", "M1"}


def test_subclass_descendants_unit(_fake_graph_store):
    _fake_graph_store.subclasses = {("B", "A"), ("C", "B"), ("D", "A")}
    assert subclass_descendants(_fake_graph_store, "A") == ["B", "D", "C"]
    assert subclass_descendants(_fake_graph_store, "C") == []


def test_subclass_descendants_cycle_terminates(_fake_graph_store):
    _fake_graph_store.subclasses = {("B", "A"), ("A", "B"), ("C", "B")}
    assert subclass_descendants(_fake_graph_store, "A") == ["B", "C"]
