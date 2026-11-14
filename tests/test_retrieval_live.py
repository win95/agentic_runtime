"""Phase 5 live wiring tests — live factory + real-source doc counts/texts.

Mocked stores only: no LiteLLM proxy, no Qdrant/Neo4j server.
Main features only — no parametrized explosion.

The gazetteer cache is empty unless a test refreshes it: the
autouse fixture stubs ``refresh_gazetteer_codes`` (so the factory
never touches the real store), and tests that need seeds call
``_refresh_live_keys`` first — the stub refresh during the factory
build leaves the populated cache alone.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.config import settings
from infra.qdrant_client import (
    count_documents_per_node,
    get_document_links,
    get_document_payloads,
)
from runtime.retrieval_graph import (
    ALLOWED_RELATIONS,
    build_expansion_adjacency,
    build_live_adjacency,
    clear_gazetteer_codes,
    expand_via_graph_query,
    gazetteer_seeds,
    graph_features,
)
from runtime.retrieval_graph import (
    refresh_gazetteer_codes as _real_refresh_gazetteer_codes,
)
from runtime.retrieval_hybrid import (
    HybridRetriever,
    build_hybrid_retriever,
)
from runtime.retrieval_ppr import personalized_pagerank, seed_specificity
from tests.retrieval_stubs import (
    GazetteerKeyStore,
    StubEmbeddings,
    canned_adjacency,
)

QUESTION = "F-204 motor overheating O21"

LIVE_PAYLOADS = {
    "doc-live-motor": {
        "doc_id": "doc-live-motor",
        "title": "Motor overheat F-204",
        "content": "O21 motor-M2 overheats, fault F-204",
        "node_ids": ["O21", "motor-M2", "doc-live-motor"],
    },
    "doc-live-brake": {
        "doc_id": "doc-live-brake",
        "title": "Brake inspection",
        "content": "brake-BK1 inspection",
        "node_ids": ["O21", "brake-BK1"],
    },
}

# Keys the live ontology holds in these tests. QUESTION seeds to
# [O21] only: "motor" bare does not match motor-M2, and fault
# codes link nothing.
LIVE_GAZETTEER_KEYS = {
    "Asset": ["O21"],
    "Component": ["motor-M2", "pump-P9", "brake-BK1"],
    "Building": ["building-B1"],
    "Document": ["doc-live-motor", "doc-live-pump"],
}


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{doc_id}"))


def _refresh_live_keys() -> dict:
    """Populate the gazetteer cache with the live keys (real refresh)."""
    return _real_refresh_gazetteer_codes(store=GazetteerKeyStore(LIVE_GAZETTEER_KEYS))


class MockQdrantClient:
    """In-memory Qdrant stand-in with real-source payloads.

    ``vector_ids`` are the docs the vector arm returns (the rest are
    graph-only); ``fail_search`` makes ``query_points`` raise.
    """

    def __init__(self, payloads, *, vector_ids=None, fail_search=False):
        self._payloads = {k: dict(v) for k, v in payloads.items()}
        self._vector_ids = list(vector_ids) if vector_ids is not None else sorted(payloads)
        self._fail_search = fail_search
        self.scroll_calls = 0
        self.retrieve_calls = 0

    def get_collection(self, name):  # noqa: ARG002
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=4))
            )
        )

    def query_points(self, collection_name, query, limit, with_payload=True):  # noqa: ARG002
        if self._fail_search:
            raise RuntimeError("connection refused")
        points = [
            SimpleNamespace(
                id=doc_id,
                score=0.9 - 0.01 * i,
                payload=dict(self._payloads[doc_id]) if with_payload else {},
            )
            for i, doc_id in enumerate(self._vector_ids[:limit])
        ]
        return SimpleNamespace(points=points)

    def scroll(self, collection_name, limit=100, offset=None, with_payload=True):  # noqa: ARG002
        self.scroll_calls += 1
        doc_ids = sorted(self._payloads)
        start = int(offset) if offset is not None else 0
        page = doc_ids[start:start + limit]
        points = [
            SimpleNamespace(
                id=_point_id(doc_id),
                payload=dict(self._payloads[doc_id]) if with_payload else {},
            )
            for doc_id in page
        ]
        next_offset = start + len(page) if start + len(page) < len(doc_ids) else None
        return points, next_offset

    def retrieve(self, collection_name, ids, with_payload=True):  # noqa: ARG002
        self.retrieve_calls += 1
        by_point = {_point_id(doc_id): doc_id for doc_id in self._payloads}
        points = []
        for point_id in ids:
            doc_id = by_point.get(str(point_id))
            if doc_id is None:
                continue
            points.append(SimpleNamespace(
                id=str(point_id),
                payload=dict(self._payloads[doc_id]) if with_payload else {},
            ))
        return points


class MockGraphStore:
    """Neo4j stand-in: ``run_query`` returns fixed rows or raises."""

    def __init__(self, rows=None, *, fail=False):
        self._rows = list(rows or [])
        self._fail = fail
        self.calls = 0

    def run_query(self, cypher, params=None):  # noqa: ARG002
        self.calls += 1
        if self._fail:
            raise RuntimeError("Neo4j down")
        return [dict(row) for row in self._rows]


@pytest.fixture(autouse=True)
def _live_settings(monkeypatch):
    """Dummy embedding credentials (no .env in this VM) + stubbed gazetteer.

    The factory fails fast on missing embedding config, so the tests
    supply a placeholder key; the gazetteer refresh is stubbed out
    per test (the real one flips global cache state — tests that
    need seeds call ``_refresh_live_keys`` explicitly). The cache
    is cleared after every test.
    """
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "qdrant_url", "http://localhost:6333")
    monkeypatch.setattr(
        "runtime.retrieval_graph.refresh_gazetteer_codes",
        lambda store=None: {"assets": 0, "components": 0, "buildings": 0,
                            "documents": 0, "source": "stubbed"},
    )
    yield
    clear_gazetteer_codes()


def _live_retriever(client=None, store=None, **kwargs):
    client = client if client is not None else MockQdrantClient(
        LIVE_PAYLOADS, vector_ids=["doc-live-brake"]
    )
    store = store if store is not None else MockGraphStore(rows=[
        {"from_id": "O21", "rel_type": "DOCUMENTED_BY",
         "to_id": "doc-live-motor", "to_label": "Document"},
    ])
    return build_hybrid_retriever(
        embeddings=StubEmbeddings(), client=client, graph_store=store, **kwargs
    )


# --- Live factory ----------------------------------------------------------

def test_live_factory_injects_real_stores(monkeypatch):
    refresh_calls = []

    def _refresh(store=None):
        refresh_calls.append(store)
        return {"source": "stubbed"}

    monkeypatch.setattr(
        "runtime.retrieval_graph.refresh_gazetteer_codes", _refresh
    )
    embeddings = StubEmbeddings()
    client = MockQdrantClient(LIVE_PAYLOADS)
    store = MockGraphStore()
    retriever = build_hybrid_retriever(
        embeddings=embeddings, client=client, graph_store=store
    )
    assert retriever.embeddings is embeddings
    assert retriever.client is client
    assert retriever.graph_store is store
    assert retriever.use_live_graph is True
    assert retriever.__name__ == "hybrid"
    assert refresh_calls == [store]


def test_live_factory_builds_query_model_embeddings(monkeypatch):
    built = {}

    def _build_embeddings(model=None, **kwargs):  # noqa: ARG002
        built["model"] = model
        return StubEmbeddings()

    monkeypatch.setattr("runtime.llm_factory.build_embeddings", _build_embeddings)
    retriever = build_hybrid_retriever(client=MockQdrantClient(LIVE_PAYLOADS))
    assert built["model"] == settings.resolved_embedding_model_query
    assert built["model"] == "text-embedding-3-small"  # EMBEDDING_MODEL default
    assert isinstance(retriever.embeddings, StubEmbeddings)


def test_live_factory_fails_fast_without_qdrant_url(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_url", "")
    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        build_hybrid_retriever(
            embeddings=StubEmbeddings(), client=MockQdrantClient(LIVE_PAYLOADS)
        )


def test_live_factory_fails_fast_without_embedding_config(monkeypatch):
    monkeypatch.setattr(settings, "embedding_model", "")
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        build_hybrid_retriever(
            embeddings=StubEmbeddings(), client=MockQdrantClient(LIVE_PAYLOADS)
        )


def test_live_factory_refresh_failure_does_not_crash(monkeypatch):
    def _boom(store=None):  # noqa: ARG002
        raise RuntimeError("Neo4j down")

    monkeypatch.setattr("runtime.retrieval_graph.refresh_gazetteer_codes", _boom)
    retriever = build_hybrid_retriever(
        embeddings=StubEmbeddings(), client=MockQdrantClient(LIVE_PAYLOADS)
    )
    assert retriever.use_live_graph is True


def test_live_factory_refresh_missing_does_not_crash(monkeypatch):
    monkeypatch.delattr(
        "runtime.retrieval_graph.refresh_gazetteer_codes", raising=False
    )
    retriever = build_hybrid_retriever(
        embeddings=StubEmbeddings(), client=MockQdrantClient(LIVE_PAYLOADS)
    )
    assert retriever.use_live_graph is True


def test_live_factory_tolerates_real_refresh_finding_nothing(monkeypatch):
    # The real refresh against an empty store returns source
    # "ontology" with zero keys and never raises; the factory must
    # still build (gazetteer state is restored by the fixture).
    monkeypatch.setattr(
        "runtime.retrieval_graph.refresh_gazetteer_codes",
        _real_refresh_gazetteer_codes,
    )
    retriever = build_hybrid_retriever(
        embeddings=StubEmbeddings(),
        client=MockQdrantClient(LIVE_PAYLOADS),
        graph_store=MockGraphStore(rows=[]),
    )
    assert retriever.use_live_graph is True


# --- Error contract ----------------------------------------------------------

def test_qdrant_search_failure_surfaces_as_runtime_error():
    _refresh_live_keys()
    retriever = _live_retriever(
        client=MockQdrantClient(LIVE_PAYLOADS, fail_search=True)
    )
    with pytest.raises(RuntimeError, match="Qdrant"):
        retriever.retrieve(QUESTION)


def test_graph_store_failure_falls_back_to_vector_only():
    _refresh_live_keys()
    retriever = _live_retriever(store=MockGraphStore(fail=True))
    ranked, trace = retriever.retrieve(QUESTION)
    assert trace["graph_missing"] is True
    assert trace["expansion_nodes_returned"] == 0
    # The graph arm still finds the motor doc through Qdrant payload
    # links (not the graph store), so it is ranked, vector-only.
    assert "doc-live-motor" in ranked
    assert "doc-live-brake" in ranked


# --- Real-source doc counts / links / texts ---------------------------------

def test_count_documents_per_node_from_payloads():
    counts = count_documents_per_node(client=MockQdrantClient(LIVE_PAYLOADS))
    assert counts == {
        "O21": 2, "motor-M2": 1, "doc-live-motor": 1, "brake-BK1": 1,
    }


def test_count_documents_per_node_scrolls_past_100():
    payloads = {
        f"doc-batch-{i:03d}": {
            "doc_id": f"doc-batch-{i:03d}",
            "title": "t", "content": "c", "node_ids": ["O21"],
        }
        for i in range(250)
    }
    client = MockQdrantClient(payloads)
    assert count_documents_per_node(client=client) == {"O21": 250}
    assert client.scroll_calls == 3  # 100 + 100 + 50


def test_count_documents_per_node_empty_store():
    assert count_documents_per_node(client=MockQdrantClient({})) == {}


def test_count_documents_per_node_no_client_raises(monkeypatch):
    monkeypatch.setattr("infra.qdrant_client.get_client", lambda: None)
    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        count_documents_per_node()


def test_get_document_links_from_payloads():
    assert get_document_links(client=MockQdrantClient(LIVE_PAYLOADS)) == {
        "doc-live-motor": ["O21", "motor-M2", "doc-live-motor"],
        "doc-live-brake": ["O21", "brake-BK1"],
    }


def test_get_document_payloads_from_mock_client():
    payloads = get_document_payloads(
        ["doc-live-motor", "doc-nope"], client=MockQdrantClient(LIVE_PAYLOADS)
    )
    assert list(payloads) == ["doc-live-motor"]
    assert payloads["doc-live-motor"]["title"] == "Motor overheat F-204"


def test_get_document_payloads_empty_list_never_calls_client():
    client = MockQdrantClient(LIVE_PAYLOADS)
    assert get_document_payloads([], client=client) == {}
    assert client.retrieve_calls == 0


def test_live_retriever_reads_counts_and_texts_from_qdrant():
    _refresh_live_keys()
    client = MockQdrantClient(LIVE_PAYLOADS, vector_ids=["doc-live-brake"])
    retriever = _live_retriever(client=client)
    # Lazy: no Qdrant scroll before the first query.
    assert client.scroll_calls == 0
    assert retriever._doc_counts_cache is None
    ranked, trace = retriever.retrieve(QUESTION)
    assert client.scroll_calls >= 1
    assert retriever._resolved_doc_counts() == {
        "O21": 2, "motor-M2": 1, "doc-live-motor": 1, "brake-BK1": 1,
    }
    # doc-live-motor is graph-only (not in the vector results): its
    # title/content come from the payload fetch, not a canned corpus.
    rows = {row["doc_id"]: row for row in trace["feature_rows"]}
    assert "doc-live-motor" in rows
    assert "doc-live-motor" in ranked
    candidates, _seeds, _exp, _gm = retriever._candidate_rows(QUESTION)
    motor = next(c for c in candidates if c["doc_id"] == "doc-live-motor")
    assert motor["title"] == "Motor overheat F-204"
    assert motor["content"] == "O21 motor-M2 overheats, fault F-204"
    assert motor["exact"] == 1  # F-204 in the fetched content
    assert motor["arms"] == ["graph"]


def test_live_retriever_empty_store_degrades():
    _refresh_live_keys()
    retriever = _live_retriever(
        client=MockQdrantClient({}, vector_ids=[]),
        store=MockGraphStore(rows=[]),
    )
    ranked, trace = retriever.retrieve(QUESTION)
    assert ranked == []
    assert retriever._resolved_doc_counts() == {}
    assert trace["graph_missing"] is True


def test_precomputed_dict_sources_are_used():
    _refresh_live_keys()
    adjacency = {
        "O21": {"type": "Asset", "neighbors": [
            ("DOCUMENTED_BY", "doc-live-motor", "Document"),
        ]},
        "motor-M2": {"type": "Component", "neighbors": [
            ("DOCUMENTED_BY", "doc-live-motor", "Document"),
        ]},
        "doc-live-motor": {"type": "Document", "neighbors": [
            ("DOCUMENTED_BY", "O21", "Asset"),
            ("DOCUMENTED_BY", "motor-M2", "Component"),
        ]},
    }
    retriever = HybridRetriever(
        embeddings=StubEmbeddings(),
        client=MockQdrantClient(LIVE_PAYLOADS),
        adjacency=adjacency,
        doc_counts={"O21": 100, "motor-M2": 1},
        doc_links={"doc-live-motor": ["O21", "motor-M2"]},
        doc_texts={"doc-live-motor": {
            "title": "Motor overheat F-204", "content": "F-204",
        }},
    )
    assert retriever._resolved_doc_counts() == {"O21": 100, "motor-M2": 1}
    assert retriever._resolved_doc_links() == {
        "doc-live-motor": ["O21", "motor-M2"]
    }
    ranked, _trace = retriever.retrieve(QUESTION)
    assert "doc-live-motor" in ranked


# --- Live adjacency (PPR / coverage / hub substrate) ---------------------

def _row(frm, rel, to, to_label):
    return {"from_id": frm, "rel_type": rel, "to_id": to, "to_label": to_label}


# Canned live graph. The pump branch exists ONLY here (never in the
# canned adjacency): O21 -> pump-P9 -> doc-live-pump is the real
# second-hop edge that proves the live substrate is used.
LIVE_GRAPH = {
    "O21": [
        _row("O21", "HAS_COMPONENT", "motor-M2", "Component"),
        _row("O21", "HAS_COMPONENT", "pump-P9", "Component"),
        _row("O21", "DOCUMENTED_BY", "doc-live-motor", "Document"),
        _row("O21", "LOCATED_IN", "building-B1", "Building"),
        _row("O21", "OWNS", "owner-X", "Entity"),  # not allowlisted
    ],
    "motor-M2": [
        _row("motor-M2", "HAS_COMPONENT", "O21", "Asset"),
        _row("motor-M2", "DOCUMENTED_BY", "doc-live-motor", "Document"),
    ],
    "pump-P9": [
        _row("pump-P9", "HAS_COMPONENT", "O21", "Asset"),
        _row("pump-P9", "DOCUMENTED_BY", "doc-live-pump", "Document"),
    ],
    "doc-live-motor": [
        _row("doc-live-motor", "DOCUMENTED_BY", "O21", "Asset"),
        _row("doc-live-motor", "DOCUMENTED_BY", "motor-M2", "Component"),
    ],
    "doc-live-pump": [
        _row("doc-live-pump", "DOCUMENTED_BY", "pump-P9", "Component"),
    ],
    "building-B1": [
        _row("building-B1", "LOCATED_IN", "O21", "Asset"),
    ],
}

LIVE_SEEDS = [
    {"type": "Asset", "key": "O21", "confidence": 1.0},
    {"type": "Component", "key": "motor-M2", "confidence": 1.0},
]

LIVE_PUMP_PAYLOADS = {
    **LIVE_PAYLOADS,
    "doc-live-pump": {
        "doc_id": "doc-live-pump",
        "title": "Pump inspection",
        "content": "pump-P9 inspection",
        "node_ids": ["pump-P9", "doc-live-pump"],
    },
}


class KeyedMockGraphStore:
    """Neo4j stand-in returning per-key neighbour rows.

    ``by_key`` maps a node key -> rows as ``graph_query.neighbors``
    returns them (``from_id``/``rel_type``/``to_id``/``to_label``);
    an unknown key returns []. ``fail_keys`` makes ``run_query``
    raise for that key only (a mid-build store failure — the
    expansion, which only queries seeds, still succeeds).
    """

    def __init__(self, by_key, *, fail_keys=()):
        self._by_key = {k: [dict(r) for r in v] for k, v in by_key.items()}
        self._fail_keys = set(fail_keys)
        self.calls = []

    def run_query(self, cypher, params=None):  # noqa: ARG002
        key = (params or {}).get("key")
        self.calls.append(key)
        if key in self._fail_keys:
            raise RuntimeError(f"Neo4j down for {key}")
        return [dict(row) for row in self._by_key.get(key, [])]


def test_build_live_adjacency_has_real_second_hop_edges():
    store = KeyedMockGraphStore(LIVE_GRAPH)
    expansion = expand_via_graph_query(LIVE_SEEDS, store=store)
    adjacency = build_live_adjacency(LIVE_SEEDS, expansion, store=store)
    assert set(adjacency) == {
        "O21",
        "motor-M2",
        "pump-P9",
        "doc-live-motor",
        "doc-live-pump",
        "building-B1",
    }
    # Canned-adjacency shape: type + neighbour (rel, to_key, to_type) tuples.
    assert adjacency["doc-live-pump"]["type"] == "Document"
    assert ("DOCUMENTED_BY", "doc-live-pump", "Document") in (adjacency["pump-P9"]["neighbors"])
    assert ("DOCUMENTED_BY", "pump-P9", "Component") in (adjacency["doc-live-pump"]["neighbors"])
    # Neighbours sorted by key, every edge allowlisted.
    for entry in adjacency.values():
        keys = [to for _rel, to, _t in entry["neighbors"]]
        assert keys == sorted(keys)
        assert all(rel in ALLOWED_RELATIONS for rel, _to, _t in entry["neighbors"])
    # The disallowed OWNS edge (and its target) never lands.
    assert "owner-X" not in adjacency


def test_build_live_adjacency_no_seeds_is_empty():
    assert build_live_adjacency([], {"nodes": {}}, store=KeyedMockGraphStore({})) == {}


def test_build_live_adjacency_cap_is_deterministic():
    store = KeyedMockGraphStore(LIVE_GRAPH)
    expansion = expand_via_graph_query(LIVE_SEEDS, store=store)
    first = build_live_adjacency(LIVE_SEEDS, expansion, store=store, max_nodes=5)
    second = build_live_adjacency(LIVE_SEEDS, expansion, store=store, max_nodes=5)
    assert first == second
    # Seeds + expansion nodes are the 5 keys; the second-hop doc is
    # the one dropped when the cap binds.
    assert set(first) == {
        "O21",
        "motor-M2",
        "pump-P9",
        "doc-live-motor",
        "building-B1",
    }
    assert "doc-live-pump" not in first


def test_build_live_adjacency_cap_keeps_sorted_first_neighbours():
    rows = {"S1": [_row("S1", "RELATED_TO", "n01", "Component")]}
    rows["n01"] = [_row("n01", "RELATED_TO", f"m{i:02d}", "Component") for i in range(1, 11)]
    rows.update({f"m{i:02d}": [] for i in range(1, 11)})
    store = KeyedMockGraphStore(rows)
    seeds = [{"type": "Asset", "key": "S1", "confidence": 1.0}]
    expansion = {
        "nodes": {
            "n01": {"type": "Component", "rel": "RELATED_TO", "from_seed": "S1", "via_hub": False},
        }
    }
    adjacency = build_live_adjacency(seeds, expansion, store=store, max_nodes=4)
    assert set(adjacency) == {"S1", "n01", "m01", "m02"}
    assert ("RELATED_TO", "m01", "Component") in adjacency["n01"]["neighbors"]


def test_build_live_adjacency_timeout_returns_partial():
    store = KeyedMockGraphStore(LIVE_GRAPH)
    expansion = expand_via_graph_query(LIVE_SEEDS, store=store)
    adjacency = build_live_adjacency(LIVE_SEEDS, expansion, store=store, timeout_seconds=0)
    # No store call ran, but seeds + expansion nodes are present —
    # the partial graph is never empty for a non-empty seed set.
    assert set(adjacency) == {
        "O21",
        "motor-M2",
        "pump-P9",
        "doc-live-motor",
        "building-B1",
    }
    assert all(entry["neighbors"] == [] for entry in adjacency.values())


def test_live_retriever_uses_live_adjacency_for_ppr_coverage():
    _refresh_live_keys()
    client = MockQdrantClient(LIVE_PUMP_PAYLOADS, vector_ids=["doc-live-brake"])
    retriever = _live_retriever(client=client, store=KeyedMockGraphStore(LIVE_GRAPH))
    ranked, trace = retriever.retrieve(QUESTION)
    assert trace["graph_adjacency_source"] == "live"
    assert trace["graph_adjacency_nodes"] == 6
    assert "doc-live-pump" in ranked
    rows = {row["doc_id"]: row for row in trace["feature_rows"]}
    pump = rows["doc-live-pump"]
    # 2-hop coverage via the real second-hop edge (pump-P9 -> O21):
    # the question seeds to [O21] only, and the pump doc is within
    # 2 hops of that one seed. PPR mass reaches the live-only doc.
    assert pump["g6_coverage"] == 1.0
    assert pump["g7_ppr"] > 0
    # The same question scored over the canned adjacency: the
    # live-only doc is not a canned node, so both features are zero —
    # proving the live values above came from the live substrate.
    canned = canned_adjacency()
    seeds = gazetteer_seeds(QUESTION)
    demo_ppr = personalized_pagerank(canned, seeds)
    demo_feats = graph_features("doc-live-pump", seeds, {"nodes": {}}, canned, ppr_scores=demo_ppr)
    assert demo_feats["g6_coverage"] == 0.0
    assert demo_feats["g7_ppr"] == 0.0
    assert pump["g6_coverage"] != demo_feats["g6_coverage"]
    assert pump["g7_ppr"] != demo_feats["g7_ppr"]


def test_live_build_failure_falls_back_to_expansion_only():
    _refresh_live_keys()
    client = MockQdrantClient(LIVE_PUMP_PAYLOADS, vector_ids=["doc-live-brake"])
    # pump-P9 is never queried by the 1-hop expansion (seeds only),
    # so the expansion succeeds and the build fails mid-way.
    store = KeyedMockGraphStore(LIVE_GRAPH, fail_keys={"pump-P9"})
    retriever = _live_retriever(client=client, store=store)
    ranked, trace = retriever.retrieve(QUESTION)
    assert trace["graph_adjacency_source"] == "expansion-only"
    assert "doc-live-motor" in ranked
    rows = {row["doc_id"]: row for row in trace["feature_rows"]}
    # The second-hop doc is still a candidate (linked via the
    # expansion node pump-P9) but its live coverage/PPR are zero —
    # no canned edges were substituted for the failed live build.
    assert rows["doc-live-pump"]["g6_coverage"] == 0.0
    assert rows["doc-live-pump"]["g7_ppr"] == 0.0
    # And the surviving features equal the expansion-only
    # computation exactly (not the canned-adjacency values).
    seeds = gazetteer_seeds(QUESTION)
    expansion = expand_via_graph_query(seeds, store=KeyedMockGraphStore(LIVE_GRAPH))
    fallback = build_expansion_adjacency(seeds, expansion)
    assert "doc-live-pump" not in fallback
    spec = seed_specificity(seeds, retriever._resolved_doc_counts())
    weights = {s["key"]: float(s.get("confidence", 1.0)) * spec[s["key"]] for s in seeds}
    ppr = personalized_pagerank(fallback, seeds, seed_weights=weights)
    assert rows["doc-live-motor"]["g7_ppr"] == round(ppr["doc-live-motor"], 6)


def test_live_empty_expansion_stays_graph_missing():
    _refresh_live_keys()
    retriever = _live_retriever(store=MockGraphStore(rows=[]))
    ranked, trace = retriever.retrieve(QUESTION)
    assert trace["graph_missing"] is True
    assert "graph_adjacency_source" not in trace  # no live build attempted
    assert "doc-live-motor" in ranked  # payload links still rank it
