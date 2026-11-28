"""Optimisation tests (B1–B14) — main features only.

Mocked stores only: no LiteLLM proxy, no Qdrant/Neo4j server.
Each test pins a behaviour change, not an internal: generic links,
one retriever per execution, substrate/gazetteer caching, batched
graph reads keeping their caps, slim vector fetch keeping exact +
the 4,000-char cut, and hook relations from the stashed expansion.
B12 (drain UNWIND batching) is NOT done — see the delivery report —
so there is no B12 test here.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.config import settings
from infra.qdrant_client import (
    count_documents_per_node,
    get_document_links,
    invalidate_qdrant_substrate_cache,
    search_documents,
)
from runtime.hooks import builtin as builtin_hooks
from runtime.hooks.events import HookContext
from runtime.retrieval_graph import (
    build_live_adjacency,
    clear_gazetteer_codes,
    expand_via_graph_query,
    gazetteer_seeds,
)
from runtime.retrieval_graph import (
    refresh_gazetteer_codes as _real_refresh,
)
from runtime.retrieval_hybrid import build_hybrid_retriever
from runtime.retrieval_linking import compute_document_links
from runtime.retrieval_live import (
    drop_execution_retriever,
    get_execution_retriever,
    hybrid_documents,
)
from tests.retrieval_stubs import GazetteerKeyStore, StubEmbeddings

QUESTION = "F-204 motor overheating O21"


@pytest.fixture(autouse=True)
def _opt_settings(monkeypatch):
    """Dummy embedding credentials (no .env in this VM).

    The factory fails fast on missing embedding config, so the
    tests supply a placeholder key. The gazetteer cache is
    cleared after every test (same hygiene as
    test_retrieval_live._live_settings, minus its refresh stub —
    the B7 tests below need the REAL refresh).
    """
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "qdrant_url", "http://localhost:6333")
    yield
    clear_gazetteer_codes()

LIVE_KEYS = {
    "Asset": ["O21"],
    "Component": ["motor-M2", "pump-P9", "brake-BK1"],
    "Building": ["building-B1"],
    "Document": ["doc-live-motor", "doc-live-pump"],
}


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{doc_id}"))


def _row(frm, rel, to, to_label):
    return {"from_id": frm, "rel_type": rel, "to_id": to, "to_label": to_label}


# --- B1: generic document links ----------------------------------------


def test_b1_generic_links_from_explicit_fields():
    # A real doc (not in the demo table) links from its own fields:
    # asset + component + building, plus its own id (it is a Document
    # node exactly when it has an asset_id — ingest creates the edge).
    assert compute_document_links(
        {"id": "doc-new-1", "asset_id": "O99", "component_id": "motor-M9",
         "building_id": "building-B9"}
    )["node_ids"] == ["O99", "motor-M9", "building-B9", "doc-new-1"]
    # Component-only: no own id (no asset_id -> no DOCUMENTED_BY edge).
    assert compute_document_links(
        {"id": "doc-new-2", "component_id": "motor-M9"}
    )["node_ids"] == ["motor-M9"]
    # No link field at all: no invented link.
    assert compute_document_links({"id": "doc-new-3"})["node_ids"] == []


def test_b1_demo_table_still_wins_and_keeps_asymmetry():
    # The deliberate M05 asymmetry is preserved: doc-motor-overheat
    # links O21 / motor-M2 (+ its building + its own id) — never
    # O22 / motor-M3, even though the remedy text also fits O22's F-206.
    # The demo table is consulted only for demo/seed sources
    # (allow_demo_links=True).
    links = compute_document_links(
        {"id": "doc-motor-overheat"}, allow_demo_links=True
    )["node_ids"]
    assert links == ["O21", "motor-M2", "building-B1", "doc-motor-overheat"]
    assert "O22" not in links and "motor-M3" not in links
    # Real ingest (default) must NOT inherit demo links even when the
    # id collides with a demo id: a bare demo id with no explicit
    # fields links to nothing.
    assert compute_document_links({"id": "doc-motor-overheat"})["node_ids"] == []


# --- B2: one retriever per execution ------------------------------------


class _StubRetriever:
    def __init__(self):
        self.calls = []

    def _resolved_doc_links(self):
        return {}

    def retrieve(self, question, query_plan=None):
        self.calls.append((question, query_plan))
        return [], {"graph_missing": False}


def test_b2_one_retriever_per_execution(monkeypatch):
    builds = []
    monkeypatch.setattr(
        "runtime.retrieval_live.build_hybrid_retriever",
        lambda: builds.append(_StubRetriever()) or builds[-1],
    )
    try:
        first = get_execution_retriever("exec-b2")
        second = get_execution_retriever("exec-b2")
        assert first is second
        assert len(builds) == 1
        assert get_execution_retriever("exec-b2-other") is not first
        drop_execution_retriever("exec-b2")
        assert get_execution_retriever("exec-b2") is not first
        assert len(builds) == 3
    finally:
        drop_execution_retriever("exec-b2")
        drop_execution_retriever("exec-b2-other")


def test_b2_hybrid_documents_reuses_execution_retriever(monkeypatch):
    stub = _StubRetriever()
    monkeypatch.setattr(
        "runtime.retrieval_live.build_hybrid_retriever", lambda: stub
    )
    try:
        hybrid_documents("O21 motor", execution_id="exec-b2-docs")
        hybrid_documents("faults maintenance O21", execution_id="exec-b2-docs")
        # Both seams, different questions, ONE retriever build and
        # one retrieve call each — the substrate is shared, the
        # questions are not merged.
        assert [c[0] for c in stub.calls] == [
            "O21 motor", "faults maintenance O21",
        ]
    finally:
        drop_execution_retriever("exec-b2-docs")


# --- B6/B7: substrate + gazetteer caching --------------------------------


class _ScrollClient:
    """Qdrant stand-in counting scrolls; payloads carry node_ids."""

    def __init__(self, payloads):
        self._payloads = {k: dict(v) for k, v in payloads.items()}
        self.scroll_calls = 0

    def get_collection(self, name):  # noqa: ARG002
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=4))
            )
        )

    def scroll(self, collection_name, limit=100, offset=None, with_payload=True):  # noqa: ARG002
        self.scroll_calls += 1
        doc_ids = sorted(self._payloads)
        start = int(offset) if offset is not None else 0
        page = doc_ids[start:start + limit]
        points = [
            SimpleNamespace(
                id=_point_id(doc_id),
                payload=dict(self._payloads[doc_id]),
            )
            for doc_id in page
        ]
        next_offset = start + len(page) if start + len(page) < len(doc_ids) else None
        return points, next_offset


def test_b6_counts_and_links_share_one_scroll_and_invalidate():
    client = _ScrollClient({
        "doc-a": {"doc_id": "doc-a", "node_ids": ["O21"]},
        "doc-b": {"doc_id": "doc-b", "node_ids": ["O21", "motor-M2"]},
    })
    try:
        assert count_documents_per_node(client=client) == {"O21": 2, "motor-M2": 1}
        assert get_document_links(client=client) == {
            "doc-a": ["O21"], "doc-b": ["O21", "motor-M2"],
        }
        # Two retriever builds' worth of reads, one scroll total.
        assert client.scroll_calls == 1
        count_documents_per_node(client=client)
        assert client.scroll_calls == 1
        invalidate_qdrant_substrate_cache()
        count_documents_per_node(client=client)
        assert client.scroll_calls == 2
    finally:
        invalidate_qdrant_substrate_cache()


def test_b7_gazetteer_refresh_cached_then_forced():
    clear_gazetteer_codes()
    store = GazetteerKeyStore(LIVE_KEYS)
    try:
        first = _real_refresh(store=store)
        assert first == {
            "assets": 1, "components": 3, "buildings": 1,
            "documents": 2, "source": "ontology",
        }
        assert len(store.calls) == 4  # one list_entity_keys per label
        second = _real_refresh()
        assert second == first
        assert len(store.calls) == 4  # cached: no new store call
        _real_refresh(store=store, force=True)
        assert len(store.calls) == 8  # drain/seed refresh re-fetches
    finally:
        clear_gazetteer_codes()


def test_b7_failed_refresh_leaves_cache_empty_and_retries():
    clear_gazetteer_codes()

    class _FailingStore:
        def run_query(self, cypher, params=None):  # noqa: ARG002
            raise RuntimeError("Neo4j down")

    try:
        failed = _real_refresh(store=_FailingStore())
        assert failed["source"] == "unreachable"
        assert gazetteer_seeds(QUESTION) == []  # fail-closed
        # The failure never sets the flag: a good refresh after it
        # is real, not a cached empty summary.
        good = _real_refresh(store=GazetteerKeyStore(LIVE_KEYS))
        assert good["source"] == "ontology"
        assert [s["key"] for s in gazetteer_seeds(QUESTION)] == ["O21"]
    finally:
        clear_gazetteer_codes()


# --- B9: batched graph reads keep the caps -------------------------------


class _CountingKeyStore:
    """Neo4j stand-in answering batched + single-key neighbour reads."""

    def __init__(self, by_key, *, fail_keys=()):
        self._by_key = {k: [dict(r) for r in v] for k, v in by_key.items()}
        self._fail_keys = set(fail_keys)
        self.calls = []

    def run_query(self, cypher, params=None):  # noqa: ARG002
        params = params or {}
        if "keys" in params:
            wanted = params["keys"]
            self.calls.append(tuple(wanted))
            for key in wanted:
                if key in self._fail_keys:
                    raise RuntimeError(f"Neo4j down for {key}")
            out = []
            for key in wanted:
                out.extend(dict(row) for row in self._by_key.get(key, []))
            return out
        key = params.get("key")
        self.calls.append(key)
        if key in self._fail_keys:
            raise RuntimeError(f"Neo4j down for {key}")
        return [dict(row) for row in self._by_key.get(key, [])]


def test_b9_expansion_is_one_batched_call_for_many_seeds():
    graph = {
        "O21": [_row("O21", "HAS_COMPONENT", "motor-M2", "Component")],
        "motor-M2": [_row("motor-M2", "DOCUMENTED_BY", "doc-m", "Document")],
        "pump-P9": [_row("pump-P9", "DOCUMENTED_BY", "doc-p", "Document")],
    }
    store = _CountingKeyStore(graph)
    seeds = [
        {"type": "Asset", "key": "O21", "confidence": 1.0},
        {"type": "Component", "key": "motor-M2", "confidence": 1.0},
        {"type": "Component", "key": "pump-P9", "confidence": 1.0},
    ]
    expansion = expand_via_graph_query(seeds, store=store)
    assert store.calls == [("O21", "motor-M2", "pump-P9")]  # was 3 calls
    assert set(expansion["nodes"]) == {"motor-M2", "doc-m", "doc-p"}


def test_b9_live_adjacency_batches_per_level_and_keeps_caps():
    graph = {
        "O21": [_row("O21", "HAS_COMPONENT", "motor-M2", "Component"),
                _row("O21", "HAS_COMPONENT", "pump-P9", "Component")],
        "motor-M2": [_row("motor-M2", "DOCUMENTED_BY", "doc-m", "Document")],
        "pump-P9": [_row("pump-P9", "DOCUMENTED_BY", "doc-p", "Document")],
        "doc-m": [],
        "doc-p": [],
    }
    store = _CountingKeyStore(graph)
    seeds = [{"type": "Asset", "key": "O21", "confidence": 1.0}]
    expansion = expand_via_graph_query(seeds, store=store)
    assert len(store.calls) == 1
    adjacency = build_live_adjacency(seeds, expansion, store=store)
    # Level 0 (seeds + expansion) in one call, level 1 (second hop)
    # in one call — 2 batched calls, not 5 per-node calls.
    assert len(store.calls) == 3
    assert set(adjacency) == {"O21", "motor-M2", "pump-P9", "doc-m", "doc-p"}
    # The max_nodes cap still binds deterministically.
    capped = build_live_adjacency(
        seeds, expansion, store=_CountingKeyStore(graph), max_nodes=3
    )
    assert set(capped) == {"O21", "motor-M2", "pump-P9"}
    assert "doc-m" not in capped and "doc-p" not in capped


def test_b9_expansion_fanout_cap_still_binds():
    rows = {"S1": [_row("S1", "RELATED_TO", f"n{i:02d}", "Component")
                   for i in range(1, 26)]}
    store = _CountingKeyStore(rows)
    expansion = expand_via_graph_query(
        [{"type": "Asset", "key": "S1", "confidence": 1.0}], store=store
    )
    assert len(store.calls) == 1
    assert expansion["nodes_returned"] == 10  # fan-out 10 per type
    assert expansion["truncated"] is True


# --- B8: hook relations from the stashed expansion ------------------------


def test_b8_hook_relations_reuse_expansion(monkeypatch):
    stub = SimpleNamespace(
        _last_expansion={
            "nodes": {
                "motor-M2": {"type": "Component", "rel": "HAS_COMPONENT",
                             "from_seed": "O21", "via_hub": False},
                "doc-m": {"type": "Document", "rel": "DOCUMENTED_BY",
                          "from_seed": "O21", "via_hub": False},
                "other": {"type": "Component", "rel": "RELATED_TO",
                          "from_seed": "O99", "via_hub": False},
            },
        }
    )
    monkeypatch.setattr(
        "runtime.retrieval_live.build_hybrid_retriever", lambda: stub
    )
    try:
        get_execution_retriever("exec-b8")

        def _no_neo4j(asset_id):  # noqa: ARG001
            raise AssertionError("Neo4j must not be called when expansion holds the rows")

        monkeypatch.setattr(
            "infra.repositories.query_asset_relationships", _no_neo4j
        )
        ctx = HookContext(execution_id="exec-b8", mission_id="fault_analysis")
        rows, _is_stub = builtin_hooks._load_relations(ctx, "O21")
        assert rows == [
            {"from": "O21", "type": "DOCUMENTED_BY", "to": "doc-m",
             "to_label": "Document"},
            {"from": "O21", "type": "HAS_COMPONENT", "to": "motor-M2",
             "to_label": "Component"},
        ]
    finally:
        drop_execution_retriever("exec-b8")


# --- B14: slim vector fetch ------------------------------------------------


class _SlimClient:
    """Qdrant stand-in honouring a with_payload field list."""

    def __init__(self, payloads, *, vector_ids=None):
        self._payloads = {k: dict(v) for k, v in payloads.items()}
        self._vector_ids = list(vector_ids) if vector_ids is not None else sorted(payloads)

    def get_collection(self, name):  # noqa: ARG002
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=4))
            )
        )

    def query_points(self, collection_name, query, limit, with_payload=True):  # noqa: ARG002
        points = []
        for i, doc_id in enumerate(self._vector_ids[:limit]):
            payload = dict(self._payloads[doc_id])
            if isinstance(with_payload, list):
                payload = {k: v for k, v in payload.items() if k in with_payload}
            points.append(SimpleNamespace(
                id=doc_id, score=0.9 - 0.01 * i, payload=payload,
            ))
        return SimpleNamespace(points=points)

    def scroll(self, collection_name, limit=100, offset=None, with_payload=True):  # noqa: ARG002
        doc_ids = sorted(self._payloads)
        start = int(offset) if offset is not None else 0
        page = doc_ids[start:start + limit]
        points = [
            SimpleNamespace(
                id=_point_id(doc_id),
                payload=dict(self._payloads[doc_id]),
            )
            for doc_id in page
        ]
        next_offset = start + len(page) if start + len(page) < len(doc_ids) else None
        return points, next_offset

    def retrieve(self, collection_name, ids, with_payload=True):  # noqa: ARG002
        by_point = {_point_id(doc_id): doc_id for doc_id in self._payloads}
        points = []
        for point_id in ids:
            doc_id = by_point.get(str(point_id))
            if doc_id is None:
                continue
            points.append(SimpleNamespace(
                id=str(point_id), payload=dict(self._payloads[doc_id]),
            ))
        return points


def test_b14_slim_search_fetches_no_content():
    client = _SlimClient({
        "doc-a": {"doc_id": "doc-a", "title": "Motor guide",
                  "content": "x" * 9000, "node_ids": ["O21"]},
    })
    rows = search_documents(
        "O21 motor", limit=5, embeddings=StubEmbeddings(), client=client, slim=True,
    )
    assert rows == [{
        "doc_id": "doc-a", "title": "Motor guide", "node_ids": ["O21"],
        "score": 0.9,
    }]
    assert "content" not in rows[0]
    # The plain path is unchanged: full payload + content present.
    full = search_documents(
        "O21 motor", limit=5, embeddings=StubEmbeddings(), client=client,
    )
    assert full[0]["content"] == "x" * 9000


def test_b14_exact_uses_fetched_content_and_cut_is_4000(monkeypatch):
    monkeypatch.setattr(
        "runtime.retrieval_graph.refresh_gazetteer_codes",
        lambda store=None, **kw: {"source": "stubbed"},
    )
    clear_gazetteer_codes()
    try:
        _real_refresh(store=GazetteerKeyStore(LIVE_KEYS))
        # Title has NO code; the code lives only in the content, and
        # the content is longer than the 4,000-char cut.
        long_content = "O21 motor-M2 overheats, fault F-204. " + "x" * 5000
        payloads = {
            "doc-live-motor": {
                "doc_id": "doc-live-motor", "title": "Motor overheat guide",
                "content": long_content, "node_ids": ["O21", "motor-M2"],
            },
        }
        client = _SlimClient(payloads, vector_ids=["doc-live-motor"])
        store = _CountingKeyStore({
            "O21": [_row("O21", "HAS_COMPONENT", "motor-M2", "Component")],
            "motor-M2": [],
        })
        retriever = build_hybrid_retriever(
            embeddings=StubEmbeddings(), client=client, graph_store=store,
        )
        candidates, _seeds, _exp, _gm = retriever._candidate_rows(QUESTION)
        motor = next(c for c in candidates if c["doc_id"] == "doc-live-motor")
        # exact=1 comes from the FETCHED content, not the slim row:
        # without the RRF top-30 fill it would be 0 (title has no code).
        assert motor["exact"] == 1
        # Same cut as the plain repositories.search_documents path:
        # 4,000 chars + the truncation marker.
        assert motor["content"] == long_content[:4000].rstrip() + " … (truncated)"
        assert motor["content"].startswith("O21 motor-M2 overheats, fault F-204.")
    finally:
        clear_gazetteer_codes()
