"""Wiring tests: hybrid retrieval into the hook + capability node.

The shared helper (``runtime.retrieval_live.hybrid_documents``) and
its two production call sites — the ``retrieval_context`` hook (Seam
A) and the ``knowledge_retrieval`` capability node (Seam B). All
stores are mocked; no Qdrant/Neo4j/LLM is touched. Hook-level tests
live in ``tests/test_hooks.py`` (they need its fixtures).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infra import repositories
from runtime.capabilities import fault_analysis as fault_cap
from runtime.retrieval_live import hybrid_documents, triples_from_expansion


class _StubRetriever:
    """Minimal HybridRetriever shape: retrieve + embeddings + links."""

    def __init__(self, ranked, trace=None):
        self.ranked = ranked
        self.trace = trace or {"graph_missing": False}
        self.embeddings = object()
        self.calls = []

    def _resolved_doc_links(self):
        return {"doc-a": ["O21"], "doc-b": ["M2"]}

    def retrieve(self, question, query_plan=None):
        self.calls.append((question, query_plan))
        return self.ranked, self.trace


_PAYLOADS = {
    "doc-a": {"title": "Motor overheating guide",
              "content": "Check winding temperature."},
    "doc-b": {"title": "Door sensor guide",
              "content": "Clean the sensor gap."},
}


def _patch_factory(monkeypatch, retriever):
    monkeypatch.setattr(
        "runtime.retrieval_live.build_hybrid_retriever",
        lambda: retriever,
    )
    monkeypatch.setattr(
        "runtime.retrieval_live.get_document_payloads",
        lambda doc_ids: {d: _PAYLOADS[d] for d in doc_ids if d in _PAYLOADS},
    )


def test_hybrid_documents_no_plan(monkeypatch):
    stub = _StubRetriever(["doc-a", "doc-b"])
    _patch_factory(monkeypatch, stub)
    result = hybrid_documents("O21 motor overheating")
    assert [d["doc_id"] for d in result["docs"]] == ["doc-a", "doc-b"]
    assert result["docs"][0]["title"] == "Motor overheating guide"
    assert result["docs"][0]["content"] == "Check winding temperature."
    assert result["triples"] == []
    assert result["plan"] is None
    assert result["trace"] == {"graph_missing": False}
    assert stub.calls == [("O21 motor overheating", None)]


def test_hybrid_documents_with_plan(monkeypatch):
    stub = _StubRetriever(["doc-a"])
    _patch_factory(monkeypatch, stub)
    plan = SimpleNamespace(intent="diagnosis")
    monkeypatch.setattr(
        "runtime.retrieval_live.gazetteer_seeds",
        lambda q: [{"type": "Asset", "key": "O21", "confidence": 1.0}],
    )
    monkeypatch.setattr(
        "runtime.retrieval_live.expand_via_graph_query",
        lambda seeds: {
            "nodes": {
                "M2": {"type": "Component", "rel": "HAS_COMPONENT",
                       "from_seed": "O21", "via_hub": False},
                "X9": {"type": "Component", "rel": "RELATED_TO",
                       "from_seed": "O21", "via_hub": False},
            },
            "truncated": False,
            "nodes_returned": 2,
        },
    )
    monkeypatch.setattr(
        "runtime.retrieval_live.build_live_adjacency",
        lambda seeds, expansion: {"O21": {"type": "Asset", "neighbors": []}},
    )
    monkeypatch.setattr(
        "runtime.retrieval_live.build_query_plan",
        lambda *a, **k: plan,
    )
    result = hybrid_documents("O21 motor overheating", query_plan=True)
    assert result["plan"] is plan
    assert stub.calls == [("O21 motor overheating", plan)]
    triples = {(t["subject"], t["relation"], t["object"]): t
               for t in result["triples"]}
    assert triples[("O21", "HAS_COMPONENT", "M2")]["source_doc_id"] == "doc-b"
    # X9 has no linked doc of its own: the triple sources to the
    # seed's doc (doc-a documents O21, the triple's subject).
    assert triples[("O21", "RELATED_TO", "X9")]["source_doc_id"] == "doc-a"


def test_hybrid_documents_failure_propagates(monkeypatch):
    class _FailingRetriever(_StubRetriever):
        def retrieve(self, question, query_plan=None):
            raise RuntimeError("Qdrant is down")

    _patch_factory(monkeypatch, _FailingRetriever([]))
    with pytest.raises(RuntimeError, match="Qdrant is down"):
        hybrid_documents("O21 motor overheating")


def test_triples_from_expansion_sources_to_linked_doc():
    expansion = {
        "nodes": {
            "M2": {"type": "Component", "rel": "HAS_COMPONENT",
                   "from_seed": "O21", "via_hub": False},
        },
        "truncated": False,
        "nodes_returned": 1,
    }
    triples = triples_from_expansion(expansion, {"doc-a": ["O21"]})
    assert triples == [{
        "subject": "O21", "relation": "HAS_COMPONENT", "object": "M2",
        "source_doc_id": "doc-a",
    }]


# --- Seam B: knowledge_retrieval capability node ----------------------


def _hybrid_state():
    return {
        "input": {"asset_id": "O21",
                  "start_date": "2026-10-01", "end_date": "2026-10-31"},
        "mission": {"execution": {"retrieval": {"hybrid": True}}},
    }


def test_knowledge_node_hybrid_on(monkeypatch):
    docs = [
        {"doc_id": "doc-a", "title": "Motor overheating guide",
         "content": "Check winding temperature."},
        {"doc_id": "doc-b", "title": "Door sensor guide",
         "content": "Clean the sensor gap."},
    ]
    monkeypatch.setattr(
        "runtime.retrieval_live.hybrid_documents",
        lambda question: {"docs": docs},
    )
    items = fault_cap.knowledge_retrieval_node(_hybrid_state())["evidence"]
    assert [i["kind"] for i in items] == ["document", "document"]
    assert items[0]["summary"] == "Document: Motor overheating guide"
    assert items[0]["data"] == docs[0]
    assert items[1]["data"] == docs[1]


def test_knowledge_node_hybrid_failure_falls_back_to_vector(monkeypatch):
    def _boom(question):
        raise RuntimeError("Qdrant is down")

    monkeypatch.setattr("runtime.retrieval_live.hybrid_documents", _boom)
    monkeypatch.setattr(
        repositories, "search_documents",
        lambda query, limit=5: (
            [{"doc_id": "doc-v", "title": "Vector doc",
              "content": "plain vector"}], False),
    )
    items = fault_cap.knowledge_retrieval_node(_hybrid_state())["evidence"]
    assert len(items) == 1 and items[0]["kind"] == "document"
    assert items[0]["summary"] == "Document: Vector doc"


def test_knowledge_node_both_arms_fail_is_error_evidence(monkeypatch):
    def _boom(question):
        raise RuntimeError("Qdrant is down")

    monkeypatch.setattr("runtime.retrieval_live.hybrid_documents", _boom)

    def _also_boom(query, limit=5):
        raise RuntimeError("Qdrant is down")

    monkeypatch.setattr(repositories, "search_documents", _also_boom)
    items = fault_cap.knowledge_retrieval_node(_hybrid_state())["evidence"]
    assert len(items) == 1 and items[0]["kind"] == "error"
    assert "Qdrant is down" in items[0]["summary"]


def test_knowledge_node_vector_when_hybrid_off(monkeypatch):
    seen = {}

    def _vector(query, limit=5):
        seen["query"] = query
        seen["limit"] = limit
        return ([{"doc_id": "doc-v", "title": "Vector doc",
                  "content": "plain vector"}], False)

    monkeypatch.setattr(repositories, "search_documents", _vector)
    state = {"input": {"asset_id": "O21",
                       "start_date": "2026-10-01", "end_date": "2026-10-31"}}
    items = fault_cap.knowledge_retrieval_node(state)["evidence"]
    assert seen == {"query": "faults maintenance O21", "limit": 5}
    assert len(items) == 1 and items[0]["kind"] == "document"
    assert items[0]["summary"] == "Document: Vector doc"
