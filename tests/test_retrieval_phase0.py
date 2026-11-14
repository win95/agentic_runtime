"""Phase 0 hybrid retrieval tests — metrics, golden set, harness, baseline.

Fake-only: no LiteLLM proxy, no Qdrant/Neo4j server, no live embedding
model. The MLflow test uses a throwaway local SQLite backend (real
MLflow code, no network).
"""

from __future__ import annotations

import pytest

from runtime.retrieval_baseline import (
    DeterministicFakeEmbeddings,
    FakeQdrantClient,
    baseline_search,
    build_baseline_retriever,
)
from runtime.retrieval_harness import (
    evaluate_retriever,
    load_golden_set,
    log_to_mlflow,
    validate_golden_set,
)
from runtime.retrieval_metrics import (
    evaluate_ranking,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from seeds.demo_data import DOCUMENTS, RELATIONS

# --- Metrics ---------------------------------------------------------------

def test_metrics_perfect_ranking():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, ["a", "b"], 6) == 1.0
    assert ndcg_at_k(ranked, ["a", "b"], 6) == 1.0
    assert reciprocal_rank(ranked, ["a", "b"]) == 1.0


def test_metrics_empty_and_missing():
    assert recall_at_k([], ["a"], 6) == 0.0
    assert ndcg_at_k([], ["a"], 6) == 0.0
    assert reciprocal_rank([], ["a"]) == 0.0
    assert recall_at_k(["x", "y"], ["a"], 6) == 0.0
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0
    # Empty gold set: nothing to recall, defined as 0.0 (harness excludes
    # unanswerable cases from means instead of scoring them).
    assert recall_at_k(["a"], [], 6) == 0.0
    assert ndcg_at_k(["a"], [], 6) == 0.0


def test_metrics_partial_ranking_and_k_beyond_length():
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(ranked, ["a", "b"], 6) == 1.0
    assert recall_at_k(ranked, ["a", "b"], 2) == 0.5
    assert reciprocal_rank(ranked, ["a", "b"]) == 0.5
    # k larger than the ranking: the whole ranking is used.
    assert recall_at_k(ranked, ["a", "b"], 100) == 1.0
    # Rank matters for nDCG: gold at the top beats gold at the bottom.
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 3) > ndcg_at_k(
        ["x", "a", "b"], ["a", "b"], 3
    )
    # Duplicates count once, at first occurrence.
    assert recall_at_k(["a", "a", "a"], ["a"], 6) == 1.0
    assert reciprocal_rank(["a", "a"], ["a"]) == 1.0


def test_evaluate_ranking_keys_in_zero_one():
    metrics = evaluate_ranking(["x", "a"], ["a", "b"])
    assert set(metrics) == {"recall@6", "ndcg@6", "mrr", "recall@20"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


# --- Golden set ------------------------------------------------------------

@pytest.fixture(scope="module")
def golden():
    return load_golden_set()


def test_golden_set_schema_valid(golden):
    assert validate_golden_set(golden) == []
    assert golden["mission"] == "fault_analysis"
    assert golden["version"] == "phase0-v1"


def test_golden_set_size_and_stratification(golden):
    cases = golden["cases"]
    assert 30 <= len(cases) <= 50
    assert len({c["id"] for c in cases}) == len(cases)
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["query_type"]] = counts.get(case["query_type"], 0) + 1
    for query_type in ("lookup", "multi-hop", "aggregate", "comparison"):
        assert counts.get(query_type, 0) >= 4, f"too few {query_type} cases"
    unanswerable_share = counts.get("unanswerable", 0) / len(cases)
    assert 0.15 <= unanswerable_share <= 0.20
    siblings = [c for c in cases if c.get("sibling_confusion")]
    assert len(siblings) >= 4


def test_golden_set_grounded_in_demo_data(golden):
    doc_ids = {d["id"] for d in DOCUMENTS}
    node_keys = {rel[0] for rel in RELATIONS} | {rel[3] for rel in RELATIONS}
    for case in golden["cases"]:
        assert set(case["gold_chunk_ids"]) <= doc_ids, case["id"]
        assert set(case["gold_node_ids"]) <= node_keys, case["id"]
        if case["query_type"] == "unanswerable":
            assert case["gold_chunk_ids"] == []
        else:
            assert case["gold_chunk_ids"], case["id"]


# --- Harness ---------------------------------------------------------------

def _two_cases():
    return [
        {
            "id": "t1", "question": "q1", "expected_answer": "a",
            "must_include_facts": [], "gold_chunk_ids": ["d1"],
            "gold_node_ids": ["O21"], "query_type": "lookup",
        },
        {
            "id": "t2", "question": "q2", "expected_answer": "unanswerable",
            "must_include_facts": [], "gold_chunk_ids": [],
            "gold_node_ids": ["O23"], "query_type": "unanswerable",
        },
    ]


def test_harness_perfect_retriever_scores_one():
    result = evaluate_retriever(lambda q: ["d1", "d2"], _two_cases(),
                                retriever_name="perfect")
    assert result["aggregate"]["recall@6"] == 1.0
    assert result["aggregate"]["ndcg@6"] == 1.0
    assert result["aggregate"]["mrr"] == 1.0
    assert result["aggregate"]["recall@20"] == 1.0
    # Unanswerable case excluded from the means, still counted.
    assert result["aggregate"]["n_answerable"] == 1
    assert result["aggregate"]["n_unanswerable"] == 1


def test_harness_empty_retriever_scores_zero():
    result = evaluate_retriever(lambda q: [], _two_cases(), retriever_name="empty")
    assert result["aggregate"]["recall@6"] == 0.0
    assert result["aggregate"]["ndcg@6"] == 0.0
    assert result["aggregate"]["mrr"] == 0.0
    assert result["aggregate"]["recall@20"] == 0.0


def test_harness_accepts_trace_tuple_and_reports_sibling_subset(golden):
    def stub(question):
        return ["doc-motor-overheat", "doc-door-sensor"], {"latency_seconds": 0.01}

    result = evaluate_retriever(stub, golden["cases"], retriever_name="stub")
    assert result["sibling_confusion"]["n"] >= 4
    assert 0.0 <= result["sibling_confusion"]["recall@6"] <= 1.0
    assert set(result["by_query_type"]) == {
        "lookup", "multi-hop", "aggregate", "comparison", "unanswerable",
    }


def test_harness_mlflow_logs_locally_and_never_crashes(tmp_path):
    result = evaluate_retriever(lambda q: ["d1"], _two_cases(), retriever_name="t")
    run_id = log_to_mlflow(
        result, tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        run_name="phase0-test",
    )
    assert run_id
    # Unreachable server: returns None, does not raise.
    assert log_to_mlflow(result, tracking_uri="http://127.0.0.1:1") is None
    # No tracking URI: skipped cleanly.
    assert log_to_mlflow(result, tracking_uri=None) is None


# --- Baseline (fakes only) -------------------------------------------------

def test_baseline_uses_real_search_path_and_ranks_door_doc_first():
    docs = baseline_search("door sensor faults F-101")
    assert docs[0]["doc_id"] == "doc-door-sensor"
    assert all("doc_id" in d and "score" in d for d in docs)
    assert len(docs) == len(DOCUMENTS)


def test_baseline_deterministic_and_smoke_through_harness(golden):
    retriever = build_baseline_retriever()
    first = retriever("What causes motor overheating?")
    second = retriever("What causes motor overheating?")
    assert first[0] == second[0]  # same ranking on repeat
    assert first[0][0] == "doc-motor-overheat"

    result = evaluate_retriever(retriever, golden["cases"],
                                retriever_name="plain_vector_baseline")
    agg = result["aggregate"]
    assert agg["n_cases"] == len(golden["cases"])
    assert agg["n_answerable"] + agg["n_unanswerable"] == agg["n_cases"]
    for key in ("recall@6", "ndcg@6", "mrr", "recall@20"):
        assert 0.0 <= agg[key] <= 1.0
    # With only 7 docs and a limit-20 pool, Recall@20 saturates;
    # Recall@6 is the discriminating metric.
    assert agg["recall@20"] >= agg["recall@6"]
    assert result["sibling_confusion"]["n"] >= 4


def test_fake_embeddings_share_vocabulary_with_fake_client():
    embeddings = DeterministicFakeEmbeddings()
    client = FakeQdrantClient(embeddings)
    assert embeddings.dimension > 0
    assert client.embeddings is embeddings
    assert len(embeddings.embed_query("motor overheating")) == embeddings.dimension
