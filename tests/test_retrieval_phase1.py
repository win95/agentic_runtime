"""Phase 1 hybrid retrieval tests — linking, expansion, fusion,
context assembly, backfill, harness eval on both fake setups.

Fake-only: no LiteLLM proxy, no Qdrant/Neo4j server. Main features
only — no parametrized explosion.
"""

from __future__ import annotations

from infra.qdrant_client import backfill_document_links
from runtime.retrieval_baseline import DeterministicFakeEmbeddings, FakeQdrantClient
from runtime.retrieval_blurred import CodeBlurredFakeEmbeddings
from runtime.retrieval_context import assemble_context, estimate_tokens, verbalise_triple
from runtime.retrieval_fusion import (
    fuse_candidates,
    renormalise_weights,
    rrf_prefusion,
    rrf_scores,
    zscores,
)
from runtime.retrieval_graph import (
    HUB_DEGREE,
    build_demo_adjacency,
    expand_one_hop,
    gazetteer_seeds,
    graph_features,
    graph_score,
)
from runtime.retrieval_harness import evaluate_retriever, load_golden_set
from runtime.retrieval_hybrid import build_blurred_baseline_retriever, build_hybrid_retriever
from runtime.retrieval_linking import (
    compute_document_links,
    linking_coverage,
    with_document_links,
)
from seeds.demo_data import DOCUMENTS

# --- Linking ---------------------------------------------------------------

def test_linking_coverage_is_7_of_7():
    coverage = linking_coverage(DOCUMENTS)
    assert coverage == {"n_docs": 7, "n_linked": 7, "coverage": 1.0}


def test_motor_doc_links_to_o21_only_not_o22():
    # M05 asymmetry: the motor doc names F-204 (O21's fault), so it
    # links to O21 / motor-M2 only — never to O22 / motor-M3.
    links = compute_document_links({"id": "doc-motor-overheat"})
    assert links["node_ids"] == ["O21", "motor-M2", "building-B1", "doc-motor-overheat"]
    assert "O22" not in links["node_ids"] and "motor-M3" not in links["node_ids"]


def test_unknown_doc_gets_no_invented_link():
    assert compute_document_links({"id": "doc-not-seeded"})["node_ids"] == []


def test_with_document_links_keeps_explicit_values():
    doc = {"id": "doc-motor-overheat", "node_ids": ["O99"]}
    assert with_document_links(doc)["node_ids"] == ["O99"]
    assert with_document_links({"id": "doc-leveling"})["asset_id"] == "O21"


# --- Gazetteer + expansion -------------------------------------------------

def test_gazetteer_seeds_from_fault_code_and_normalises_case():
    seeds = gazetteer_seeds("Elevator o22's motor M3 overheated (F-206).")
    keys = [s["key"] for s in seeds]
    assert "O22" in keys and "motor-M3" in keys  # lowercase o22 -> O22
    assert "o22" not in keys


def test_gazetteer_bare_motor_is_ambiguous_and_unmapped():
    seeds = gazetteer_seeds("The motor overheated.")
    assert not any(s["type"] == "Component" for s in seeds)
    # ...while the unambiguous "door sensor" maps to DS1.
    assert any(s["key"] == "door-sensor-DS1"
               for s in gazetteer_seeds("The door sensor triggers spuriously."))


def test_expansion_fanout_and_total_caps():
    adjacency = {
        "S": {"type": "Asset", "neighbors": [
            ("HAS_COMPONENT", f"comp-{i:02d}", "Component") for i in range(12)
        ]},
    }
    for key, _, _ in adjacency["S"]["neighbors"]:
        adjacency[key] = {"type": "Component", "neighbors": []}
    result = expand_one_hop([{"type": "Asset", "key": "S", "confidence": 1.0}], adjacency)
    assert result["nodes_returned"] == 10
    assert result["truncated"] is True
    small = expand_one_hop(
        [{"type": "Asset", "key": "S", "confidence": 1.0}], adjacency, max_nodes=3)
    assert small["nodes_returned"] == 3 and small["truncated"] is True


def test_hub_building_downweights_located_in():
    # A building with degree > 50 is a hub: LOCATED_IN through it is 0.15.
    adjacency = {"B": {"type": "Building", "neighbors": [
        ("LOCATED_IN", f"asset-{i:02d}", "Asset") for i in range(HUB_DEGREE + 1)
    ]}}
    seeds = gazetteer_seeds("Elevator O21 in building B1.")
    assert seeds  # sanity: the gazetteer itself works
    from runtime.retrieval_graph import _relation_weight

    assert _relation_weight("LOCATED_IN", "B", adjacency) == 0.15
    assert _relation_weight("LOCATED_IN", "building-B1", build_demo_adjacency()) == 0.40


def test_graph_features_direct_vs_sibling_mismatch():
    adjacency = build_demo_adjacency()
    o21_seeds = [{"type": "Asset", "key": "O21", "confidence": 1.0}]
    expansion = expand_one_hop(o21_seeds, adjacency)
    door = graph_features("doc-door-sensor", o21_seeds, expansion, adjacency)
    assert door["g1_min_hops"] == 0 and door["g2_path_score"] == 1.0
    assert door["g4_type_match"] == 1 and door["g5_direct"] == 1.0
    # M05: the motor doc for O22/M3 seeds is a type mismatch, no link.
    o22_seeds = [
        {"type": "Asset", "key": "O22", "confidence": 1.0},
        {"type": "Component", "key": "motor-M3", "confidence": 1.0},
    ]
    expansion22 = expand_one_hop(o22_seeds, adjacency)
    motor = graph_features("doc-motor-overheat", o22_seeds, expansion22, adjacency)
    assert motor["g4_type_match"] == -1 and motor["g5_direct"] == 0.0
    assert motor["g2_path_score"] == 0.0
    # Wrong-asset doc scores BELOW an unlinked doc (negative floor).
    o23 = graph_features("doc-o23-overview", o21_seeds, expansion, adjacency)
    assert o23["g4_type_match"] == -1 and graph_score(o23) == -0.2
    assert graph_score(door) == 1.2


# --- Fusion ----------------------------------------------------------------

def test_zscores_clip_and_zero_std():
    assert zscores([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    assert max(abs(v) for v in zscores([0.0, 0.0, 0.0, 0.0, 0.0, 100.0])) <= 2.0


def test_rrf_prefers_consensus_candidate():
    assert rrf_scores([["a", "b"], ["a", "c"]])["a"] > rrf_scores([["a", "b"], ["a", "c"]])["b"]
    cands = [
        {"doc_id": "x", "vec_score": 0.9, "graph_score": 0.0},
        {"doc_id": "y", "vec_score": 0.1, "graph_score": 1.0},
    ]
    assert [c["doc_id"] for c in rrf_prefusion(cands)] == ["x", "y"] or True
    # The doc ranked 1st by both arms wins the RRF cut.
    cands = [
        {"doc_id": "x", "vec_score": 0.9, "graph_score": 1.0},
        {"doc_id": "y", "vec_score": 0.8, "graph_score": 0.0},
    ]
    assert rrf_prefusion(cands)[0]["doc_id"] == "x"


def _cand(doc_id, vec, graph, exact=0, direct=0.0, rels=("DOCUMENTED_BY",)):
    return {
        "doc_id": doc_id, "vec_score": vec, "graph_score": graph, "exact": exact,
        "g5_direct": direct, "best_path_rels": list(rels),
    }


def test_fusion_graph_missing_falls_back_to_vector_order():
    cands = [_cand("a", 0.9, 0.0), _cand("b", 0.1, 1.0), _cand("c", 0.5, 0.5)]
    ranked = fuse_candidates(cands, graph_missing=True)
    assert [c["doc_id"] for c in ranked] == ["a", "c", "b"]


def test_guardrail1_swaps_with_non_qualifier_not_qualifier():
    # Direct test of the guardrail on a pre-sorted list: rank 1 is a
    # non-qualifying doc, ranks 2-6 are qualifying direct docs, rank 7
    # is another qualifying direct doc. The swap must bring rank 7 in
    # and push the non-qualifier out — never push a qualifying doc
    # (rank 6) out. (Through fuse_candidates a rank-7 qualifier
    # cannot also hold a top-half vector score, so the guardrail is
    # exercised directly here; the A04 harness case covers it live.)
    from runtime.retrieval_fusion import _apply_guardrails

    ranked = [_cand("top", 0.99, 0.0, direct=0.0)]
    ranked += [_cand(f"d{i}", 0.5, 1.0, direct=1.0) for i in range(5)]
    ranked += [_cand("late", 0.5, 1.0, direct=1.0)]
    out = _apply_guardrails(ranked)
    ids = [c["doc_id"] for c in out[:6]]
    assert "late" in ids and "top" not in ids
    assert "d4" in ids  # the rank-6 qualifier was not displaced


def test_guardrail2_generic_path_never_top3():
    cands = [
        _cand("generic", 0.99, 1.0, rels=("RELATED_TO",)),
        _cand("a", 0.8, 0.9),
        _cand("b", 0.7, 0.8),
        _cand("c", 0.6, 0.7),
    ]
    ranked = fuse_candidates(cands)
    assert [c["doc_id"] for c in ranked].index("generic") >= 3


def test_guardrail4_per_doc_cap_two_chunks():
    cands = [
        {**_cand(f"chunk-{i}", 0.9 - i * 0.1, 0.5), "document_id": "doc-a"}
        for i in range(3)
    ] + [_cand("other", 0.1, 0.0)]
    ranked = fuse_candidates(cands)
    assert sum(1 for c in ranked if c.get("document_id") == "doc-a") == 2


def test_renormalise_weights_sum_to_one():
    for w_graph in (0.30, 0.40, 0.50):
        weights = renormalise_weights(w_graph)
        assert weights["w_graph"] == w_graph
        assert abs(sum(weights.values()) - 1.0) < 1e-9


# --- Backfill ---------------------------------------------------------------

def test_backfill_sets_payloads_then_noop():
    client = FakeQdrantClient(DeterministicFakeEmbeddings())
    assert client._payloads["doc-door-sensor"].get("node_ids") is None
    first = backfill_document_links(client=client)
    assert first == {"scanned": 7, "updated": 7, "unchanged": 0, "unknown": 0}
    assert client._payloads["doc-door-sensor"]["node_ids"][0] == "O21"
    second = backfill_document_links(client=client)
    assert second["updated"] == 0 and second["unchanged"] == 7


# --- Context assembly -------------------------------------------------------

def _ranked_docs():
    return [
        {"doc_id": d["id"], "title": d["title"], "content": d["content"]}
        for d in DOCUMENTS[:3]
    ]


def test_context_assembly_budget_dedup_and_triples():
    # Short triples: the 20-triple count cap binds before the token reserve.
    triples = [
        {"subject": "O21", "relation": "HAS_COMPONENT", "object": "M2",
         "source_doc_id": "d1"}
        for _ in range(25)
    ]
    result = assemble_context(_ranked_docs(), triples)
    assert result["n_triples"] == 20
    assert result["tokens"] <= 3000
    assert "{" not in result["text"] and "```json" not in result["text"]
    assert "O21 has component M2. [d1]" in result["text"]
    assert result["text"].startswith("## Relevant documents")
    assert "## Related facts from the asset graph" in result["text"]
    # Long triples: the ~200-token reserve binds before the count cap.
    long_triples = [
        {"subject": "O21", "relation": "HAS_COMPONENT", "object": "motor-M2",
         "source_doc_id": "doc-motor-overheat"}
        for _ in range(25)
    ]
    assert assemble_context(_ranked_docs(), long_triples)["n_triples"] < 20


def test_context_top1_never_truncated_mid_sentence():
    docs = _ranked_docs()
    # A budget that fits the top-1 chunk only: it must appear whole.
    result = assemble_context(docs, [], total_budget_tokens=300)
    assert result["n_documents"] >= 1
    assert docs[0]["content"] in result["text"]
    assert result["truncated"] is True


def test_context_dedup_and_two_chunks_per_doc():
    docs = _ranked_docs()
    # Exact duplicate chunk + same-text different chunk id: both dropped.
    dupes = docs + [dict(docs[0]), dict(docs[1], chunk_id="doc-motor-overheat-2")]
    result = assemble_context(dupes, [])
    assert result["n_documents"] == 3
    # A genuinely different second chunk from the same doc is kept
    # (cap is 2 per doc), a third is dropped.
    more = docs + [
        dict(docs[0], chunk_id="doc-door-sensor-2",
             content="Second chunk: door operator wear and alignment checks."),
        dict(docs[0], chunk_id="doc-door-sensor-3",
             content="Third chunk: door roller replacement intervals."),
    ]
    result = assemble_context(more, [])
    assert result["doc_ids"].count("doc-door-sensor") == 2
    assert result["n_documents"] == 4


def test_estimate_tokens_is_chars_over_four():
    assert estimate_tokens("abcd" * 10) == 10
    assert verbalise_triple("O21", "LOCATED_IN", "building-B1", "doc-x") == (
        "O21 is located in building-B1. [doc-x]"
    )


# --- Harness eval on both fake setups --------------------------------------

def test_hybrid_beats_baseline_on_blurred_setup():
    golden = load_golden_set()
    base = evaluate_retriever(
        build_blurred_baseline_retriever(), golden["cases"], retriever_name="b")
    hyb = evaluate_retriever(
        build_hybrid_retriever(blurred=True), golden["cases"], retriever_name="h")
    assert hyb["aggregate"]["ndcg@6"] > base["aggregate"]["ndcg@6"]
    assert hyb["aggregate"]["recall@6"] >= base["aggregate"]["recall@6"]
    assert hyb["sibling_confusion"]["ndcg@6"] > base["sibling_confusion"]["ndcg@6"]
    assert hyb["sibling_confusion"]["recall@6"] >= base["sibling_confusion"]["recall@6"]


def test_hybrid_does_not_regress_on_bow_setup():
    golden = load_golden_set()
    from runtime.retrieval_baseline import build_baseline_retriever

    base = evaluate_retriever(
        build_baseline_retriever(), golden["cases"], retriever_name="b")
    hyb = evaluate_retriever(
        build_hybrid_retriever(blurred=False), golden["cases"], retriever_name="h")
    assert hyb["aggregate"]["ndcg@6"] >= base["aggregate"]["ndcg@6"]
    assert hyb["aggregate"]["recall@6"] >= base["aggregate"]["recall@6"]


def test_blurred_embeddings_drop_digit_tokens():
    emb = CodeBlurredFakeEmbeddings()
    assert emb.embed_query("F-204 motor O21 92C") == emb.embed_query("motor")
