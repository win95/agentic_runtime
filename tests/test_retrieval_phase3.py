"""Phase 3 hybrid retrieval tests — gated expansion, decomposition
detection, variant merge + rerank, precision@k, LTR feature-row
logging (structure only — NO training code).

No LiteLLM proxy, no Qdrant/Neo4j server. The live generators are
verified with a stub chat model (no LLM call at construction, one
call per generation). Retrieval runs over canned inputs from
``tests/retrieval_stubs.py``; the old harness A/B numbers on the
fake setups are gone with the fakes and are not reproduced here.
Main features only.
"""

from __future__ import annotations

from runtime.retrieval_expansion import (
    EXPANSION_PROMPT,
    LOW_TOP1_VEC,
    SMALL_MARGIN_VEC,
    build_live_variant_generator,
    detect_multi_part,
    gating_decision,
    merge_candidate_rows,
    stub_decompose_question,
    stub_paraphrase_variants,
)
from runtime.retrieval_harness import (
    evaluate_retriever,
    load_golden_set,
    ltr_feature_rows,
    ltr_pair_count,
)
from runtime.retrieval_metrics import precision_at_k
from tests.retrieval_stubs import (
    make_expanded_retriever,
    make_hybrid_retriever,
)

L02 = ("What should be done when a traction motor overheats (F-204) "
       "with winding temperature above 90C?")
L06 = "How often should traction ropes be inspected, and what defects should be checked?"
C01 = ("O21's motor reached 92C (F-204) and O22's motor reached 88.5C (F-206). "
       "Which elevator had the hotter motor, and which document covers overheating?")
C03 = ("O21 has a door sensor fault (F-101) and O22 has a motor overheating fault (F-206). "
       "Which document covers the door sensor problem?")
A01 = "List all the fault codes the technical documents cover for elevator O21."
M06 = "Which building is elevator O21 in, and which document covers its motor overheating?"
# Confident, single-part, seeded: no gating trigger, no decomposition.
CONFIDENT_Q = ("Which document covers the remedy for elevator O21's "
               "motor M2 overheating (F-204)?")

# Confident vector arm: one clear top hit (0.9) with a wide margin.
CONFIDENT_SCORES = {
    "doc-motor-overheat": 0.9,
    "doc-door-sensor": 0.1,
    "doc-brake-inspection": 0.09,
    "doc-controller-errors": 0.08,
    "doc-leveling": 0.07,
    "doc-rope-inspection": 0.06,
    "doc-o23-overview": 0.05,
}
# Flat, weak vector arm (0.3 everywhere): top-1 below the gate and
# zero margin, so gating triggers on every question.
WEAK_SCORES = {
    "doc-door-sensor": 0.3,
    "doc-motor-overheat": 0.3,
    "doc-brake-inspection": 0.3,
    "doc-controller-errors": 0.3,
    "doc-leveling": 0.3,
    "doc-rope-inspection": 0.3,
    "doc-o23-overview": 0.3,
}


def _rows(*vec_scores):
    return [{"doc_id": f"doc-{i}", "vec_score": v} for i, v in enumerate(vec_scores)]


# --- precision@k ------------------------------------------------------------

def test_precision_at_k_binary_relevance():
    assert precision_at_k(["a", "b", "x"], ["a", "b"], 6) == 2 / 6
    assert precision_at_k(["x", "a"], ["a", "b"], 6) == 1 / 6
    assert precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    # Duplicates count once, at first occurrence.
    assert precision_at_k(["a", "a", "a"], ["a"], 6) == 1 / 6


def test_precision_at_k_edges():
    assert precision_at_k([], ["a"], 6) == 0.0
    assert precision_at_k(["a"], [], 6) == 0.0
    assert precision_at_k(["a"], ["a"], 0) == 0.0
    assert precision_at_k(["a"], ["a"], -1) == 0.0
    # k larger than the ranking: denominator is still k.
    assert precision_at_k(["a"], ["a"], 10) == 1 / 10


# --- Gating rule ------------------------------------------------------------

def test_gating_thresholds_are_named_constants():
    assert LOW_TOP1_VEC == 0.35
    assert SMALL_MARGIN_VEC == 0.15


def test_gating_low_top1_triggers():
    decision = gating_decision(L06, _rows(0.21, 0.16, 0.0))
    assert decision["triggered"] and "low_top1" in decision["reasons"]


def test_gating_small_margin_triggers():
    decision = gating_decision(L02, _rows(0.60, 0.55, 0.0))
    assert decision["triggered"] and "small_margin" in decision["reasons"]


def test_gating_gazetteer_miss_triggers():
    # Confident-looking scores, but the question names no code at
    # all — the graph arm has no seeds.
    decision = gating_decision(L06, _rows(0.60, 0.10, 0.0))
    assert decision["triggered"] and "gazetteer_miss" in decision["reasons"]


def test_gating_confident_exact_match_does_not_trigger(demo_gazetteer):
    # Seeded (O21, motor-M2), confident top-1, wide margin: nothing fires.
    decision = gating_decision(CONFIDENT_Q, _rows(0.52, 0.10, 0.0))
    assert not decision["triggered"] and decision["reasons"] == []


def test_gating_end_to_end_on_canned_retriever(demo_gazetteer):
    # L06 (no code) expands even with a confident vector arm
    # (gazetteer miss); the seeded, confident question does not.
    base = make_hybrid_retriever(vector_scores=CONFIDENT_SCORES)
    rows06, _, _, _ = base._candidate_rows(L06)
    rows_conf, _, _, _ = base._candidate_rows(CONFIDENT_Q)
    assert gating_decision(L06, rows06)["triggered"]
    assert not gating_decision(CONFIDENT_Q, rows_conf)["triggered"]


# --- Decomposition detection -------------------------------------------------

def test_decomposition_detects_comparison_and_two_assets():
    assert detect_multi_part(C01)["is_multi_part"]
    assert "two_assets" in detect_multi_part(C01)["reasons"]
    assert "comparison_word" in detect_multi_part(C01)["reasons"]
    assert detect_multi_part(C03)["is_multi_part"]


def test_decomposition_does_not_fire_on_single_hop_lookup():
    assert not detect_multi_part(L02)["is_multi_part"]
    # List-all aggregate: one list request, one retrieval.
    assert not detect_multi_part(A01)["is_multi_part"]


def test_decomposition_detection_limit_is_documented_by_m06():
    # M06 IS two-part (building + document) but names only one code
    # and no comparison word, so the heuristic misses it. This is
    # the documented under-trigger, not a bug to fix by widening
    # the rule (widening would fire on every two-clause lookup).
    assert not detect_multi_part(M06)["is_multi_part"]


def test_stub_decompose_splits_on_and():
    clauses = stub_decompose_question(C03)
    assert len(clauses) >= 2
    assert all("?" in c for c in clauses)
    assert stub_decompose_question(L02) == [L02]


# --- Stub generators ----------------------------------------------------------

def test_stub_paraphrases_are_bounded_and_code_safe():
    variants = stub_paraphrase_variants(C01)
    assert 1 <= len(variants) <= 3
    assert all(v != C01 for v in variants)
    # Codes and numbers are copied verbatim, never invented/dropped
    # by the synonym variant; no "guideed"-style mangling.
    assert "F-204" in variants[0] and "92C" in variants[0]
    assert "guideed" not in " ".join(variants)


# --- Live generator (stub model — no LLM) -------------------------------------

class _StubChatModel:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        return type("Msg", (), {"content": self.reply})()


def test_live_generator_builds_no_model_until_called():
    model = _StubChatModel("1. First paraphrase?\n2. Second paraphrase?\n- Third paraphrase")
    generate = build_live_variant_generator(model=model)
    assert model.calls == []  # construction must not call the LLM
    variants = generate("What causes door faults?")
    assert variants == ["First paraphrase?", "Second paraphrase?", "Third paraphrase"]
    assert len(model.calls) == 1
    assert "do not add facts" in model.calls[0].lower()
    assert "paraphrase only" in model.calls[0].lower()


def test_live_generator_prompt_is_constrained():
    assert "do not add facts" in EXPANSION_PROMPT.lower()
    assert "asset names" in EXPANSION_PROMPT.lower()


def test_live_generator_drops_duplicates_and_original():
    model = _StubChatModel("What causes door faults?\nWhat causes door faults?\n\nNew wording?")
    generate = build_live_variant_generator(model=model)
    assert generate("What causes door faults?") == ["New wording?"]


# --- Merge + rerank through the existing fusion -------------------------------

def test_merge_takes_mean_vec_graph_and_max_exact():
    rows_a = [{"doc_id": "d1", "vec_score": 0.6, "graph_score": 0.4, "exact": 0,
               "g5_direct": 1.0, "best_path_rels": ["DOCUMENTED_BY"], "arms": ["vector"]},
              {"doc_id": "d2", "vec_score": 0.2, "graph_score": 0.1, "exact": 0,
               "g5_direct": 0.0, "best_path_rels": [], "arms": ["vector"]}]
    rows_b = [{"doc_id": "d1", "vec_score": 0.4, "graph_score": 0.8, "exact": 1,
               "g5_direct": 1.0, "best_path_rels": ["DOCUMENTED_BY"], "arms": ["graph"]},
              {"doc_id": "d2", "vec_score": 0.4, "graph_score": 0.3, "exact": 0,
               "g5_direct": 0.0, "best_path_rels": [], "arms": ["vector"]}]
    merged = {r["doc_id"]: r for r in merge_candidate_rows([rows_a, rows_b])}
    assert merged["d1"]["vec_score"] == 0.5  # mean, not max
    assert merged["d1"]["graph_score"] == 0.6
    assert merged["d1"]["exact"] == 1  # max
    # Feature row comes from the highest-graph row (rows_b for d1).
    assert merged["d1"]["arms"] == ["graph"]


def test_expanded_retriever_returns_base_when_nothing_triggers(demo_gazetteer):
    base = make_hybrid_retriever(vector_scores=CONFIDENT_SCORES)
    expanded = make_expanded_retriever(vector_scores=CONFIDENT_SCORES)
    ranked_base, _ = base.retrieve(CONFIDENT_Q)
    ranked_exp, trace = expanded.retrieve(CONFIDENT_Q)
    assert ranked_exp == ranked_base
    assert trace["expansion"]["triggered"] is False
    assert trace["expansion"]["kind"] == "none"


def test_expanded_retriever_merges_variants_and_logs_overlap(demo_gazetteer):
    expanded = make_expanded_retriever(vector_scores=WEAK_SCORES)
    ranked, trace = expanded.retrieve(C01)
    assert trace["expansion"]["triggered"] is True
    assert trace["expansion"]["kind"] in ("decomposition", "multi-query")
    assert 1 <= len(trace["expansion"]["variants"]) <= 3
    overlap = trace["expansion"]["variant_overlap"]
    assert 0.0 <= overlap["mean_top6_jaccard"] <= 1.0
    assert overlap["mean_candidate_jaccard"] == 1.0  # 7-doc corpus
    assert len(ranked) == 6


def test_expanded_retriever_keeps_base_top1_in_top6(demo_gazetteer):
    # Expansion reranks the merged set; it must not lose the base
    # retriever's top hit from the final top-6 on this canned set.
    base = make_hybrid_retriever(vector_scores=WEAK_SCORES)
    expanded = make_expanded_retriever(vector_scores=WEAK_SCORES)
    base_ranked, _ = base.retrieve(C01)
    ranked, _ = expanded.retrieve(C01)
    assert base_ranked[0] in ranked


# --- LTR feature-row logging (structure only — no training) -------------------

def test_ltr_rows_carry_every_section_4_1_field(demo_gazetteer):
    golden = load_golden_set()
    result = evaluate_retriever(
        make_expanded_retriever(vector_scores=WEAK_SCORES), golden["cases"][:3],
        retriever_name="ltr-check")
    rows = ltr_feature_rows(result)
    assert rows  # 3 cases x 7 candidates
    required = {
        "case_id", "query_type", "question", "label", "doc_id",
        "vec_score", "graph_score", "exact",
        "g1_min_hops", "g2_path_score", "g3_best_path_rels",
        "g4_type_match", "g5_direct", "g5_n_seeds_linked",
        "g6_coverage", "g7_ppr", "g8_hub", "g8_transit_key",
        "g9_exact", "bm25_rank", "rrf_rank", "rank_before",
        "rank_after", "final_score", "provenance_variants",
        "expansion_triggered", "expansion_kind", "expansion_variants",
    }
    for row in rows:
        assert required <= set(row), f"missing fields in {row['case_id']}/{row['doc_id']}"
        assert row["label"] in (0, 1)
        assert row["bm25_rank"] is None  # no BM25 — recorded, not invented
    # Labels: the gold doc of L01 is labelled 1.
    l01 = [r for r in rows if r["case_id"] == "L01"]
    assert {r["doc_id"]: r["label"] for r in l01}["doc-door-sensor"] == 1


def test_ltr_pair_count_is_far_below_threshold_and_no_training_exists(demo_gazetteer):
    golden = load_golden_set()
    result = evaluate_retriever(
        make_expanded_retriever(vector_scores=WEAK_SCORES), golden["cases"],
        retriever_name="ltr-count")
    count = ltr_pair_count(result)
    assert count["pairs"] == 36 * 7  # every case x every candidate
    assert count["answerable_pairs"] == 30 * 7
    assert count["threshold"] == 3000
    assert count["meets_threshold"] is False
    # No training code anywhere in the Phase 3 surface.
    import runtime.retrieval_expansion as expansion_mod
    import runtime.retrieval_harness as harness_mod

    assert not hasattr(expansion_mod, "train_ltr")
    assert not hasattr(harness_mod, "train_ltr")
