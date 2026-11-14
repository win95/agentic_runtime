"""Phase 4 QueryPlan integration tests — plan build, guardrails, seam.

No LiteLLM proxy, no Qdrant/Neo4j server. The stub generators
stand in for the live ones everywhere; retrieval runs over canned
inputs from ``tests/retrieval_stubs.py`` (canned adjacency,
canned per-asset data, keyword-overlap embeddings). The old
fake-corpus eval measurements (link recall 0.7616, intent
accuracy 18/36, nDCG 0.958) are gone with the fakes and are not
reproduced here. Main features only.
"""

from __future__ import annotations

from pathlib import Path

from runtime.retrieval_expansion import (
    stub_decompose_question,
    stub_paraphrase_variants,
)
from runtime.retrieval_graph import gazetteer_seeds
from runtime.retrieval_harness import load_golden_set
from runtime.retrieval_intent import INTENT_ORDER
from runtime.retrieval_query_plan import (
    FALLBACK_CONFIDENT,
    FALLBACK_NO_LINK,
    QueryPlan,
    build_query_plan,
    check_rewrite_variants,
)
from tests.retrieval_stubs import (
    CANNED_ASSET_DATA,
    CANNED_DOC_LINKS,
    KeywordEmbeddings,
    canned_adjacency,
    demo_vocabulary,
    make_hybrid_retriever,
)

L06 = ("How often should traction ropes be inspected, and what defects "
       "should be checked?")
M04 = ("Elevator O21's controller C1 reports CAN bus errors. Which document "
       "explains the diagnosis, and what other fault code does it mention?")
C01 = ("O21's motor reached 92C (F-204) and O22's motor reached 88.5C (F-206). "
       "Which elevator had the hotter motor, and which document covers overheating?")
M01 = ("Elevator O21's traction motor M2 overheated (F-204). Which document covers "
       "the remedy, and what component is involved?")
U01 = "What is the warranty period for elevator O21's traction motor M2?"

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
# Flat, weak vector arm (0.3 everywhere): gating triggers on every question.
WEAK_SCORES = {
    "doc-door-sensor": 0.3,
    "doc-motor-overheat": 0.3,
    "doc-brake-inspection": 0.3,
    "doc-controller-errors": 0.3,
    "doc-leveling": 0.3,
    "doc-rope-inspection": 0.3,
    "doc-o23-overview": 0.3,
}


def _plan(question, retriever=None, **kwargs):
    return build_query_plan(
        question,
        retriever=retriever,
        embeddings=KeywordEmbeddings(),
        adjacency=canned_adjacency(),
        asset_data=CANNED_ASSET_DATA,
        doc_links=CANNED_DOC_LINKS,
        vocabulary=demo_vocabulary(),
        paraphrase_generator=stub_paraphrase_variants,
        decomposition_generator=stub_decompose_question,
        **kwargs,
    )


# --- QueryPlan build ---------------------------------------------------------

def test_gated_off_plan_keeps_intent_and_links_but_no_variants(demo_gazetteer):
    retriever = make_hybrid_retriever(vector_scores=CONFIDENT_SCORES)
    plan = _plan(M01, retriever)
    assert plan.intent == "diagnosis"
    assert plan.linked_nodes == [
        {"type": "Asset", "key": "O21", "confidence": 1.0},
        {"type": "Component", "key": "motor-M2", "confidence": 0.8},
    ]
    assert plan.gated_off is True
    assert plan.fallback_reason == FALLBACK_CONFIDENT
    assert plan.rewrite_variants == []
    assert plan.sub_questions == []
    finals = sorted(
        (row["final"] for row in plan.intent_scores.values()), reverse=True)
    assert plan.confidence == round(finals[0] - finals[1], 4)
    assert plan.gating["triggered"] is False


def test_gated_on_decomposition_plan_carries_sub_questions(demo_gazetteer):
    retriever = make_hybrid_retriever(vector_scores=WEAK_SCORES)
    plan = _plan(C01, retriever)
    assert plan.gated_off is False
    assert plan.fallback_reason is None
    assert plan.intent == "comparison"
    assert [s["text"] for s in plan.sub_questions] == [
        "O21's motor reached 92C (F-204)?",
        "O22's motor reached 88.5C (F-206). Which elevator had the hotter motor?",
        "which document covers overheating?",
    ]
    assert plan.related is False
    assert plan.shared_nodes == []
    # Decomposition takes precedence: no paraphrase variants.
    assert plan.rewrite_variants == []


def test_gated_on_single_part_plan_carries_kept_variants(demo_gazetteer):
    retriever = make_hybrid_retriever(vector_scores=WEAK_SCORES)
    plan = _plan(M04, retriever)
    assert plan.gated_off is False
    assert plan.sub_questions == []
    assert plan.rewrite_variants == [
        "Elevator O21's controller C1 reports CAN bus errors. Which guide "
        "explains the diagnosis, and what other failure code does it mention?"
    ]
    assert plan.fallback_reason is None


def test_no_link_plan_says_so_and_carries_no_variants(demo_gazetteer):
    retriever = make_hybrid_retriever(vector_scores=WEAK_SCORES)
    plan = _plan(L06, retriever)
    assert plan.linked_nodes == []
    assert plan.gated_off is False  # gazetteer miss opens the gate
    assert plan.fallback_reason == FALLBACK_NO_LINK
    assert plan.rewrite_variants == []


def test_no_retriever_gating_falls_back_to_gazetteer_miss_only(demo_gazetteer):
    seeded = _plan(U01)  # seeds present, no retriever -> gated off
    assert seeded.gated_off is True
    assert seeded.gating["reasons"] == []
    assert seeded.gating["top1_vec"] is None
    assert seeded.fallback_reason == FALLBACK_CONFIDENT
    seedless = _plan(L06)  # no seeds, no retriever -> gate opens
    assert seedless.gated_off is False
    assert seedless.gating["reasons"] == ["gazetteer_miss"]
    assert seedless.fallback_reason == FALLBACK_NO_LINK


def test_confident_multi_part_is_gated_off_unlike_phase3(demo_gazetteer):
    # Phase 3 decomposed C02/C03/C05/M07 regardless of confidence;
    # Phase 4's single gate leaves them untouched (no LLM call).
    retriever = make_hybrid_retriever(vector_scores=CONFIDENT_SCORES)
    c02 = ("Both O21 and O22 had controller communication errors "
           "(F-410 on O21, F-207 on O22). Which document explains CAN bus "
           "diagnosis for both?")
    plan = _plan(c02, retriever)
    assert plan.is_multi_part is True
    assert plan.gated_off is True
    assert plan.sub_questions == []
    assert plan.intent == "diagnosis"  # not relabelled "lookup"


# --- Rewrite guardrails -------------------------------------------------------

def test_guardrail_keeps_faithful_variant(demo_gazetteer):
    question = ("Elevator O21's traction motor M2 overheated (F-204). "
                "Which document covers the remedy?")
    kept, dropped, reasons = check_rewrite_variants(
        question,
        ["What guide describes the fix for the overheated traction motor M2 "
         "on elevator O21 (F-204)?"],
        linked_nodes=gazetteer_seeds(question),
        adjacency=canned_adjacency(),
    )
    assert len(kept) == 1 and dropped == [] and reasons == {}


def test_guardrail_drops_variant_that_drops_a_code(demo_gazetteer):
    question = ("Elevator O21's traction motor M2 overheated (F-204). "
                "Which document covers the remedy?")
    kept, dropped, reasons = check_rewrite_variants(
        question,
        ["Elevator O21's traction motor M2 overheated. "
         "Which document covers the remedy?"],
        linked_nodes=gazetteer_seeds(question),
        adjacency=canned_adjacency(),
    )
    assert kept == []
    assert reasons[dropped[0]] == ["missing_fault_codes"]


def test_guardrail_drops_variant_with_a_foreign_entity(demo_gazetteer):
    question = ("Elevator O21's traction motor M2 overheated (F-204). "
                "Which document covers the remedy?")
    kept, dropped, reasons = check_rewrite_variants(
        question,
        ["Elevator O22's motor M3 overheated (F-206). "
         "Which document covers the remedy?"],
        linked_nodes=gazetteer_seeds(question),
        adjacency=canned_adjacency(),
    )
    assert kept == []
    assert "drift" in reasons[dropped[0]]
    assert "missing_entities" in reasons[dropped[0]]


def test_guardrail_drops_variant_inventing_a_threshold(demo_gazetteer):
    question = ("Elevator O21's traction motor M2 overheated (F-204). "
                "Which document covers the remedy?")
    kept, dropped, reasons = check_rewrite_variants(
        question,
        ["What guide describes the fix for the overheated traction motor M2 "
         "on elevator O21 (F-204) at 92C?"],
        linked_nodes=gazetteer_seeds(question),
        adjacency=canned_adjacency(),
    )
    assert kept == []
    assert reasons[dropped[0]] == ["drift"]


def test_guardrail_drops_all_variants_when_no_seeds(demo_gazetteer):
    kept, dropped, reasons = check_rewrite_variants(
        L06,
        ["How often should traction ropes be inspected?",
         "What defects should be checked on traction ropes?"],
        linked_nodes=[],
        adjacency=canned_adjacency(),
    )
    assert kept == []
    assert len(dropped) == 2
    assert all(r == ["no_seeds"] for r in reasons.values())


def test_guardrail_caps_at_three_variants(demo_gazetteer):
    question = ("Elevator O21's traction motor M2 overheated (F-204). "
                "Which document covers the remedy?")
    variants = [
        "What guide describes the fix for the overheated traction motor M2 "
        "on elevator O21 (F-204)?",
        "Which manual covers the remedy when the traction motor M2 on O21 "
        "overheats (F-204)?",
        "For elevator O21, where is the remedy documented for traction "
        "motor M2 overheating (F-204)?",
        "What document covers the remedy for traction motor M2 overheating "
        "on elevator O21 (F-204)?",
    ]
    kept, dropped, reasons = check_rewrite_variants(
        question, variants, linked_nodes=gazetteer_seeds(question),
        adjacency=canned_adjacency(),
    )
    assert len(kept) == 3
    assert dropped == [variants[3]]
    assert reasons[variants[3]] == ["over_cap"]


def test_guardrail_drops_variant_dropping_a_threshold_number(demo_gazetteer):
    # Seeded question (O21, motor-M2) with a number-with-unit: the
    # variant drops "92C", so the preservation floor fires. (The old
    # F-301 wording has no seeds any more — fault codes link
    # nothing — so it would trip the no-seeds rule instead.)
    question = ("Elevator O21's motor M2 reached 92C (F-204). "
                "What must happen before the car returns to service?")
    kept, dropped, reasons = check_rewrite_variants(
        question,
        ["Elevator O21's motor M2 overheated (F-204). "
         "What must happen before the car returns?"],
        linked_nodes=gazetteer_seeds(question),
        adjacency=canned_adjacency(),
    )
    assert kept == []
    assert "missing_numbers" in reasons[dropped[0]]


# --- Retriever seam -------------------------------------------------------------

def test_no_plan_path_is_byte_identical(demo_gazetteer):
    retriever = make_hybrid_retriever(vector_scores=CONFIDENT_SCORES)
    expected = retriever.retrieve(M01)
    assert retriever.retrieve(M01, query_plan=None) == expected
    assert make_hybrid_retriever(
        vector_scores=CONFIDENT_SCORES).retrieve(M01) == expected


def test_with_plan_uses_plan_seeds(demo_gazetteer):
    retriever = make_hybrid_retriever(vector_scores=WEAK_SCORES)
    plan = QueryPlan(
        intent="lookup",
        linked_nodes=[{"type": "Asset", "key": "O22", "confidence": 1.0}],
        gated_off=True,
        fallback_reason=FALLBACK_CONFIDENT,
    )
    ranked, trace = retriever.retrieve(M01, query_plan=plan)
    assert trace["seeds"] == ["O22"]
    assert trace["query_plan"]["linked_nodes"] == ["O22"]
    assert trace["query_plan"]["gated_off"] is True
    # Same question, no plan: seeds come from the question (O21/M2).
    assert retriever.retrieve(M01)[1]["seeds"] == ["O21", "motor-M2"]
    assert ranked  # a ranking is still returned


def test_with_plan_gated_off_ranking_matches_no_plan(demo_gazetteer):
    retriever = make_hybrid_retriever(vector_scores=CONFIDENT_SCORES)
    plan = _plan(M01, retriever)
    assert plan.gated_off is True
    assert retriever.retrieve(M01, query_plan=plan)[0] == retriever.retrieve(M01)[0]


def test_with_plan_decomposition_expands_via_merge_path(demo_gazetteer):
    retriever = make_hybrid_retriever(vector_scores=WEAK_SCORES)
    plan = _plan(C01, retriever)
    ranked, trace = retriever.retrieve(C01, query_plan=plan)
    assert trace["expansion"]["triggered"] is True
    assert trace["expansion"]["kind"] == "decomposition"
    assert trace["expansion"]["variants"] == [s["text"] for s in plan.sub_questions]
    assert ranked[0] == "doc-motor-overheat"


# --- Golden-set intent labels ---------------------------------------------------

def test_golden_set_intent_labels_present_valid_and_deliberate():
    golden = load_golden_set()
    cases = golden["cases"]
    assert len(cases) == 36
    for case in cases:
        assert case["intent"] in INTENT_ORDER, case["id"]
        assert case["intent"] != "unanswerable", (
            f"{case['id']}: unanswerable cases are labelled by their ask, "
            "never 'unanswerable'"
        )
    # Labels are NOT a copy of query_type: multi-hop cases carry
    # lookup/diagnosis/procedure, and lookup cases carry diagnosis
    # and procedure too.
    by_type = {c["id"]: c["intent"] for c in cases}
    assert by_type["L02"] == "procedure"
    assert by_type["L01"] == "diagnosis"
    assert by_type["M03"] == "procedure"
    assert by_type["M07"] == "diagnosis"
    assert by_type["U05"] == "diagnosis"
    assert by_type["U06"] == "procedure"
    assert by_type["C01"] == "comparison"
    assert by_type["A01"] == "aggregate"


def test_entity_link_labels_are_entity_only():
    golden = load_golden_set()
    for case in golden["cases"]:
        entity = [n for n in case["gold_node_ids"] if not n.startswith("doc-")]
        assert entity, case["id"]  # every case has >= 1 entity gold node
        assert all(not n.startswith("doc-") for n in entity)


# --- HyDE exclusion (structural) -------------------------------------------------

def test_no_hyde_and_no_llm_import_in_query_plan_module():
    source = Path("runtime/retrieval_query_plan.py").read_text(encoding="utf-8")
    # "HyDE" appears exactly once: the docstring sentence excluding it.
    # There is no HyDE generator, hypothetical-document builder, or
    # LLM import anywhere in the module.
    assert source.lower().count("hyde") == 1
    assert "HyDE is EXPLICITLY EXCLUDED" in source
    assert "generate_hypothetical" not in source
    assert "llm_factory" not in source
    assert "build_chat_model" not in source
