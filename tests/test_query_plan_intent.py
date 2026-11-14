"""Phase 4 intent scoring tests — fusion-style, deterministic, no LLM.

No LiteLLM proxy, no Qdrant/Neo4j server. All data is injected
from ``tests/retrieval_stubs.py``: the canned adjacency (from the
real seed RELATIONS), canned per-asset ground-truth counts, the
real payload link table, the demo-document vocabulary, and
keyword-overlap embeddings that stand in for a real embedding
model. The old fake-corpus winners that depended on the BoW
vector arm are re-derived here on those canned inputs — where a
winner changed, the test says so instead of pinning the old
number. Main features only.
"""

from __future__ import annotations

from pathlib import Path

from runtime.retrieval_graph import gazetteer_seeds
from runtime.retrieval_intent import (
    INTENT_ORDER,
    PROTOTYPES,
    W_EXACT,
    W_GRAPH,
    W_VEC,
    exact_signal,
    graph_plausibility,
    score_intents,
)
from tests.retrieval_stubs import (
    CANNED_ASSET_DATA,
    CANNED_DOC_LINKS,
    KeywordEmbeddings,
    canned_adjacency,
    demo_vocabulary,
)

# Golden-set questions, verbatim (scripts/golden_set_fault_analysis.json).
L02 = ("What should be done when a traction motor overheats (F-204) "
       "with winding temperature above 90C?")
L04 = ("How do you diagnose intermittent CAN bus errors (F-410) "
       "between the controller and the drive?")
L06 = ("How often should traction ropes be inspected, and what defects "
       "should be checked?")
L08 = "What does the door sensor troubleshooting guide say about the sensor gap?"
L09 = ("A safety circuit trip (F-103) happened with no clear trigger. "
       "What usually causes it?")
M03 = ("Elevator O21's brake BK1 lining is below 4mm. Which document covers "
       "the inspection limits, and what must be measured after replacement?")
M04 = ("Elevator O21's controller C1 reports CAN bus errors. Which document "
       "explains the diagnosis, and what other fault code does it mention?")
C01 = ("O21's motor reached 92C (F-204) and O22's motor reached 88.5C (F-206). "
       "Which elevator had the hotter motor, and which document covers overheating?")
C04 = ("Compare the motor overheating remedy for O21 (92C) with O22 (88.5C) "
       "— is the documented remedy the same?")
C05 = ("How does the normal vibration range for healthy elevator O23 compare "
       "with the vibration limit that triggers a fault?")
C06 = ("Which elevator had more faults in October 2026, O21 or O22, "
       "and what were the motor-related ones?")
A01 = "List all the fault codes the technical documents cover for elevator O21."
A04 = "Which documents are linked to elevator O21 in the asset graph?"
U05 = "What was the root cause of the hydraulic leak on elevator O20?"

KW = KeywordEmbeddings()


def _score(question, *, embeddings=KW, **kwargs):
    """score_intents with the canned injected data (overridable)."""
    params = {
        "embeddings": embeddings,
        "adjacency": canned_adjacency(),
        "asset_data": CANNED_ASSET_DATA,
        "doc_links": CANNED_DOC_LINKS,
        "vocabulary": demo_vocabulary(),
    }
    params.update(kwargs)
    return score_intents(question, **params)


def _plausibility(intent, question, seeds=None, **kwargs):
    params = {
        "adjacency": canned_adjacency(),
        "asset_data": CANNED_ASSET_DATA,
        "doc_links": CANNED_DOC_LINKS,
        "vocabulary": demo_vocabulary(),
    }
    params.update(kwargs)
    return graph_plausibility(
        intent, question,
        seeds if seeds is not None else gazetteer_seeds(question),
        params.pop("adjacency"), **params,
    )


# --- Weights / shape ---------------------------------------------------------

def test_weights_are_documented_priors_summing_to_one(demo_gazetteer):
    assert (W_VEC, W_GRAPH, W_EXACT) == (0.50, 0.30, 0.20)
    assert W_VEC + W_GRAPH + W_EXACT == 1.0
    assert _score(L08)["weights"] == {
        "w_vec": 0.50, "w_graph": 0.30, "w_exact": 0.20,
    }


def test_score_breakdown_has_all_intents_and_arms(demo_gazetteer):
    result = _score(L08)
    assert set(result["scores"]) == set(INTENT_ORDER)
    for intent, row in result["scores"].items():
        assert set(row) == {"vec", "z_vec", "graph", "exact", "final"}, intent
        assert 0.0 <= row["graph"] <= 1.0
        assert row["exact"] in (0, 1)
    assert result["prototypes"] == PROTOTYPES
    assert result["seeds"] == gazetteer_seeds(L08)


def test_prototypes_are_short_english_phrases():
    assert set(PROTOTYPES) == set(INTENT_ORDER)
    for intent, text in PROTOTYPES.items():
        assert 1 <= len(text.split()) <= 6, intent


# --- Each intent wins on a representative question ---------------------------

def test_lookup_wins_L08_and_M04(demo_gazetteer):
    assert _score(L08)["intent"] == "lookup"
    assert _score(M04)["intent"] == "lookup"
    assert _score(L06)["intent"] == "lookup"


def test_diagnosis_wins_L04_and_L02(demo_gazetteer):
    assert _score(L04)["intent"] == "diagnosis"
    assert _score(L02)["intent"] == "diagnosis"


def test_diagnosis_wins_L09_without_vector_arm(demo_gazetteer):
    # L09's diagnosis win comes from the exact signal (F-103) once
    # the vector arm is silent: with the keyword embeddings the
    # lookup prototype shares "causes" too and outranks it, so the
    # representative case is pinned here on the graph + exact arms.
    assert _score(L09, embeddings=None)["intent"] == "diagnosis"


def test_procedure_wins_M03(demo_gazetteer):
    result = _score(M03)
    assert result["intent"] == "procedure"
    assert result["margin"] > 1.0  # vector z 2.0 + linked doc, far clear


def test_comparison_wins_C01_C04_C05_C06(demo_gazetteer):
    for question in (C01, C04, C05, C06):
        assert _score(question)["intent"] == "comparison", question


def test_aggregate_wins_A01_and_A04(demo_gazetteer):
    assert _score(A01)["intent"] == "aggregate"
    assert _score(A04)["intent"] == "aggregate"


def test_unanswerable_wins_U05(demo_gazetteer):
    result = _score(U05)
    assert result["intent"] == "unanswerable"
    assert result["margin"] > 0
    # O20 holds nothing behind the link: lookup is only 0.5 plausible.
    assert result["scores"]["lookup"]["graph"] == 0.5
    assert result["scores"]["diagnosis"]["graph"] == 0.0
    assert result["scores"]["unanswerable"]["graph"] == 1.0


# --- O24 / O20: diagnosis scores low ----------------------------------------

def test_diagnosis_plausibility_o21_above_o24_above_o20(demo_gazetteer):
    # Ground truth in seeds/demo_data.py: O21 has 8 fault rows, O22
    # has 2, O23/O24 have none (but 4 sensor readings -> 0.5,
    # graded), O20 has neither faults nor sensors -> 0.0.
    plaus = {
        asset: _plausibility("diagnosis", f"elevator {asset}")
        for asset in ("O21", "O22", "O23", "O24", "O20")
    }
    assert plaus == {"O21": 1.0, "O22": 1.0, "O23": 0.5, "O24": 0.5, "O20": 0.0}


def test_diagnosis_does_not_win_for_o24_or_o20_fault_question(demo_gazetteer):
    q_o24 = "What caused the fault on elevator O24 in October 2026?"
    q_o21 = "What caused the fault on elevator O21 in October 2026?"
    q_o20 = "What caused the fault on elevator O20 in October 2026?"
    assert _score(q_o24, embeddings=None)["intent"] != "diagnosis"
    assert _score(q_o20, embeddings=None)["intent"] != "diagnosis"
    # Same wording, less data behind it: O24/O20 score strictly
    # below O21 on the diagnosis final.
    diag_o24 = _score(q_o24, embeddings=None)["scores"]["diagnosis"]["final"]
    diag_o21 = _score(q_o21, embeddings=None)["scores"]["diagnosis"]["final"]
    diag_o20 = _score(q_o20, embeddings=None)["scores"]["diagnosis"]["final"]
    assert diag_o24 < diag_o21
    assert diag_o20 < diag_o21


# --- Comparison needs >= 2 peer nodes ----------------------------------------

def test_comparison_needs_two_peer_nodes_not_asset_plus_component(demo_gazetteer):
    # A single fault code expands to asset + its component (O21 +
    # motor-M2): one subject, not a comparison. (Fault codes link
    # nothing now, so L02/L04 have no seeds at all — still 0.0.)
    assert _plausibility("comparison", L02) == 0.0
    assert _plausibility("comparison", L04) == 0.0
    # Two assets, or two fault codes, are peer subjects.
    assert _plausibility("comparison", C01) == 1.0
    assert _plausibility("comparison", C04) == 1.0
    assert _plausibility(
        "comparison", "F-410 and F-207 both happened") == 1.0


def test_procedure_needs_a_linked_document(demo_gazetteer):
    assert _plausibility("procedure", "elevator O21") == 1.0
    # O20 (ontology only) and O24 (no docs) have nothing to follow.
    assert _plausibility("procedure", "elevator O20") == 0.0
    assert _plausibility("procedure", "elevator O24") == 0.0


def test_unanswerable_graph_is_absence_not_presence(demo_gazetteer):
    # No seeds and no shared vocabulary word: nothing grounds it.
    assert _plausibility(
        "unanswerable", "What is the warranty period for a hydraulic pump?"
    ) == 1.0
    # No seeds but the words DO overlap the doc vocabulary (L06):
    # answerable from the docs, not unanswerable.
    assert _plausibility("unanswerable", L06) == 0.0
    # Fault-shaped question on a faultless asset (O23's fire alarm).
    assert _plausibility(
        "unanswerable",
        "What caused the fire alarm fault on elevator O23 in October 2026?",
    ) == 1.0


# --- Exact signals -----------------------------------------------------------

def test_exact_signals():
    # A threshold/number boosts lookup and diagnosis, not others.
    assert exact_signal("lookup", "motor reached 92C") == 1
    assert exact_signal("diagnosis", "motor reached 92C") == 1
    assert exact_signal("procedure", "motor reached 92C") == 0
    # A bare year is not a threshold.
    assert exact_signal("lookup", "faults in October 2026") == 0
    # A fault code boosts diagnosis only.
    assert exact_signal("diagnosis", "fault F-204") == 1
    assert exact_signal("lookup", "fault F-204") == 0
    # Comparison words (imported COMPARISON_RE + "more" supplement).
    assert exact_signal("comparison", "which had the hotter motor") == 1
    assert exact_signal("comparison", C06) == 1
    assert exact_signal("comparison", L08) == 0
    # List/count words and plural "documents" boost aggregate.
    assert exact_signal("aggregate", A01) == 1
    assert exact_signal("aggregate", A04) == 1
    assert exact_signal("aggregate", "How many floors does building B2 have?") == 1
    assert exact_signal("aggregate", L08) == 0


# --- Margin, determinism, tie-break ------------------------------------------

def test_margin_is_winner_minus_runner_up(demo_gazetteer):
    for question in (L08, L04, M03, C01, A01, U05):
        result = _score(question)
        finals = sorted(
            (row["final"] for row in result["scores"].values()), reverse=True)
        assert result["margin"] == round(finals[0] - finals[1], 4)
        assert result["margin"] >= 0


def test_deterministic_tie_break_lookup_first(demo_gazetteer):
    # M02 is an all-zero-vector three-way tie (lookup/diagnosis/
    # procedure all 0.30): lookup wins as first in INTENT_ORDER.
    m02 = ("Elevator O21's door sensor DS1 triggers spuriously. Which document "
           "describes the fix, and what is the sensor gap?")
    first = _score(m02)
    second = _score(m02)
    assert first == second
    assert first["intent"] == "lookup"
    assert first["scores"]["lookup"]["final"] == 0.3
    assert first["scores"]["diagnosis"]["final"] == 0.3
    assert first["scores"]["procedure"]["final"] == 0.3
    # L03 is lookup, not a tie, now: no seeds (F-301 links nothing),
    # so lookup's number-boosted 0.35 beats diagnosis's 0.20.
    l03 = ("Brake lining is below 4mm (F-301). What must happen before "
           "the car returns to service?")
    result = _score(l03)
    assert result["intent"] == "lookup"
    assert result["margin"] == 0.15


def test_empty_question_is_unanswerable_with_vocabulary(demo_gazetteer):
    assert _score("")["intent"] == "unanswerable"
    assert _score("   ")["intent"] == "unanswerable"


def test_empty_question_ties_to_lookup_without_vocabulary(demo_gazetteer):
    # No vocabulary injected: the unanswerable absence rule (a) is
    # skipped, every intent scores 0.0, and lookup wins the tie.
    assert _score("", vocabulary=None)["intent"] == "lookup"


# --- Injected embeddings contract --------------------------------------------

def test_no_embeddings_means_zero_vector_arm(demo_gazetteer):
    result = _score(L02, embeddings=None)
    assert all(row["vec"] == 0.0 and row["z_vec"] == 0.0
               for row in result["scores"].values())
    # ...and the winner is decided by graph + exact alone (lookup).
    assert result["intent"] == "lookup"


def test_keyword_embeddings_flip_L02_to_diagnosis(demo_gazetteer):
    # Same question, same injected graph data: the only difference
    # is the vector arm — with it, L02 is a diagnosis question.
    assert _score(L02, embeddings=None)["intent"] == "lookup"
    assert _score(L02, embeddings=KW)["intent"] == "diagnosis"


# --- No-LLM guarantee (structural) -------------------------------------------

def test_no_llm_import_in_module():
    source = Path("runtime/retrieval_intent.py").read_text(encoding="utf-8")
    assert "llm_factory" not in source
    assert "build_chat_model" not in source
    assert "litellm" not in source.lower()
