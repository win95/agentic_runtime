"""Phase 4 intent scoring tests — fusion-style, deterministic, no LLM.

Fake-only: no LiteLLM proxy, no Qdrant/Neo4j server. Questions are
copied verbatim from scripts/golden_set_fault_analysis.json where
they fit; the intent taxonomy has no multi-hop label, so multi-hop
golden cases map to lookup (their document-seeking shape), while
diagnosis/procedure representatives are golden lookup/multi-hop
questions whose wording is diagnosis/procedure-shaped. Main
features only.
"""

from __future__ import annotations

from pathlib import Path

from runtime.retrieval_blurred import CodeBlurredFakeEmbeddings
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


# --- Weights / shape ---------------------------------------------------------

def test_weights_are_documented_priors_summing_to_one():
    assert (W_VEC, W_GRAPH, W_EXACT) == (0.50, 0.30, 0.20)
    assert W_VEC + W_GRAPH + W_EXACT == 1.0
    assert score_intents(L08)["weights"] == {
        "w_vec": 0.50, "w_graph": 0.30, "w_exact": 0.20,
    }


def test_score_breakdown_has_all_intents_and_arms():
    result = score_intents(L08)
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

def test_lookup_wins_L08_and_M04():
    assert score_intents(L08)["intent"] == "lookup"
    assert score_intents(M04)["intent"] == "lookup"
    assert score_intents(L06)["intent"] == "lookup"


def test_diagnosis_wins_L04_L09_and_L02():
    assert score_intents(L04)["intent"] == "diagnosis"
    assert score_intents(L09)["intent"] == "diagnosis"
    assert score_intents(L02)["intent"] == "diagnosis"


def test_procedure_wins_M03():
    result = score_intents(M03)
    assert result["intent"] == "procedure"
    assert result["margin"] > 1.0  # vector z 2.0 + linked doc, far clear


def test_comparison_wins_C01_C04_C05_C06():
    for question in (C01, C04, C05, C06):
        assert score_intents(question)["intent"] == "comparison", question


def test_aggregate_wins_A01_and_A04():
    assert score_intents(A01)["intent"] == "aggregate"
    assert score_intents(A04)["intent"] == "aggregate"


def test_unanswerable_wins_U05():
    result = score_intents(U05)
    assert result["intent"] == "unanswerable"
    assert result["margin"] > 0
    # O20 holds nothing behind the link: lookup is only 0.5 plausible.
    assert result["scores"]["lookup"]["graph"] == 0.5
    assert result["scores"]["diagnosis"]["graph"] == 0.0
    assert result["scores"]["unanswerable"]["graph"] == 1.0


# --- O24 / O20: diagnosis scores low ----------------------------------------

def test_diagnosis_plausibility_o21_above_o24_above_o20():
    # Ground truth in seeds/demo_data.py: O21 has 8 fault rows, O24
    # has none (but 4 sensor readings -> 0.5, graded), O20 has
    # neither faults nor sensors -> 0.0.
    plaus = {
        asset: graph_plausibility("diagnosis", f"elevator {asset}",
                                  gazetteer_seeds(f"elevator {asset}"))
        for asset in ("O21", "O22", "O23", "O24", "O20")
    }
    assert plaus == {"O21": 1.0, "O22": 1.0, "O23": 0.5, "O24": 0.5, "O20": 0.0}


def test_diagnosis_does_not_win_for_o24_or_o20_fault_question():
    q_o24 = "What caused the fault on elevator O24 in October 2026?"
    q_o21 = "What caused the fault on elevator O21 in October 2026?"
    q_o20 = "What caused the fault on elevator O20 in October 2026?"
    assert score_intents(q_o24)["intent"] != "diagnosis"
    assert score_intents(q_o20)["intent"] != "diagnosis"
    # Same wording, less data behind it: O24/O20 score strictly
    # below O21 on the diagnosis final.
    diag_o24 = score_intents(q_o24)["scores"]["diagnosis"]["final"]
    diag_o21 = score_intents(q_o21)["scores"]["diagnosis"]["final"]
    diag_o20 = score_intents(q_o20)["scores"]["diagnosis"]["final"]
    assert diag_o24 < diag_o21
    assert diag_o20 < diag_o21


# --- Comparison needs >= 2 peer nodes ----------------------------------------

def test_comparison_needs_two_peer_nodes_not_asset_plus_component():
    # A single fault code expands to asset + its component (O21 +
    # motor-M2): one subject, not a comparison.
    assert graph_plausibility("comparison", L02) == 0.0
    assert graph_plausibility("comparison", L04) == 0.0
    # Two assets, or two fault codes, are peer subjects.
    assert graph_plausibility("comparison", C01) == 1.0
    assert graph_plausibility("comparison", C04) == 1.0
    assert graph_plausibility(
        "comparison", "F-410 and F-207 both happened") == 1.0


def test_procedure_needs_a_linked_document():
    assert graph_plausibility("procedure", "elevator O21",
                              gazetteer_seeds("elevator O21")) == 1.0
    # O20 (ontology only) and O24 (no docs) have nothing to follow.
    assert graph_plausibility("procedure", "elevator O20",
                              gazetteer_seeds("elevator O20")) == 0.0
    assert graph_plausibility("procedure", "elevator O24",
                              gazetteer_seeds("elevator O24")) == 0.0


def test_unanswerable_graph_is_absence_not_presence():
    # No seeds and no shared vocabulary word: nothing grounds it.
    assert graph_plausibility(
        "unanswerable", "What is the warranty period for a hydraulic pump?"
    ) == 1.0
    # No seeds but the words DO overlap the doc vocabulary (L06):
    # answerable from the docs, not unanswerable.
    assert graph_plausibility("unanswerable", L06) == 0.0
    # Fault-shaped question on a faultless asset (O23's fire alarm).
    assert graph_plausibility(
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

def test_margin_is_winner_minus_runner_up():
    for question in (L08, L04, M03, C01, A01, U05):
        result = score_intents(question)
        finals = sorted(
            (row["final"] for row in result["scores"].values()), reverse=True)
        assert result["margin"] == round(finals[0] - finals[1], 4)
        assert result["margin"] >= 0


def test_deterministic_tie_break_lookup_first():
    # M02 is an all-zero-vector three-way tie (lookup/diagnosis/
    # procedure all 0.30): lookup wins as first in INTENT_ORDER.
    m02 = ("Elevator O21's door sensor DS1 triggers spuriously. Which document "
           "describes the fix, and what is the sensor gap?")
    first = score_intents(m02)
    second = score_intents(m02)
    assert first == second
    assert first["intent"] == "lookup"
    assert first["scores"]["lookup"]["final"] == 0.3
    assert first["scores"]["diagnosis"]["final"] == 0.3
    assert first["scores"]["procedure"]["final"] == 0.3
    # L03 ties lookup/diagnosis at 0.50 (margin 0.0) — same rule.
    l03 = ("Brake lining is below 4mm (F-301). What must happen before "
           "the car returns to service?")
    result = score_intents(l03)
    assert result["intent"] == "lookup"
    assert result["margin"] == 0.0


def test_empty_question_is_unanswerable():
    assert score_intents("")["intent"] == "unanswerable"
    assert score_intents("   ")["intent"] == "unanswerable"


# --- Code-blurred embeddings -------------------------------------------------

def test_blurred_embeddings_keep_representative_winners():
    blurred = CodeBlurredFakeEmbeddings()
    for question, expected in (
        (L04, "diagnosis"),
        (M03, "procedure"),
        (C04, "comparison"),
        (A01, "aggregate"),
        (U05, "unanswerable"),
        (L08, "lookup"),
    ):
        assert score_intents(question, embeddings=blurred)["intent"] == expected


# --- No-LLM guarantee (structural) -------------------------------------------

def test_no_llm_import_in_module():
    source = Path("runtime/retrieval_intent.py").read_text(encoding="utf-8")
    assert "llm_factory" not in source
    assert "build_chat_model" not in source
    assert "litellm" not in source.lower()
