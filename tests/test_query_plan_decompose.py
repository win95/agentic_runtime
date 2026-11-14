"""Phase 4 decomposition-along-graph-paths tests (main features only).

Fake-only: no LiteLLM proxy, no Qdrant/Neo4j server. The live
generator is never invoked here — the stub generator and injected
callables stand in for it.
"""

from __future__ import annotations

from runtime.retrieval_decompose import (
    decompose_question,
    detect_multi_part,
    relatedness_verdict,
)
from runtime.retrieval_expansion import (
    detect_multi_part as expansion_detect_multi_part,
)
from runtime.retrieval_expansion import stub_decompose_question
from runtime.retrieval_graph import build_demo_adjacency

C01 = (
    "O21's motor reached 92C (F-204) and O22's motor reached 88.5C (F-206). "
    "Which elevator had the hotter motor, and which document covers overheating?"
)
C02 = (
    "Both O21 and O22 had controller communication errors "
    "(F-410 on O21, F-207 on O22). Which document explains CAN bus diagnosis for both?"
)
C03 = (
    "O21 has a door sensor fault (F-101) and O22 has a motor overheating fault (F-206). "
    "Which document covers the door sensor problem?"
)
C04 = (
    "Compare the motor overheating remedy for O21 (92C) with O22 (88.5C) "
    "— is the documented remedy the same?"
)
C05 = (
    "How does the normal vibration range for healthy elevator O23 compare "
    "with the vibration limit that triggers a fault?"
)
C06 = (
    "Which elevator had more faults in October 2026, O21 or O22, "
    "and what were the motor-related ones?"
)
M06 = "Which building is elevator O21 in, and which document covers its motor overheating?"
L02 = (
    "What should be done when a traction motor overheats (F-204) "
    "with winding temperature above 90C?"
)

ADJACENCY = build_demo_adjacency()


def _synthetic_adjacency():
    """Two assets sharing one component node; a third, disconnected."""
    def node(neighbours):
        return {
            "type": "Asset",
            "neighbors": [("HAS_COMPONENT", n, "Component") for n in neighbours],
        }
    return {
        "A": node(["X"]),
        "B": node(["X"]),
        "X": {"type": "Component", "neighbors": []},
        "C": {"type": "Asset", "neighbors": []},
    }


# --- Detection is re-exported, not re-implemented --------------------------

def test_detection_reexported_from_expansion():
    assert detect_multi_part is expansion_detect_multi_part
    assert detect_multi_part(C01)["is_multi_part"] is True
    assert set(detect_multi_part(C01)["reasons"]) >= {"two_assets", "two_fault_codes"}
    assert detect_multi_part(L02) == {"is_multi_part": False, "reasons": []}


# --- Relatedness verdict ---------------------------------------------------

def test_related_verdict_shared_seed_synthetic():
    verdict = relatedness_verdict([["A", "X"], ["B", "X"]], _synthetic_adjacency())
    assert verdict["related"] is True
    assert verdict["min_distance"] == 0
    assert "X" in verdict["shared_nodes"]


def test_related_verdict_shared_neighbour_distance_two_synthetic():
    # A - X - B: seeds 2 hops apart, X within 1 hop of both.
    verdict = relatedness_verdict([["A"], ["B"]], _synthetic_adjacency())
    assert verdict["related"] is True
    assert verdict["min_distance"] == 2
    assert verdict["shared_nodes"] == ["X"]


def test_independent_verdict_o21_vs_o22_shape_on_demo_graph():
    verdict = relatedness_verdict(
        [["O21", "motor-M2"], ["O22", "motor-M3"]], ADJACENCY
    )
    assert verdict["related"] is False
    assert verdict["shared_nodes"] == []
    # No path at all: the B1 and B2 clusters are disconnected.
    assert verdict["min_distance"] is None


def test_related_verdict_none_with_fewer_than_two_seeded_parts():
    assert relatedness_verdict([["O21"], []], ADJACENCY)["related"] is None
    assert relatedness_verdict([[], []], ADJACENCY)["related"] is None
    assert relatedness_verdict([], ADJACENCY)["related"] is None


# --- Per-question decomposition (fake path: stub generator) ----------------

def test_c01_independent_with_per_part_seeds():
    out = decompose_question(C01, decomposition_generator=stub_decompose_question,
                             adjacency=ADJACENCY)
    assert out["is_multi_part"] is True
    assert out["related"] is False
    assert out["shared_nodes"] == []
    assert [s["linked_nodes"] for s in out["sub_questions"]] == [
        ["O21", "motor-M2"],
        ["O22", "motor-M3"],
        [],
    ]


def test_c03_related_via_shared_seed_in_stub_split():
    # Surprising but correct given the stub split: the second clause
    # re-mentions the door sensor, so its seeds overlap part 1's.
    out = decompose_question(C03, decomposition_generator=stub_decompose_question,
                             adjacency=ADJACENCY)
    assert out["is_multi_part"] is True
    assert out["related"] is True
    assert "door-sensor-DS1" in out["shared_nodes"]
    assert out["sub_questions"][0]["linked_nodes"] == ["O21", "door-sensor-DS1"]
    assert "door-sensor-DS1" in out["sub_questions"][1]["linked_nodes"]


def test_stub_splitter_limits_no_decomposition():
    # "Both X and Y" (C02) and "Compare X with Y" (C04/C05): the stub
    # returns the question unchanged — reported, not worked around.
    for question in (C02, C04, C05):
        out = decompose_question(question, decomposition_generator=stub_decompose_question,
                                 adjacency=ADJACENCY)
        assert out["is_multi_part"] is True
        assert out["sub_questions"] == []
        assert out["related"] is None
        assert out["shared_nodes"] == []


def test_c06_seedless_second_part_has_no_verdict():
    out = decompose_question(C06, decomposition_generator=stub_decompose_question,
                             adjacency=ADJACENCY)
    assert out["is_multi_part"] is True
    assert [s["linked_nodes"] for s in out["sub_questions"]] == [["O21", "O22"], []]
    # Only one seeded part: no pair to compare, so no verdict —
    # NOT forced to INDEPENDENT.
    assert out["related"] is None


def test_m06_under_trigger_preserved():
    out = decompose_question(M06, decomposition_generator=stub_decompose_question,
                             adjacency=ADJACENCY)
    assert out["is_multi_part"] is False
    assert out["reasons"] == []
    assert out["related"] is None
    assert out["sub_questions"] == []


def test_single_fact_lookup_not_decomposed():
    out = decompose_question(L02, decomposition_generator=stub_decompose_question,
                             adjacency=ADJACENCY)
    assert out == {
        "is_multi_part": False,
        "reasons": [],
        "related": None,
        "shared_nodes": [],
        "sub_questions": [],
    }


# --- Pluggable generator -----------------------------------------------------

def test_pluggable_generator_accepted_and_seeds_per_part():
    def fake_generator(question):
        return [
            "Which building is elevator O21 in?",
            "Which document covers O21's motor overheating?",
        ]

    out = decompose_question(M06, decomposition_generator=fake_generator,
                             adjacency=ADJACENCY)
    # Detection still gates: M06 is not detected multi-part, so even
    # a generator that could split it is not consulted.
    assert out["is_multi_part"] is False
    assert out["sub_questions"] == []

    out = decompose_question(C01, decomposition_generator=fake_generator,
                             adjacency=ADJACENCY)
    assert out["is_multi_part"] is True
    assert out["sub_questions"] == [
        {"text": "Which building is elevator O21 in?", "linked_nodes": ["O21"]},
        {
            # Bare "motor" is ambiguous (M2/M3/M0) and deliberately
            # NOT gazetteer-mapped, so this part seeds to O21 only.
            "text": "Which document covers O21's motor overheating?",
            "linked_nodes": ["O21"],
        },
    ]
    # Both parts seed to O21: shared seed -> RELATED.
    assert out["related"] is True
    assert "O21" in out["shared_nodes"]


def test_pluggable_generator_returning_question_unchanged():
    out = decompose_question(
        C01, decomposition_generator=lambda q: [q], adjacency=ADJACENCY
    )
    assert out["is_multi_part"] is True
    assert out["sub_questions"] == []
    assert out["related"] is None
