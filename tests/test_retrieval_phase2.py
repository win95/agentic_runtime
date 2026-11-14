"""Phase 2 hybrid retrieval tests — PPR (G7), coverage (G6), hub (G8),
the locked full graph_score formula, the PPR ablation flag, and the
Phase 1 -> Phase 2 ranking flip on canned inputs.

No LiteLLM proxy, no Qdrant/Neo4j server. All retrieval data is
injected from ``tests/retrieval_stubs.py`` (canned adjacency from
the real seed RELATIONS, canned payloads, stub embeddings/client);
the old harness exit-criterion numbers on the fake setups
(multi-hop nDCG 0.954 -> 1.000 and friends) are gone with the
fakes and are not reproduced here. Main features only — no
parametrized explosion.
"""

from __future__ import annotations

import math

from runtime.retrieval_graph import (
    expand_one_hop,
    gazetteer_seeds,
    graph_features,
    graph_score,
    phase2_graph_scores,
    seed_coverage,
)
from runtime.retrieval_ppr import (
    RESTART_PROB,
    personalized_pagerank,
    seed_specificity,
)
from tests.retrieval_stubs import (
    CANNED_DOC_LINKS,
    canned_adjacency,
    make_hybrid_retriever,
)

M08 = ("Elevator O22's controller C2 has CAN bus errors (F-207). "
       "Which document explains CAN bus diagnosis?")

# Vector arm deliberately ranks the wrong doc first (the hub-transiting
# overview); the graph arm must flip it in Phase 2, not Phase 1.
M08_VECTOR_SCORES = {
    "doc-o23-overview": 0.9,
    "doc-controller-errors": 0.5,
    "doc-motor-overheat": 0.4,
    "doc-door-sensor": 0.3,
    "doc-brake-inspection": 0.2,
    "doc-leveling": 0.1,
    "doc-rope-inspection": 0.05,
}


def _toy_adjacency() -> dict:
    """Two seeds, one node adjacent to BOTH seeds (A), one 1-hop node
    (X), and a 3-hop chain (X - Y - Z) off seed S1.

    Edges are stored both ways (undirected for distance/P PR).
    """
    def _node(node_type, *neighbors):
        return {"type": node_type, "neighbors": list(neighbors)}

    return {
        "S1": _node("Asset",
                    ("HAS_COMPONENT", "A", "Component"),
                    ("HAS_COMPONENT", "X", "Component")),
        "S2": _node("Asset", ("HAS_COMPONENT", "A", "Component")),
        "A": _node("Component",
                   ("HAS_COMPONENT", "S1", "Asset"),
                   ("HAS_COMPONENT", "S2", "Asset")),
        "X": _node("Component",
                   ("HAS_COMPONENT", "S1", "Asset"),
                   ("RELATED_TO", "Y", "Component")),
        "Y": _node("Component",
                   ("RELATED_TO", "X", "Component"),
                   ("RELATED_TO", "Z", "Component")),
        "Z": _node("Component", ("RELATED_TO", "Y", "Component")),
    }


def _toy_seeds():
    return [
        {"type": "Asset", "key": "S1", "confidence": 1.0},
        {"type": "Asset", "key": "S2", "confidence": 1.0},
    ]


# --- PPR -------------------------------------------------------------------

def test_ppr_restart_prob_is_hipporag_convention():
    # restart_prob = 0.5 means follow-edge probability 0.5
    # (HippoRAG damping 0.5) — the parameter is named for the
    # restart side so the convention cannot be silently inverted.
    assert RESTART_PROB == 0.5


def test_ppr_one_hop_node_outranks_three_hop_node():
    # The report's mandated toy-graph assertion (§4.4): two seeds,
    # one 1-hop node, one 3-hop node — the 1-hop node must win
    # before any tuning.
    scores = personalized_pagerank(_toy_adjacency(), _toy_seeds())
    assert scores["A"] > scores["Z"]
    assert scores["A"] > scores["X"]  # adjacent to both seeds beats one
    assert scores["S1"] > scores["A"] and scores["S2"] > scores["A"]
    assert abs(sum(scores.values()) - 1.0) < 1e-6
    assert all(v >= 0.0 for v in scores.values())


def test_ppr_restart_prob_controls_diffusion():
    # The convention is pinned by varying restart_prob, not by the
    # 0.5 default (0.5 is self-dual: restart == follow). High
    # restart keeps mass at the seeds; low restart lets it diffuse
    # to the 3-hop node.
    adj, seeds = _toy_adjacency(), _toy_seeds()
    near = personalized_pagerank(adj, seeds, restart_prob=0.9)
    far = personalized_pagerank(adj, seeds, restart_prob=0.1)
    assert near["S1"] > far["S1"]
    assert far["Z"] > near["Z"]
    assert far["Z"] > 0.0


def test_ppr_no_seeds_is_all_zero():
    assert all(v == 0.0 for v in personalized_pagerank(_toy_adjacency(), []).values())


def test_seed_specificity_rare_seed_outweighs_ubiquitous_seed():
    # R appears in 1 document, H in 6: spec 1.0 vs 1/6, so R's
    # neighbour outranks H's neighbour.
    adjacency = {
        "R": {"type": "Asset", "neighbors": [("RELATED_TO", "A", "Component")]},
        "H": {"type": "Asset", "neighbors": [("RELATED_TO", "B", "Component")]},
        "A": {"type": "Component", "neighbors": [("RELATED_TO", "R", "Asset")]},
        "B": {"type": "Component", "neighbors": [("RELATED_TO", "H", "Asset")]},
    }
    seeds = [
        {"type": "Asset", "key": "R", "confidence": 1.0},
        {"type": "Asset", "key": "H", "confidence": 1.0},
    ]
    spec = seed_specificity(seeds, {"R": 1, "H": 6})
    assert spec == {"R": 1.0, "H": 1.0 / 6}
    assert seed_specificity(seeds, {}) == {"R": 1.0, "H": 1.0}
    weights = {s["key"]: s["confidence"] * spec[s["key"]] for s in seeds}
    scores = personalized_pagerank(adjacency, seeds, seed_weights=weights)
    assert scores["A"] > scores["B"]
    # The canned counts: O21 is in 6 docs, motor-M2 in 1.
    demo_spec = seed_specificity(
        [{"type": "Asset", "key": "O21", "confidence": 1.0},
         {"type": "Component", "key": "motor-M2", "confidence": 1.0}],
        {"O21": 6, "motor-M2": 1},
    )
    assert demo_spec == {"O21": 1.0 / 6, "motor-M2": 1.0}


# --- Coverage (G6) ----------------------------------------------------------

def test_coverage_counts_seeds_within_two_hops():
    adjacency = {
        "D": {"type": "Document", "neighbors": [
            ("DOCUMENTED_BY", "S1", "Asset"), ("DOCUMENTED_BY", "X", "Component")]},
        "S1": {"type": "Asset", "neighbors": [("DOCUMENTED_BY", "D", "Document")]},
        "X": {"type": "Component", "neighbors": [
            ("DOCUMENTED_BY", "D", "Document"), ("RELATED_TO", "S2", "Asset")]},
        "S2": {"type": "Asset", "neighbors": [("RELATED_TO", "X", "Component")]},
        "E": {"type": "Document", "neighbors": [("DOCUMENTED_BY", "X", "Component")]},
    }
    seeds = [
        {"type": "Asset", "key": "S1", "confidence": 1.0},
        {"type": "Asset", "key": "S2", "confidence": 1.0},
    ]
    # D touches S1 directly and S2 via X (2 hops): both seeds.
    assert seed_coverage("D", seeds, adjacency) == 1.0
    # E touches S2 via X (2 hops) but S1 is 3 hops away: one seed.
    assert seed_coverage("E", seeds, adjacency) == 0.5
    assert seed_coverage("D", [], adjacency) == 0.0


# --- Hub (G8) ---------------------------------------------------------------

def test_hub_value_is_building_transit_only(demo_gazetteer):
    # M08: doc-o23-overview's best path transits through building-B2
    # (degree 3 in the canned adjacency) — a Building — so it
    # carries a hub value; the directly seed-linked motor doc for
    # M01 carries none.
    adjacency = canned_adjacency()
    seeds08 = gazetteer_seeds(M08)
    expansion08 = expand_one_hop(seeds08, adjacency)
    o23 = graph_features(
        "doc-o23-overview", seeds08, expansion08, adjacency,
        payload={"node_ids": list(CANNED_DOC_LINKS["doc-o23-overview"])},
    )
    assert o23["g8_transit_key"] == "building-B2"
    assert o23["g8_hub"] == round(math.log1p(3), 4)
    seeds01 = gazetteer_seeds(
        "Elevator O21's traction motor M2 overheated (F-204).")
    expansion01 = expand_one_hop(seeds01, adjacency)
    motor = graph_features(
        "doc-motor-overheat", seeds01, expansion01, adjacency,
        payload={"node_ids": list(CANNED_DOC_LINKS["doc-motor-overheat"])},
    )
    assert motor["g8_transit_key"] is None and motor["g8_hub"] == 0.0


def test_hub_path_scores_lower_than_equal_component_path():
    # Two docs with IDENTICAL Phase 1 scores (same path, same G4/G5)
    # except one's best path transits a Building and the other's a
    # Component: the hubbed one must score lower in Phase 2.
    adjacency = {
        "S": {"type": "Asset", "neighbors": [
            ("HAS_COMPONENT", "C", "Component"), ("PART_OF", "B", "Building")]},
        "C": {"type": "Component", "neighbors": [
            ("HAS_COMPONENT", "S", "Asset"), ("DOCUMENTED_BY", "doc-c", "Document")]},
        "B": {"type": "Building", "neighbors": [
            ("PART_OF", "S", "Asset"), ("DOCUMENTED_BY", "doc-b", "Document")]},
        "doc-c": {"type": "Document", "neighbors": [
            ("DOCUMENTED_BY", "C", "Component")]},
        "doc-b": {"type": "Document", "neighbors": [
            ("DOCUMENTED_BY", "B", "Building")]},
    }
    seeds = [{"type": "Asset", "key": "S", "confidence": 1.0}]
    expansion = expand_one_hop(seeds, adjacency)
    # The synthetic docs are not in DOCUMENT_LINKS, so link them
    # explicitly via payload: doc-c -> C, doc-b -> B.
    feats_c = graph_features("doc-c", seeds, expansion, adjacency,
                             payload={"node_ids": ["C", "doc-c"]})
    feats_b = graph_features("doc-b", seeds, expansion, adjacency,
                             payload={"node_ids": ["B", "doc-b"]})
    assert graph_score(feats_c) == graph_score(feats_b)
    assert feats_c["g8_hub"] == 0.0 and feats_b["g8_hub"] > 0.0
    scores = phase2_graph_scores([feats_c, feats_b])
    assert scores[0] > scores[1]


# --- Phase 2 formula + ablation flag ----------------------------------------

def test_phase2_formula_ppr_term_is_log_minmax_not_zscore():
    # Two rows identical except PPR: with the default weights the
    # high-PPR row wins; with w_ppr=0 they tie exactly (the ablation
    # zeroes the PPR contribution, no renormalising).
    base = {"g1_min_hops": 0, "g2_path_score": 1.0, "g4_type_match": 1,
            "g5_direct": 1.0, "g5_n_seeds_linked": 1, "best_path_rels": ["DOCUMENTED_BY"],
            "g6_coverage": 0.5, "g7_ppr": 0.01, "g8_hub": 0.0, "g8_transit_key": None}
    high = {**base, "g7_ppr": 0.10}
    assert phase2_graph_scores([base, high])[1] > phase2_graph_scores([base, high])[0]
    ablated = phase2_graph_scores([base, high], w_ppr=0.0)
    assert ablated[0] == ablated[1]


def test_m08_fixed_by_phase2_not_phase1(demo_gazetteer):
    # M08, canned inputs: the vector arm ranks the hub-transiting
    # overview first. Phase 2 (hub penalty inside the path term)
    # puts the controller doc first; Phase 1 does not.
    ranked2, _ = make_hybrid_retriever(
        vector_scores=M08_VECTOR_SCORES).retrieve(M08)
    assert ranked2[0] == "doc-controller-errors"
    ranked1, _ = make_hybrid_retriever(
        vector_scores=M08_VECTOR_SCORES, phase2=False).retrieve(M08)
    assert ranked1[0] == "doc-o23-overview"
