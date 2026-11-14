"""Phase 4 — decomposition along graph paths (Query Understanding).

Pilot mission: ``fault_analysis`` only. No live LLM, Qdrant, or
Neo4j is built here; the live decomposition generator is lazy
(no model is built until a gated query actually needs one).

This module answers ONE question about a multi-part question:
are its parts about the SAME neighbourhood of the graph (answer
them jointly) or about DISCONNECTED neighbourhoods (answer them
independently)? It does NOT decide whether a question is
multi-part — that is ``detect_multi_part`` from
``runtime/retrieval_expansion.py`` (rules (a)–(d)), imported here,
never copied — and it does NOT split questions itself: sub-questions
come only from the EXISTING pluggable generators
(``stub_decompose_question`` for the deterministic stub path,
``build_live_decomposition_generator`` by default).

Relatedness rule (exact, as built)
----------------------------------
For a detected multi-part question, split it with the decomposition
generator, then take the gazetteer seeds of each sub-question
(``runtime/retrieval_graph.gazetteer_seeds`` per part). Let S_i be
the seed set of part i (seed keys only). Only parts with a non-empty
seed set count as *seeded* parts.

* Two seeded parts are RELATED iff the BFS distance between their
  seed sets on the UNDIRECTED adjacency supplied by the caller
  (the live graph adjacency — see ``relatedness_verdict``) is
  <= 2 — i.e. the parts share a node within 1 hop of both parts'
  seeds — or they share a seed outright (distance 0, a special
  case of the same rule).
* ``shared_nodes`` is the intersection of the parts' closed 1-hop
  neighbourhoods (each part's seeds UNION their 1-hop neighbours),
  sorted, for the related pair(s). It therefore names the shared
  seed itself when the parts share one, and the common neighbour
  (e.g. a shared document or building) when the seeds are 2 hops
  apart. A distance-1 pair (adjacent seeds) names both seeds.
* The question-level verdict is RELATED iff ANY seeded pair is
  related; INDEPENDENT iff there are >= 2 seeded parts and NO pair
  is related. ``min_distance`` is the smallest BFS distance over
  seeded pairs (``None`` when no path exists between any pair).

Design decisions that differ from the brief (deliberate, not tuning)
--------------------------------------------------------------------
1. ``related`` is ``bool | None``, not always a bool. It is ``None``
   in three cases where no verdict exists, rather than forcing one:
   (a) the question is not detected multi-part (M06 — see below);
   (b) the generator returns the question unchanged ("no
   decomposition" — C02, C04, C05 in the stub path), so there are
   no parts to compare; (c) fewer than two parts carry seeds
   (C06: the second stub part names no code). Forcing
   INDEPENDENT in case (c) would be wrong: a seedless part is
   unlocated, not disconnected.
2. With more than two parts (the stub splits C01 into three), the
   verdict aggregates over pairs with ANY-related, not ALL-related:
   one jointly-answerable pair is enough to route jointly, and
   seedless parts are excluded from the pairs rather than vetoing.
   ``shared_nodes`` is the union over the related pairs.

Known limits, preserved (not fixed here)
-----------------------------------------
* M06 ("Which building is O21 in, and which document covers its
  motor overheating?") is NOT detected by ``detect_multi_part``
  (one code, no comparison word), so this module reports
  ``is_multi_part=False`` and ``related=None`` — the known
  under-trigger from the Phase 3 doc. Hand-partitioned, its two
  facts would be RELATED (both seed to O21), but the detection
  rule is NOT widened to catch it; a learned router (or the LLM
  itself judging multi-part) is the live follow-up.
* The stub splitter cannot split "Both X and Y" (C02) or
  "Compare X with Y" (C04/C05) wordings — those report "no
  decomposition" (``sub_questions=[]``, ``related=None``) in the
  stub path. Do not write a smarter splitter here; the live
  generator is the path that splits them.
* The stub splitter is naive about clause boundaries: in C03 its
  second clause re-mentions the door sensor, so that part's seeds
  overlap the first part's and the rule (correctly, given those
  seeds) returns RELATED via the shared seed ``door-sensor-DS1``
  — even though the O21-vs-O22 assets themselves are
  disconnected. The verdict is about the parts AS SPLIT, not
  about the assets in the abstract; see the golden-case verdicts
  in the Phase 4 report.
"""

from __future__ import annotations

from runtime.retrieval_expansion import (
    build_live_decomposition_generator,
    detect_multi_part,
    stub_decompose_question,
)
from runtime.retrieval_graph import gazetteer_seeds

__all__ = [
    "decompose_question",
    "detect_multi_part",
    "relatedness_verdict",
    "stub_decompose_question",
]

# BFS distance at or below which two seed sets count as related:
# distance 0 = shared seed, 1 = adjacent seeds, 2 = one node within
# 1 hop of both seeds' sets.
RELATED_MAX_DISTANCE = 2


# --- Undirected BFS over the adjacency -----------------------------------

def _undirected_neighbours(adjacency: dict, node: str) -> set[str]:
    """Neighbours of ``node`` on the undirected graph.

    The adjacency stores each edge in both directions, but the
    lookup is written direction-insensitively (a listed edge counts
    either way) so a synthetic adjacency that lists an edge once
    still behaves as undirected, as the rule specifies.
    """
    neighbours = {
        to_key
        for _rel, to_key, _to_type in adjacency.get(node, {}).get("neighbors", [])
    }
    for other, entry in adjacency.items():
        if other == node:
            continue
        if any(to_key == node for _rel, to_key, _t in entry.get("neighbors", [])):
            neighbours.add(other)
    return neighbours


def _bfs_distance(adjacency: dict, start: str, targets: set[str]) -> int | None:
    """Shortest undirected hop distance from ``start`` to any target."""
    if start in targets:
        return 0
    seen = {start}
    frontier = {start}
    distance = 0
    while frontier:
        distance += 1
        nxt: set[str] = set()
        for node in frontier:
            for neighbour in _undirected_neighbours(adjacency, node):
                if neighbour in targets:
                    return distance
                if neighbour not in seen:
                    seen.add(neighbour)
                    nxt.add(neighbour)
        frontier = nxt
    return None


def _closed_one_hop(adjacency: dict, seeds: list[str]) -> set[str]:
    """A seed set plus every node within 1 hop of any of its seeds."""
    closed = set(seeds)
    for seed in seeds:
        closed |= _undirected_neighbours(adjacency, seed)
    return closed


# --- Relatedness verdict ---------------------------------------------------

def relatedness_verdict(seed_sets: list[list[str]], adjacency: dict | None = None) -> dict:
    """RELATED / INDEPENDENT verdict for a list of per-part seed sets.

    ``seed_sets`` is one seed-key list per sub-question, in order.
    Returns ``{"related": bool | None, "shared_nodes": [...],
    "min_distance": int | None}``. ``related`` is ``None`` when
    fewer than two parts carry seeds (no pair to compare);
    otherwise ``True`` iff any seeded pair is within
    ``RELATED_MAX_DISTANCE`` hops, with ``shared_nodes`` the sorted
    union of the closed-1-hop intersections of the related pairs.

    ``adjacency`` is REQUIRED (the live graph adjacency): there is
    no demo adjacency to fall back to.
    """
    if adjacency is None:
        raise ValueError(
            "relatedness_verdict requires 'adjacency' (the live graph "
            "adjacency); no default adjacency exists."
        )
    seeded = [list(seeds) for seeds in seed_sets if seeds]
    if len(seeded) < 2:
        return {"related": None, "shared_nodes": [], "min_distance": None}
    related = False
    shared: set[str] = set()
    min_distance: int | None = None
    for i, left in enumerate(seeded):
        for right in seeded[i + 1:]:
            left_set, right_set = set(left), set(right)
            distance = min(
                (
                    d
                    for d in (
                        _bfs_distance(adjacency, seed, right_set) for seed in left_set
                    )
                    if d is not None
                ),
                default=None,
            )
            if distance is not None and (
                min_distance is None or distance < min_distance
            ):
                min_distance = distance
            if distance is not None and distance <= RELATED_MAX_DISTANCE:
                related = True
                shared |= _closed_one_hop(adjacency, left) & _closed_one_hop(
                    adjacency, right
                )
    return {
        "related": related,
        "shared_nodes": sorted(shared),
        "min_distance": min_distance,
    }


# --- Per-question decomposition --------------------------------------------

def decompose_question(
    question: str,
    *,
    decomposition_generator=None,
    adjacency: dict | None = None,
) -> dict:
    """Decompose one question and judge whether its parts are related.

    Returns ``{"is_multi_part", "reasons", "related", "shared_nodes",
    "sub_questions": [{"text", "linked_nodes": [...]}]}``.

    ``decomposition_generator`` is a pluggable callable
    ``question -> list[str]``; the default is
    ``build_live_decomposition_generator`` (lazy model build — no
    LLM call until a gated query needs one). Pass
    ``stub_decompose_question`` for the deterministic stub path.
    ``adjacency`` is REQUIRED (the live graph adjacency) and is
    passed through to ``relatedness_verdict``. When the generator
    returns the question unchanged
    (or nothing), that is reported as "no decomposition":
    ``sub_questions=[]`` and ``related=None`` — sub-questions are
    never invented here.
    """
    if adjacency is None:
        raise ValueError(
            "decompose_question requires 'adjacency' (the live graph "
            "adjacency); no default adjacency exists."
        )
    detection = detect_multi_part(question)
    if not detection["is_multi_part"]:
        return {
            "is_multi_part": False,
            "reasons": detection["reasons"],
            "related": None,
            "shared_nodes": [],
            "sub_questions": [],
        }
    generator = decomposition_generator or build_live_decomposition_generator()
    raw_parts = list(generator(question) or [])
    parts = []
    for part in raw_parts:
        part = (part or "").strip()
        if part and part not in parts:
            parts.append(part)
    if len(parts) == 1 and parts[0].strip().lower() == (question or "").strip().lower():
        # Generator says single-part (or cannot split this wording):
        # "no decomposition", not a one-element split.
        return {
            "is_multi_part": True,
            "reasons": detection["reasons"],
            "related": None,
            "shared_nodes": [],
            "sub_questions": [],
        }
    sub_questions = [
        {
            "text": part,
            "linked_nodes": [seed["key"] for seed in gazetteer_seeds(part)],
        }
        for part in parts
    ]
    verdict = relatedness_verdict(
        [sub["linked_nodes"] for sub in sub_questions], adjacency
    )
    return {
        "is_multi_part": True,
        "reasons": detection["reasons"],
        "related": verdict["related"],
        "shared_nodes": verdict["shared_nodes"],
        "sub_questions": sub_questions,
    }
