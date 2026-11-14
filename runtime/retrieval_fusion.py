"""Deterministic fusion — Phase 1 (report §4.7, locked decisions).

No reranker model in Phase 1: the "reranker" is this formula, with
the vector score as the text arm and ``graph_score`` (G1/G2/G4/G5,
``runtime/retrieval_graph.py``) as the ontology arm.

EXACT Phase 1 formula (as built)::

    z(x)      = per-query z-score of x over the fused candidates,
                clipped to [-2, 2] (std 0 -> z = 0 for all)
    final     = 0.50 * z(vec) + 0.40 * z(graph_score) + 0.10 * exact
    exact     = 1 if a gazetteer code from the question appears verbatim
                in the candidate's title/content, else 0

``graph_score`` enters via its per-query z-score, exactly as §4.7's
``graph_score`` (itself a z-combination) enters the original
``0.50*z(ce) + 0.30*graph_score + 0.15*z(vec) + 0.05*exact`` — with
the cross-encoder arm removed (no model in Phase 1) and its weight
moved to the graph arm (0.30 -> 0.40, locked decision).

Pipeline per query: candidates = Qdrant top-40 dense UNION Neo4j
1-hop linked docs; RRF (k=60) over vector rank + graph rank ->
top-30; fusion -> guardrails -> top-6.

Guardrails (hard post-fusion rules, §4.7, adapted to the no-CE
Phase 1 — "ce in the top half" becomes "vector score in the top
half", guardrail 1 reworded per the locked decisions):
1. Direct-link floor: direct=1 and vec in the top half of the fused
   30 can never fall below rank 6 — implemented as a swap with the
   lowest-ranked non-qualifying doc in the top-6, so promoting one
   qualifier never displaces another qualifier.
2. Generic-path cap: a candidate whose best path uses ONLY
   RELATED_TO (or only LOCATED_IN through a hub) can never enter
   the top-3.
3. Graph-missing fallback: no seeds linked, or expansion empty ->
   graph_score = 0 for all, weights renormalise over the remaining
   arms: vec 0.50/0.60 = 0.833, exact 0.10/0.60 = 0.167; the query
   is flagged ``graph_missing=true``. The system degrades to
   vector-only, never fails.
4. Per-document cap: at most 2 chunks from the same document_id in
   the final 6.
"""

from __future__ import annotations

import math

RRF_K = 60
TOP_FUSED = 30
TOP_FINAL = 6

W_VEC = 0.50
W_GRAPH = 0.40
W_EXACT = 0.10
Z_CLIP = 2.0


def zscores(values: list[float]) -> list[float]:
    """Per-query z-scores, clipped to [-2, 2]. Std 0 -> all zeros."""
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    if std == 0:
        return [0.0] * len(values)
    return [max(-Z_CLIP, min(Z_CLIP, (v - mean) / std)) for v in values]


def rrf_scores(rank_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion over ranked id lists (rank is 1-based)."""
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def renormalise_weights(w_graph: float) -> dict[str, float]:
    """Grid-search weights: w_graph varies, vec:exact keep their 5:1 ratio."""
    rest = 1.0 - w_graph
    return {
        "w_vec": round(rest * W_VEC / (W_VEC + W_EXACT), 4),
        "w_graph": round(w_graph, 4),
        "w_exact": round(rest * W_EXACT / (W_VEC + W_EXACT), 4),
    }


def fuse_candidates(
    candidates: list[dict],
    *,
    w_vec: float = W_VEC,
    w_graph: float = W_GRAPH,
    w_exact: float = W_EXACT,
    graph_missing: bool = False,
) -> list[dict]:
    """Score + order candidates, applying guardrails 1, 2 and 4.

    Each candidate needs ``doc_id``, ``vec_score``, ``graph_score``,
    ``exact`` (0/1), ``g5_direct``, ``best_path_rels`` and
    ``document_id`` (for the per-doc cap; defaults to ``doc_id``).
    Returns the candidates sorted by final score, each with
    ``final_score`` and ``rank`` (1-based). Guardrail 3 (fallback
    weights) is applied by the caller via ``graph_missing=True``,
    which renormalises the weights before scoring.
    """
    if not candidates:
        return []
    if graph_missing:
        # Renormalise over the surviving arms (0.50 + 0.10 = 0.60).
        total = w_vec + w_exact
        w_vec, w_graph, w_exact = w_vec / total, 0.0, w_exact / total
    vec_z = zscores([c.get("vec_score", 0.0) for c in candidates])
    graph_z = zscores([c.get("graph_score", 0.0) for c in candidates])
    scored: list[dict] = []
    for cand, vz, gz in zip(candidates, vec_z, graph_z):
        final = w_vec * vz + w_graph * gz + w_exact * float(cand.get("exact", 0))
        scored.append({**cand, "final_score": round(final, 4),
                       "z_vec": round(vz, 4), "z_graph": round(gz, 4)})
    # Ties (common on a 7-doc corpus where several docs share a
    # graph score and a ~0 vector score) break toward the stronger
    # graph score, then the stronger vector score, then doc_id.
    scored.sort(key=lambda c: (
        -c["final_score"], -c.get("graph_score", 0.0),
        -c.get("vec_score", 0.0), c["doc_id"],
    ))
    ranked = _apply_guardrails(scored)
    # Guardrail 4: per-document cap, applied when taking the final 6.
    out: list[dict] = []
    per_doc: dict[str, int] = {}
    for cand in ranked:
        doc_id = cand.get("document_id", cand["doc_id"])
        if per_doc.get(doc_id, 0) >= 2:
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        out.append(cand)
        if len(out) >= TOP_FINAL:
            break
    for i, cand in enumerate(out, start=1):
        cand["rank"] = i
    return out


def _apply_guardrails(ranked: list[dict]) -> list[dict]:
    """Guardrails 1 (direct-link floor) and 2 (generic-path cap)."""
    ranked = list(ranked)
    if not ranked:
        return ranked
    vec_scores = sorted((c.get("vec_score", 0.0) for c in ranked), reverse=True)
    median_vec = vec_scores[len(vec_scores) // 2]
    # Guardrail 1: a qualifying direct-linked candidate below rank 6
    # swaps with the lowest-ranked NON-qualifying candidate inside
    # the top-6. A plain promotion would push another qualifying doc
    # (e.g. a second O21-linked doc) out of the top-6 to satisfy the
    # first — the swap keeps the floor without displacing a qualifier.
    # If every top-6 doc already qualifies, there is nothing to swap
    # with and the order stands (the 7th qualifier loses to a
    # qualifier, which the guardrail does not forbid).
    def _qualifies(cand: dict) -> bool:
        return (
            cand.get("g5_direct") == 1.0
            and cand.get("vec_score", 0.0) >= median_vec
        )

    for i, cand in enumerate(list(ranked)):
        if i < TOP_FINAL or not _qualifies(cand):
            continue
        for j in range(TOP_FINAL - 1, -1, -1):
            if not _qualifies(ranked[j]):
                ranked[j], ranked[i] = ranked[i], ranked[j]
                break
    # Guardrail 2: demote generic-path-only candidates out of top-3.
    for i, cand in enumerate(list(ranked)):
        if i >= 3:
            break
        rels = cand.get("best_path_rels") or []
        if not rels:
            continue
        generic = all(r == "RELATED_TO" for r in rels) or all(
            r == "LOCATED_IN" and cand.get("via_hub") for r in rels
        )
        if generic:
            ranked.remove(cand)
            ranked.insert(3, cand)
    return ranked


def rrf_prefusion(candidates: list[dict]) -> list[dict]:
    """RRF over vector rank + graph rank -> top-30 (pre-fusion cut).

    Candidates need ``doc_id``, ``vec_score``, ``graph_score``.
    Ties break by doc_id for determinism.
    """
    if not candidates:
        return []
    vec_ranked = [c["doc_id"] for c in sorted(
        candidates, key=lambda c: (-c.get("vec_score", 0.0), c["doc_id"]))]
    graph_ranked = [c["doc_id"] for c in sorted(
        candidates, key=lambda c: (-c.get("graph_score", 0.0), c["doc_id"]))]
    scores = rrf_scores([vec_ranked, graph_ranked])
    by_id = {c["doc_id"]: c for c in candidates}
    ordered = sorted(by_id, key=lambda d: (-scores[d], d))
    return [by_id[doc_id] for doc_id in ordered[:TOP_FUSED]]
