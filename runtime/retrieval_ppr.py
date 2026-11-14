"""Personalized PageRank over the candidate subgraph — Phase 2
(report §4.1 G7, §4.4c, §4.7 step 2; locked decisions: PPR is a
*feature* inside the deterministic fusion, never the sole ranker).

Pure-Python power iteration — no new dependency. ``igraph`` and
``networkx`` are NOT installed in the project venv (checked
2026-11-14); ``scipy`` is installed but a sparse-matrix PPR would
drag a heavy dependency into a hot path for a <=~200-node graph
where power iteration is ~30 lines. No GDS dependency either: the
GDS PageRank docs page returned 403 during research (§4.4).

DAMPING-CONVENTION TRAP (report §4.4, mandatory): HippoRAG uses
``damping = 0.5`` in ``igraph.personalized_pagerank``, where damping
= probability of *following an edge* (restart = 0.5). Neo4j GDS
``gds.pageRank`` uses ``dampingFactor: 0.85`` in the SAME convention
(follow = 0.85, restart = 0.15) — so copying "0.85" from GDS
examples gives a far more diffuse walk than HippoRAG's 0.5, and
copying "0.5" into a library that defines damping as *restart*
probability inverts it. This module therefore names the parameter
``restart_prob`` and documents it at every use:

    restart_prob = 0.5  <=>  follow-edge probability 0.5
                          <=>  HippoRAG damping 0.5
                          <=>  GDS dampingFactor 0.5

Never call it "damping" without stating which convention. Note
0.5 is self-dual (restart == follow), so the convention is pinned
by the *tests* varying restart_prob (0.9 seeds dominate, 0.1 mass
diffuses far), not by the default value alone.

Seeds: the personalization vector weights each seed by
``confidence * specificity`` (``spec_s = 1 / chunks-per-seed``,
HippoRAG node-specificity), normalised to sum to 1. Specificity
counts come from the caller (``seed_specificity``): a seed that
appears in 6 documents (O21) seeds with weight 1/6, a seed that
appears in 1 (motor-M2) with weight 1, a seed in no document with
weight 1 (maximally specific — it appears nowhere, so ``max(1, n)``
avoids the divide-by-zero and caps spec at 1.0).

Edge weights: the §4.2 relation weights (DOCUMENTED_BY 1.00,
HAS_COMPONENT/PART_OF 0.90, LOCATED_IN 0.40 / 0.15 via a hub,
RELATED_TO 0.25), row-normalised per node — weighted PPR, as
§4.4c recommends for hub mitigation. Dangling nodes (no
neighbours) teleport to the personalization vector. Iteration
stops when the L1 change drops below ``tol`` (default 1e-9) or
``max_iter`` (default 200) is reached; the demo graph converges
in far fewer iterations.

Scores sum to 1.0 over the adjacency nodes. A document's PPR is
its *document node's* score (``runtime/retrieval_graph.py``
``graph_features`` reads it as G7). The alternative — max over the
doc's linked entities — was prototyped and REJECTED: it lets a doc
inherit a seed-adjacent entity's mass even when the doc itself is
far, and it dropped aggregate nDCG@6 0.838 -> 0.822 on the blurred
setup (see docs/hybrid-retrieval-phase2.md).
"""

from __future__ import annotations

# Restart probability (see module docstring): 0.5 restarts to the
# seeds, 0.5 follows an edge — the HippoRAG convention.
RESTART_PROB = 0.5
DEFAULT_MAX_ITER = 200
DEFAULT_TOL = 1e-9


def seed_specificity(seeds: list[dict], doc_counts: dict[str, int]) -> dict[str, float]:
    """Specificity per seed key: ``1 / max(1, docs containing the seed)``.

    ``doc_counts`` maps node key -> number of documents whose
    ``node_ids`` contain it. A seed in no document gets spec 1.0
    (maximally specific); a seed in 6 documents gets 1/6.
    """
    return {
        s["key"]: 1.0 / max(1, int(doc_counts.get(s["key"], 0)))
        for s in seeds
    }


def personalized_pagerank(
    adjacency: dict,
    seeds: list[dict],
    *,
    restart_prob: float = RESTART_PROB,
    seed_weights: dict[str, float] | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> dict[str, float]:
    """PPR over ``adjacency``, seeded on ``seeds``.

    ``adjacency`` is the ``runtime/retrieval_graph.py`` shape:
    ``key -> {"type": str, "neighbors": [(rel, to_key, to_type)]}``,
    edges stored directed as modelled but treated as undirected for
    distance (each edge appears in both endpoint lists). Edge weight
    is the §4.2 relation weight (see ``_relation_weight``).

    ``seed_weights`` maps seed key -> personalization weight
    (``confidence * specificity`` when precomputed by the caller);
    when omitted each seed weighs its ``confidence`` (default 1.0).
    Seeds absent from the adjacency are ignored; if no seed weight
    lands on a node, every score is 0.0.

    Returns ``{node_key: score}`` summing to 1.0 (or all zeros).
    Pure Python; deterministic (nodes processed in sorted order).
    """
    from runtime.retrieval_graph import _relation_weight

    nodes = sorted(adjacency)
    scores = {node: 0.0 for node in nodes}
    personalization = {node: 0.0 for node in nodes}
    total = 0.0
    for seed in seeds:
        key = seed["key"]
        if key not in personalization:
            continue
        weight = (
            seed_weights[key]
            if seed_weights is not None and key in seed_weights
            else float(seed.get("confidence", 1.0))
        )
        personalization[key] += weight
        total += weight
    if total == 0:
        return scores
    personalization = {node: value / total for node, value in personalization.items()}

    # Row-normalised weighted transitions, in sorted node order.
    transitions: dict[str, dict[str, float]] = {}
    for node in nodes:
        out: dict[str, float] = {}
        for rel, to_key, _to_type in adjacency[node]["neighbors"]:
            out[to_key] = out.get(to_key, 0.0) + _relation_weight(rel, to_key, adjacency)
        weight_sum = sum(out.values())
        transitions[node] = (
            {to_key: w / weight_sum for to_key, w in out.items()} if weight_sum > 0 else {}
        )

    follow_prob = 1.0 - restart_prob
    ranks = dict(personalization)
    for _ in range(max_iter):
        nxt = {node: restart_prob * personalization[node] for node in nodes}
        for node in nodes:
            mass = follow_prob * ranks[node]
            if mass == 0.0:
                continue
            outs = transitions[node]
            if not outs:
                # Dangling node: teleport to the personalization vector.
                for target in nodes:
                    nxt[target] += mass * personalization[target]
            else:
                for target, prob in outs.items():
                    nxt[target] += mass * prob
        delta = sum(abs(nxt[node] - ranks[node]) for node in nodes)
        ranks = nxt
        if delta < tol:
            break
    return ranks
