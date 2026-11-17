"""Retrieval weight sets — one home for the three hardcoded sets.

Three DIFFERENT sets, three different jobs — centralised here so a
tune touches one file, not three modules that happen to share names:

* ``FUSION_*`` (0.50 / 0.40 / 0.10): the final weighted fusion in
  ``runtime/retrieval_fusion.py`` — ``W_VEC * z(vec) + W_GRAPH *
  graph + W_EXACT * exact``. Locked 2026-11-14 (report
  §5.2, ``docs/hybrid-retrieval.md`` §5).
* ``GRAPH_*`` (0.45 / 0.30 / 0.25): the graph_score components in
  ``runtime/retrieval_graph.py`` — PPR : path : coverage.
* ``INTENT_*`` (0.50 / 0.30 / 0.20): the intent scorer's fusion
  priors in ``runtime/retrieval_intent.py`` — a different prior
  from the final fusion (more weight on exact, less on graph).

Values are unchanged by the move (2026-11-16): each module keeps
its local names (``W_VEC`` etc.) as aliases of these, so existing
imports (``runtime/retrieval_hybrid.py``, ``tests/test_query_plan_intent.py``)
keep working. Each set sums to 1.0.

The second half of this file (added 2026-11-17, hardcode cleanup)
holds the OTHER shared retrieval literals — the graph path-term
formula, the guardrail sub-values, and the defaults that used to
be copy-pasted across modules. Values are unchanged; only the
spelling moved here so a tune touches one file.
"""

from __future__ import annotations

# Final fusion (retrieval_fusion.py).
FUSION_W_VEC = 0.50
FUSION_W_GRAPH = 0.40
FUSION_W_EXACT = 0.10

# Graph score components (retrieval_graph.py): PPR : path : coverage.
GRAPH_W_PPR = 0.45
GRAPH_W_PATH = 0.30
GRAPH_W_COVERAGE = 0.25

# Intent scorer priors (retrieval_intent.py).
INTENT_W_VEC = 0.50
INTENT_W_GRAPH = 0.30
INTENT_W_EXACT = 0.20

# --- Graph path term (runtime/retrieval_graph.py::graph_score) --------
# graph_score = clip(path_score
#     + GRAPH_PATH_W_G4 * G4
#     + GRAPH_PATH_W_G5 * (G5_direct - 0.5)
#     + hop_adj                          # G1 0 -> +HOP_ADJ, 1 -> 0, None -> -HOP_ADJ
#     + GRAPH_PATH_SEED_BONUS * min(max(n_seeds_linked - 1, 0), SEED_BONUS_CAP),
#     SCORE_MIN, SCORE_MAX)
# The floor is -0.3, not 0: a G4 mismatch must score *below* an
# unlinked doc (see graph_score's docstring). Values unchanged.
GRAPH_PATH_W_G4 = 0.10
GRAPH_PATH_W_G5 = 0.10
GRAPH_PATH_HOP_ADJ = 0.05
GRAPH_PATH_SEED_BONUS = 0.02
GRAPH_PATH_SEED_BONUS_CAP = 2
GRAPH_PATH_SCORE_MIN = -0.3
GRAPH_PATH_SCORE_MAX = 1.2

# G5 direct-link values (retrieval_graph.py::graph_features).
G5_DIRECT = 1.0
G5_ONE_HOP = 0.5

# --- Guardrail sub-values (runtime/retrieval_fusion.py, locked) -------
# Guardrail 2: a generic-path-only candidate can never enter the
# top-GENERIC_PATH_TOP_N. Guardrail 4: at most PER_DOC_CAP chunks
# from the same document in the final 6. Guardrail 1: a direct
# link (g5_direct == DIRECT_LINK_VALUE) can never fall below rank 6.
GENERIC_PATH_TOP_N = 3
PER_DOC_CAP = 2
DIRECT_LINK_VALUE = 1.0

# --- Shared defaults (one spelling, used by every module) --------------
# Default document limit for a retrieval call: the hook's
# pre-retrieval block, the fault_analysis knowledge node, and
# infra/qdrant_client.search_documents all default to this.
DEFAULT_DOC_LIMIT = 5
# Default seed confidence when a seed carries none: a seed the
# gazetteer named is trusted (1.0); the normalised-variant score
# (0.8, NORMALISED_CONFIDENCE) is the lower, named value.
DEFAULT_SEED_CONFIDENCE = 1.0
# Score precision in traces and fused rows (fusion + hybrid + graph).
TRACE_SCORE_PRECISION = 4
# Multi-part threshold: how many distinct assets / fault codes make
# a question multi-part (expansion gating + intent comparison).
MULTI_PART_MIN_COUNT = 2
