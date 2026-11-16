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
