"""Hybrid retriever — Phases 1–2 (report §4.7, §8 Phases 1–2).

NEW module, evaluated through the Phase 0 harness. It does NOT
change ``infra.qdrant_client.search_documents`` or the
``retrieval_context`` hook — those keep their current behaviour
(see ``docs/hybrid-retrieval-phase1.md`` for the follow-up wiring
step, opt-in behind mission config).

Pipeline per question:
1. Vector arm: ``search_documents`` (real code path) top-40 dense,
   via the injected embeddings + client (fakes in this VM, real
   LiteLLM/Qdrant live).
2. Gazetteer seeds (``runtime/retrieval_graph.py``) from the question.
3. Graph arm: 1-hop expansion over the demo adjacency (fakes) or
   ``expand_via_graph_query`` (live); every doc linked to a seed or
   a 1-hop neighbour becomes a candidate, with G1/G2/G4/G5 (Phase 1)
   plus G6 coverage, G7 PPR and G8 hub (Phase 2, default on).
4. Candidates = vector top-40 UNION graph-linked docs, deduped.
5. RRF (k=60) over vector rank + graph rank -> top-30.
6. Fusion (``runtime/retrieval_fusion.py``) -> guardrails -> top-6.
7. ``exact`` = 1 when a gazetteer code from the question appears
   verbatim in the candidate's title/content.

Phase 2 (default): ``graph_score`` is the locked full formula
``0.45 * minmax_log(PPR) + 0.30 * z(path_term) + 0.25 * coverage``
(``runtime/retrieval_graph.py::phase2_graph_scores``), with the
hub penalty inside the path term. ``phase2=False`` reproduces the
Phase 1 score (``scripts/eval_retrieval_phase1.py`` pins it so
Phase 1's numbers stay reproducible). ``w_ppr=0`` is the PPR
ablation flag: it zeroes the PPR sub-weight with no renormalising.

Trace per query (logged by the harness): seeds, expansion nodes
returned / truncated, graph_missing, per-candidate features and
scores, guardrail flags.
"""

from __future__ import annotations

import logging

from runtime.retrieval_baseline import (
    DeterministicFakeEmbeddings,
    FakeQdrantClient,
)
from runtime.retrieval_blurred import CodeBlurredFakeEmbeddings
from runtime.retrieval_fusion import (
    W_EXACT,
    W_GRAPH,
    W_VEC,
    fuse_candidates,
    rrf_prefusion,
)
from runtime.retrieval_graph import (
    W_PPR,
    build_demo_adjacency,
    expand_one_hop,
    expand_via_graph_query,
    gazetteer_seeds,
    graph_features,
    graph_score,
    phase2_graph_scores,
)
from runtime.retrieval_linking import DOCUMENT_LINKS

logger = logging.getLogger(__name__)

VECTOR_LIMIT = 40


def _exact_match(question: str, doc: dict) -> int:
    """1 if a code the question names appears verbatim in the doc.

    Codes are the gazetteer seeds (asset/component/building/doc
    codes) plus fault codes found in the question text (``F-204`` —
    the codes the docs actually print). Only codes the question
    names count, never the whole fault table.
    """
    import re

    question_codes = {s["key"].lower() for s in gazetteer_seeds(question)}
    for match in re.finditer(r"\bf-\d{3}\b", question.lower()):
        question_codes.add(match.group(0))
    text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
    return 1 if any(code in text for code in question_codes) else 0


class HybridRetriever:
    """Dense + 1-hop graph + deterministic fusion. Fake-friendly."""

    def __init__(
        self,
        embeddings=None,
        client=None,
        adjacency: dict | None = None,
        *,
        use_live_graph: bool = False,
        graph_store=None,
        w_vec: float = W_VEC,
        w_graph: float = W_GRAPH,
        w_exact: float = W_EXACT,
        phase2: bool = True,
        w_ppr: float = W_PPR,
    ):
        self.embeddings = embeddings or DeterministicFakeEmbeddings()
        self.client = client or FakeQdrantClient(self.embeddings)
        self.adjacency = adjacency if adjacency is not None else build_demo_adjacency()
        self.use_live_graph = use_live_graph
        self.graph_store = graph_store
        self.weights = {"w_vec": w_vec, "w_graph": w_graph, "w_exact": w_exact}
        self.phase2 = phase2
        # PPR ablation flag: w_ppr=0 zeroes the PPR sub-weight
        # (no renormalising — see phase2_graph_scores).
        self.w_ppr = w_ppr
        # Doc counts for seed specificity (PPR personalization):
        # number of linked documents per node, from the grounded
        # linking table. Live wiring must replace this with real
        # corpus counts (see docs/hybrid-retrieval-phase2.md §7).
        self._doc_counts: dict[str, int] = {}
        for links in DOCUMENT_LINKS.values():
            for node_key in links["node_ids"]:
                self._doc_counts[node_key] = self._doc_counts.get(node_key, 0) + 1
        # Doc texts for the graph arm / exact feature (fake corpus).
        from seeds.demo_data import DOCUMENTS

        self._docs = {d["id"]: d for d in DOCUMENTS}

    def __call__(self, question: str):
        return self.retrieve(question)

    def _rank_candidates(self, candidates: list[dict], graph_missing: bool) -> list[dict]:
        """RRF pre-fusion -> deterministic fusion -> guardrails -> top-6.

        Shared by ``retrieve`` and by the Phase 3 expansion wrapper
        (``runtime/retrieval_expansion.py``), so a merged multi-variant
        candidate set is ranked by exactly the same fusion as a
        single-question set.
        """
        fused = rrf_prefusion(candidates)
        return fuse_candidates(
            fused,
            w_vec=self.weights["w_vec"],
            w_graph=self.weights["w_graph"],
            w_exact=self.weights["w_exact"],
            graph_missing=graph_missing,
        )

    def _candidate_rows(self, question: str):
        """Per-question candidate rows (pre-RRF, pre-fusion).

        Returns ``(candidates, seeds, expansion, graph_missing)``.
        Each candidate carries the full §4.1 feature row
        (``graph_g1_min_hops`` … ``graph_g8_transit_key``) plus
        ``vec_score`` / ``graph_score`` / ``exact`` / ``g5_direct`` /
        ``best_path_rels`` — everything ``fuse_candidates`` needs.
        Phase 3 expansion calls this once per variant and merges the
        rows before ranking; ``retrieve`` calls it once.
        """
        from infra.qdrant_client import search_documents

        vector_docs = search_documents(
            question, limit=VECTOR_LIMIT, embeddings=self.embeddings, client=self.client
        )
        vec_by_id = {d["doc_id"]: d for d in vector_docs}
        vec_scores = {d["doc_id"]: float(d.get("score", 0.0)) for d in vector_docs}

        seeds = gazetteer_seeds(question)
        if self.use_live_graph:
            expansion = expand_via_graph_query(seeds, store=self.graph_store)
        else:
            expansion = expand_one_hop(seeds, self.adjacency)
        graph_missing = not seeds or not expansion.get("nodes")

        # Phase 2 PPR (once per query, over the candidate subgraph —
        # here the whole demo adjacency, <=~200 nodes by design).
        # Seeds weigh confidence * specificity; a seed in no linked
        # document is maximally specific (weight 1.0).
        ppr_scores: dict = {}
        if self.phase2 and not graph_missing:
            from runtime.retrieval_ppr import personalized_pagerank, seed_specificity

            spec = seed_specificity(seeds, self._doc_counts)
            seed_weights = {
                s["key"]: float(s.get("confidence", 1.0)) * spec[s["key"]] for s in seeds
            }
            ppr_scores = personalized_pagerank(
                self.adjacency, seeds, seed_weights=seed_weights
            )

        # Graph-linked candidate docs: docs whose linked nodes touch a
        # seed or a 1-hop neighbour, plus docs that ARE expansion nodes.
        graph_doc_ids: set[str] = set()
        for doc_id, links in DOCUMENT_LINKS.items():
            linked = set(links["node_ids"])
            if linked & {s["key"] for s in seeds}:
                graph_doc_ids.add(doc_id)
            elif linked & set(expansion.get("nodes", {})):
                graph_doc_ids.add(doc_id)
        for node_key in expansion.get("nodes", {}):
            if node_key in self._docs:
                graph_doc_ids.add(node_key)

        candidate_ids = sorted(set(vec_by_id) | graph_doc_ids)
        feature_rows: list[dict] = []
        for doc_id in candidate_ids:
            doc = self._docs.get(doc_id) or vec_by_id.get(doc_id) or {"doc_id": doc_id}
            feature_rows.append(graph_features(
                doc_id, seeds, expansion, self.adjacency,
                payload=vec_by_id.get(doc_id),
                ppr_scores=ppr_scores or None,
            ))
        if self.phase2 and not graph_missing:
            gscores = phase2_graph_scores(feature_rows, w_ppr=self.w_ppr)
        else:
            gscores = [
                0.0 if graph_missing else graph_score(features)
                for features in feature_rows
            ]
        candidates: list[dict] = []
        for doc_id, features, gscore in zip(candidate_ids, feature_rows, gscores):
            doc = self._docs.get(doc_id) or vec_by_id.get(doc_id) or {"doc_id": doc_id}
            arms = []
            if doc_id in vec_by_id:
                arms.append("vector")
            if doc_id in graph_doc_ids:
                arms.append("graph")
            candidates.append({
                "doc_id": doc_id,
                "document_id": doc_id,
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "vec_score": vec_scores.get(doc_id, 0.0),
                "graph_score": gscore,
                "exact": _exact_match(question, doc),
                "g5_direct": features["g5_direct"],
                "best_path_rels": features["best_path_rels"],
                "arms": arms,
                **{f"graph_{k}": v for k, v in features.items()},
            })
        return candidates, seeds, expansion, graph_missing

    def retrieve(self, question: str):
        candidates, seeds, expansion, graph_missing = self._candidate_rows(question)
        final = self._rank_candidates(candidates, graph_missing)
        ranked_ids = [c["doc_id"] for c in final]
        trace = {
            "fake": True,
            "phase2": self.phase2,
            "seeds": [s["key"] for s in seeds],
            "expansion_nodes_returned": expansion.get("nodes_returned", 0),
            "expansion_truncated": bool(expansion.get("truncated", False)),
            "graph_missing": graph_missing,
            "weights": dict(self.weights),
            "w_ppr": self.w_ppr,
            # Full per-candidate feature rows (Phase 3 LTR logging):
            # every fused candidate, not just the final top-6.
            "feature_rows": trace_feature_rows(candidates, final),
            "candidates": {
                c["doc_id"]: {
                    "vec": round(c["vec_score"], 4),
                    "graph": round(c["graph_score"], 4),
                    "exact": c["exact"],
                    "final": c.get("final_score"),
                    "g1": c.get("graph_g1_min_hops"),
                    "g2": c.get("graph_g2_path_score"),
                    "g4": c.get("graph_g4_type_match"),
                    "g5": c.get("graph_g5_direct"),
                    "g6_coverage": c.get("graph_g6_coverage"),
                    "g7_ppr": c.get("graph_g7_ppr"),
                    "g8_hub": c.get("graph_g8_hub"),
                    "g8_transit": c.get("graph_g8_transit_key"),
                }
                for c in final
            },
        }
        return ranked_ids, trace


def trace_feature_rows(candidates: list[dict], final: list[dict]) -> list[dict]:
    """Full per-candidate feature rows for one query (Phase 3 LTR logging).

    One row per candidate in the fused set (pre-fusion, i.e. all 7
    docs on the demo corpus — not just the final top-6), with the
    §4.1 feature row (G1–G9), both ranks (``rrf_rank`` = rank before
    the fusion rerank, ``rank`` = final rank, ``None`` when the
    candidate was cut by the top-6), the final score, and the
    retrieval arms that found it. ``fuse_candidates`` copies its
    inputs, so the rows are rebuilt here from ``candidates`` +
    ``final`` rather than read off the fused list.
    """
    rrf_ranked = rrf_prefusion(candidates)
    rrf_rank = {c["doc_id"]: i + 1 for i, c in enumerate(rrf_ranked)}
    final_by_id = {c["doc_id"]: c for c in final}
    rows: list[dict] = []
    for cand in candidates:
        fin = final_by_id.get(cand["doc_id"], {})
        rows.append({
            "doc_id": cand["doc_id"],
            "document_id": cand.get("document_id", cand["doc_id"]),
            "vec_score": cand.get("vec_score", 0.0),
            "graph_score": cand.get("graph_score", 0.0),
            "exact": cand.get("exact", 0),
            "g1_min_hops": cand.get("graph_g1_min_hops"),
            "g2_path_score": cand.get("graph_g2_path_score"),
            "g3_best_path_rels": list(cand.get("best_path_rels") or []),
            "g4_type_match": cand.get("graph_g4_type_match"),
            "g5_direct": cand.get("graph_g5_direct"),
            "g5_n_seeds_linked": cand.get("graph_g5_n_seeds_linked"),
            "g6_coverage": cand.get("graph_g6_coverage"),
            "g7_ppr": cand.get("graph_g7_ppr"),
            "g8_hub": cand.get("graph_g8_hub"),
            "g8_transit_key": cand.get("graph_g8_transit_key"),
            "g9_exact": cand.get("exact", 0),
            "arms": list(cand.get("arms") or []),
            "rrf_rank": rrf_rank.get(cand["doc_id"]),
            "rank": fin.get("rank"),
            "final_score": fin.get("final_score"),
            "z_vec": fin.get("z_vec"),
            "z_graph": fin.get("z_graph"),
        })
    return rows


def build_hybrid_retriever(*, blurred: bool = False, phase2: bool = True,
                           w_ppr: float = W_PPR, **kwargs):
    """Hybrid retriever over the fake corpus.

    ``blurred=True`` uses the code-blurred fake embeddings (the
    exit-criterion setup); ``False`` reuses the Phase 0 BoW fake.
    ``phase2=False`` reproduces the Phase 1 graph score;
    ``w_ppr=0`` is the PPR ablation flag.
    """
    embeddings = CodeBlurredFakeEmbeddings() if blurred else DeterministicFakeEmbeddings()
    client = FakeQdrantClient(embeddings)
    retriever = HybridRetriever(embeddings=embeddings, client=client,
                                phase2=phase2, w_ppr=w_ppr, **kwargs)
    retriever.__name__ = "hybrid_blurred" if blurred else "hybrid_bow"
    return retriever


def build_blurred_baseline_retriever(limit: int = 20):
    """Plain-vector baseline on the code-blurred fake (same setup as
    the hybrid arm, minus the graph — the exit-criterion comparison)."""
    from runtime.retrieval_baseline import baseline_search

    embeddings = CodeBlurredFakeEmbeddings()
    client = FakeQdrantClient(embeddings)

    def retrieve(question: str):
        docs = baseline_search(question, limit=limit, embeddings=embeddings, client=client)
        ranked_ids = [doc["doc_id"] for doc in docs]
        trace = {
            "scores": {doc["doc_id"]: round(doc.get("score", 0.0), 4) for doc in docs},
            "fake": True,
            "embedding": "code-blurred-bow (digit tokens dropped)",
            "limit": limit,
        }
        return ranked_ids, trace

    retrieve.__name__ = "plain_vector_blurred_baseline"
    return retrieve
