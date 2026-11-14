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
   plus G6 coverage, G7 PPR and G8 hub (Phase 2, default on). On the
   live path G6/G7/G8 run over ``build_live_adjacency`` (real edges
   out to the second hop), degrading to the expansion-only adjacency
   if the live build fails — never over the demo adjacency.
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
    build_expansion_adjacency,
    build_live_adjacency,
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


def _exact_match(question: str, doc: dict, seeds: list[dict] | None = None) -> int:
    """1 if a code the question names appears verbatim in the doc.

    Codes are the gazetteer seeds (asset/component/building/doc
    codes) plus fault codes found in the question text (``F-204`` —
    the codes the docs actually print). Only codes the question
    names count, never the whole fault table. ``seeds`` overrides
    the gazetteer lookup (the Phase 4 plan seam passes the plan's
    seeds); ``None`` re-derives them from the question, as before.
    """
    import re

    if seeds is None:
        seeds = gazetteer_seeds(question)
    question_codes = {s["key"].lower() for s in seeds}
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
        doc_counts=None,
        doc_texts=None,
        doc_links=None,
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
        # Live sources for the three demo tables below. Each accepts
        # a precomputed dict or a callable; a callable is resolved
        # LAZILY (once, cached, on the first query that needs it)
        # and never at construction, so the fake path (all three
        # None) never touches Qdrant. The live factory supplies
        # callables reading the real client (see
        # ``build_live_hybrid_retriever``).
        # - doc_counts: {node key: n docs linked} for seed
        #   specificity (``runtime/retrieval_ppr.py::seed_specificity``:
        #   a missing/empty count is maximally specific, so an empty
        #   store degrades to {} here, never a crash).
        # - doc_links: {doc id: [node ids]} for the graph arm.
        # - doc_texts: {doc id: payload} OR a callable taking the
        #   doc ids to fetch (``[doc_id] -> {doc_id: payload}``) —
        #   the batch payload fetch for graph-only candidates.
        self._doc_counts_source = doc_counts
        self._doc_counts_cache: dict[str, int] | None = None
        self._doc_links_source = doc_links
        self._doc_links_cache: dict[str, list[str]] | None = None
        self._doc_texts_source = doc_texts
        self._fetched_texts: dict[str, dict] = {}
        self._missing_texts: set[str] = set()
        if doc_texts is None:
            # Doc texts for the graph arm / exact feature (fake corpus).
            from seeds.demo_data import DOCUMENTS

            self._docs = {d["id"]: d for d in DOCUMENTS}
        elif isinstance(doc_texts, dict):
            self._docs = dict(doc_texts)
        else:  # callable: fetched per query, batched (see _fetch_doc_texts)
            self._docs = {}

    def _resolved_doc_counts(self) -> dict[str, int]:
        """Doc counts per node, demo table unless a live source was given."""
        if self._doc_counts_cache is not None:
            return self._doc_counts_cache
        source = self._doc_counts_source
        if source is None:
            counts: dict[str, int] = {}
            for links in DOCUMENT_LINKS.values():
                for node_key in links["node_ids"]:
                    counts[node_key] = counts.get(node_key, 0) + 1
            self._doc_counts_cache = counts
        elif callable(source):
            self._doc_counts_cache = dict(source() or {})
        else:
            self._doc_counts_cache = dict(source)
        return self._doc_counts_cache

    def _resolved_doc_links(self) -> dict[str, list[str]]:
        """Doc -> node ids, demo table unless a live source was given."""
        if self._doc_links_cache is not None:
            return self._doc_links_cache
        source = self._doc_links_source
        if source is None:
            self._doc_links_cache = {
                doc_id: list(links["node_ids"])
                for doc_id, links in DOCUMENT_LINKS.items()
            }
        elif callable(source):
            self._doc_links_cache = {
                doc_id: list(node_ids) for doc_id, node_ids in (source() or {}).items()
            }
        else:
            self._doc_links_cache = {
                doc_id: list(node_ids) for doc_id, node_ids in source.items()
            }
        return self._doc_links_cache

    def _fetch_doc_texts(self, doc_ids: list[str]) -> dict[str, dict]:
        """Batch-fetch payloads for doc ids missing from _docs / vector results.

        Only the live callable source fetches anything; the fake path
        (no source) and the precomputed-dict source return {} without
        a call. Ids already fetched, or already known missing, are
        not re-fetched. A fetch failure propagates (a Qdrant failure
        surfaces as RuntimeError naming the store, never a silent
        empty result); ids the store simply does not have stay
        missing and contribute no text feature.
        """
        wanted = [
            d
            for d in dict.fromkeys(doc_ids)
            if d not in self._docs and d not in self._fetched_texts and d not in self._missing_texts
        ]
        if not wanted:
            return {}
        source = self._doc_texts_source
        if source is None or isinstance(source, dict):
            return {}
        fetched = dict(source(wanted) or {})
        self._fetched_texts.update(fetched)
        self._missing_texts.update(set(wanted) - set(fetched))
        return fetched

    def _doc_for(self, doc_id: str, vec_by_id: dict) -> dict:
        """Candidate doc: demo/dict texts, vector payload, fetched payload, else stub."""
        return (
            self._docs.get(doc_id)
            or vec_by_id.get(doc_id)
            or self._fetched_texts.get(doc_id)
            or {"doc_id": doc_id}
        )

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

    def _candidate_rows(self, question: str, seeds: list[dict] | None = None):
        """Per-question candidate rows (pre-RRF, pre-fusion).

        Returns ``(candidates, seeds, expansion, graph_missing)``.
        Each candidate carries the full §4.1 feature row
        (``graph_g1_min_hops`` … ``graph_g8_transit_key``) plus
        ``vec_score`` / ``graph_score`` / ``exact`` / ``g5_direct`` /
        ``best_path_rels`` — everything ``fuse_candidates`` needs.
        Phase 3 expansion calls this once per variant and merges the
        rows before ranking; ``retrieve`` calls it once.
        ``seeds`` overrides the gazetteer lookup (the Phase 4 plan
        seam passes the plan's linked nodes); ``None`` (the
        default) re-derives them from the question, exactly as
        before — the no-plan path is unchanged. An explicitly
        empty list means "no confident link" (vector-only).
        """
        from infra.qdrant_client import search_documents

        vector_docs = search_documents(
            question, limit=VECTOR_LIMIT, embeddings=self.embeddings, client=self.client
        )
        vec_by_id = {d["doc_id"]: d for d in vector_docs}
        vec_scores = {d["doc_id"]: float(d.get("score", 0.0)) for d in vector_docs}

        seeds = seeds if seeds is not None else gazetteer_seeds(question)
        if self.use_live_graph:
            expansion = expand_via_graph_query(seeds, store=self.graph_store)
        else:
            expansion = expand_one_hop(seeds, self.adjacency)
        graph_missing = not seeds or not expansion.get("nodes")

        # Substrate for PPR (G7), coverage (G6) and hub (G8): on the
        # live path these must run over REAL graph edges, not the
        # demo adjacency — a live doc's PPR computed from demo edges
        # would be meaningless. The fake path keeps self.adjacency
        # (byte-identical behaviour). Degradation, live path only:
        # if the live build raises or comes back empty while the
        # 1-hop expansion itself succeeded, fall back to the
        # expansion-only adjacency (seeds + expansion nodes + their
        # seed->node edges — real data, partial), never to the demo
        # adjacency and never a crash. If the expansion itself is
        # empty, the existing graph_missing path applies unchanged
        # (no live build is attempted).
        graph_adjacency = self.adjacency
        if self.use_live_graph and not graph_missing:
            try:
                live_adjacency = build_live_adjacency(seeds, expansion, store=self.graph_store)
            except Exception as exc:  # noqa: BLE001 — live graph is advisory
                logger.warning("live adjacency build failed: %s", exc)
                live_adjacency = {}
            if live_adjacency:
                graph_adjacency = live_adjacency
                expansion["graph_adjacency_source"] = "live"
            else:
                graph_adjacency = build_expansion_adjacency(seeds, expansion)
                expansion["graph_adjacency_source"] = "expansion-only"
                logger.warning(
                    "live adjacency empty/failed; using expansion-only adjacency (%d nodes)",
                    len(graph_adjacency),
                )
            expansion["graph_adjacency_nodes"] = len(graph_adjacency)

        # Phase 2 PPR (once per query, over the candidate subgraph —
        # here the whole demo adjacency, <=~200 nodes by design).
        # Seeds weigh confidence * specificity; a seed in no linked
        # document is maximally specific (weight 1.0).
        ppr_scores: dict = {}
        if self.phase2 and not graph_missing:
            from runtime.retrieval_ppr import personalized_pagerank, seed_specificity

            spec = seed_specificity(seeds, self._resolved_doc_counts())
            seed_weights = {
                s["key"]: float(s.get("confidence", 1.0)) * spec[s["key"]] for s in seeds
            }
            ppr_scores = personalized_pagerank(graph_adjacency, seeds, seed_weights=seed_weights)

        # Graph-linked candidate docs: docs whose linked nodes touch a
        # seed or a 1-hop neighbour, plus docs that ARE expansion nodes.
        # Links come from the demo table on the fake path, from real
        # Qdrant payload node_ids on the live path (see _resolved_doc_links).
        graph_doc_ids: set[str] = set()
        for doc_id, node_ids in self._resolved_doc_links().items():
            linked = set(node_ids)
            if linked & {s["key"] for s in seeds}:
                graph_doc_ids.add(doc_id)
            elif linked & set(expansion.get("nodes", {})):
                graph_doc_ids.add(doc_id)
        # Payloads for graph-only candidates (not in the vector
        # results): title/content/node_ids for the exact feature and
        # the graph features. Fetched in one batch; on the fake path
        # this is a no-op (no live source, _docs already has them).
        self._fetch_doc_texts(
            sorted(graph_doc_ids | set(expansion.get("nodes", {})))
        )
        for node_key in expansion.get("nodes", {}):
            if node_key in self._docs or node_key in self._fetched_texts:
                graph_doc_ids.add(node_key)

        candidate_ids = sorted(set(vec_by_id) | graph_doc_ids)
        feature_rows: list[dict] = []
        for doc_id in candidate_ids:
            doc = self._doc_for(doc_id, vec_by_id)
            feature_rows.append(
                graph_features(
                    doc_id,
                    seeds,
                    expansion,
                    graph_adjacency,
                    payload=vec_by_id.get(doc_id) or self._fetched_texts.get(doc_id),
                    ppr_scores=ppr_scores or None,
                )
            )
        if self.phase2 and not graph_missing:
            gscores = phase2_graph_scores(feature_rows, w_ppr=self.w_ppr)
        else:
            gscores = [
                0.0 if graph_missing else graph_score(features)
                for features in feature_rows
            ]
        candidates: list[dict] = []
        for doc_id, features, gscore in zip(candidate_ids, feature_rows, gscores):
            doc = self._doc_for(doc_id, vec_by_id)
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
                "exact": _exact_match(question, doc, seeds=seeds),
                "g5_direct": features["g5_direct"],
                "best_path_rels": features["best_path_rels"],
                "arms": arms,
                **{f"graph_{k}": v for k, v in features.items()},
            })
        return candidates, seeds, expansion, graph_missing

    def retrieve(self, question: str, query_plan=None):
        """Rank docs for one question, optionally through a QueryPlan.

        With ``query_plan=None`` (the default) this is exactly the
        Phase 1–2 path: one ``_candidate_rows`` call, one fusion
        rank, the same trace — byte-identical to before Phase 4.
        With a plan (``runtime/retrieval_query_plan.QueryPlan``):
        the plan's ``linked_nodes`` seed the original question's
        rows instead of ``gazetteer_seeds(question)``, and — only
        when the plan is not gated off — its sub-questions (or,
        failing those, its rewrite variants) are each retrieved
        via ``_candidate_rows``, merged with
        ``merge_candidate_rows`` (the Phase 3 merge path, mean
        vec/graph, max exact) and ranked by the same
        ``_rank_candidates``. Sub-question rows re-derive their
        seeds from the sub-question text (identical to the keys
        the plan stores for them, confidences preserved); the
        original row uses the plan's seeds directly. The trace
        gains a ``query_plan`` block (intent, gating, seeds,
        sub-questions, variants, fallback reason) and an
        ``expansion`` block; every other trace key is as before.
        """
        if query_plan is None:
            return self._retrieve_no_plan(question)
        # Lazy import: retrieval_expansion imports this module at
        # its top level, so a top-level import here would be circular.
        from runtime.retrieval_expansion import merge_candidate_rows

        plan_seeds = list(query_plan.linked_nodes or [])
        base_rows, seeds, expansion, graph_missing = self._candidate_rows(
            question, seeds=plan_seeds
        )
        base_final = self._rank_candidates(base_rows, graph_missing)
        base_ranked = [c["doc_id"] for c in base_final]
        if query_plan.sub_questions:
            expansion_texts = [sq["text"] for sq in query_plan.sub_questions]
            kind = "decomposition"
        elif query_plan.rewrite_variants:
            expansion_texts = list(query_plan.rewrite_variants)
            kind = "multi-query"
        else:
            expansion_texts = []
            kind = "none"
        plan_trace = {
            "intent": query_plan.intent,
            "confidence": query_plan.confidence,
            "gated_off": query_plan.gated_off,
            "fallback_reason": query_plan.fallback_reason,
            "linked_nodes": [s["key"] for s in seeds],
            "sub_questions": [sq["text"] for sq in query_plan.sub_questions],
            "rewrite_variants": list(query_plan.rewrite_variants),
            "related": query_plan.related,
            "shared_nodes": list(query_plan.shared_nodes or []),
            "gating": dict(query_plan.gating or {}),
        }
        if not expansion_texts:
            ranked_ids, trace = base_ranked, self._trace(
                base_rows, base_final, seeds, expansion, graph_missing
            )
            trace["query_plan"] = plan_trace
            trace["expansion"] = {
                "triggered": False,
                "kind": "none",
                "gating": dict(query_plan.gating or {}),
                "variants": [],
            }
            return ranked_ids, trace
        variant_rows = []
        variant_rankings = []
        for text in expansion_texts:
            rows, _s, _e, _gm = self._candidate_rows(text)
            variant_rows.append(rows)
            variant_rankings.append(
                [c["doc_id"] for c in self._rank_candidates(rows, graph_missing)]
            )
        merged = merge_candidate_rows([base_rows, *variant_rows])
        all_rankings = [base_ranked, *variant_rankings]
        for row in merged:
            row["provenance_variants"] = [
                i for i, ranked in enumerate(all_rankings)
                if row["doc_id"] in set(ranked)
            ]
        merged_final = self._rank_candidates(merged, graph_missing)
        ranked_ids = [c["doc_id"] for c in merged_final]
        trace = self._trace(
            merged, merged_final, seeds, expansion, graph_missing
        )
        trace["query_plan"] = plan_trace
        trace["expansion"] = {
            "triggered": True,
            "kind": kind,
            "gating": dict(query_plan.gating or {}),
            "variants": expansion_texts,
        }
        return ranked_ids, trace

    def _trace(self, candidates, final, seeds, expansion, graph_missing):
        """The standard retrieve trace (shared by both paths)."""
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
        # Live path only: which adjacency PPR/coverage/hub ran over.
        # Absent on the fake path (trace byte-identical there) and
        # when the expansion itself was empty (graph_missing).
        if "graph_adjacency_source" in expansion:
            trace["graph_adjacency_source"] = expansion["graph_adjacency_source"]
            trace["graph_adjacency_nodes"] = expansion.get("graph_adjacency_nodes", 0)
        return trace

    def _retrieve_no_plan(self, question: str):
        """The Phase 1–2 retrieve path, untouched by Phase 4."""
        candidates, seeds, expansion, graph_missing = self._candidate_rows(question)
        final = self._rank_candidates(candidates, graph_missing)
        ranked_ids = [c["doc_id"] for c in final]
        trace = self._trace(candidates, final, seeds, expansion, graph_missing)
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


def build_live_hybrid_retriever(
    *,
    client=None,
    graph_store=None,
    embeddings=None,
    phase2: bool = True,
    w_ppr: float = W_PPR,
    **kwargs,
):
    """Live hybrid retriever: real LiteLLM query embeddings + real Qdrant + live Neo4j graph.

    Reads the embedding model from settings — ``EMBEDDING_MODEL_QUERY``
    falling back to ``EMBEDDING_MODEL`` (``app/config.py``); set
    ``EMBEDDING_MODEL`` in ``.env`` to the model name your LiteLLM
    proxy serves. The retriever only embeds QUERIES, so only the
    query-time model is built here; the INDEX model governs seeding
    via the existing ``infra.qdrant_client.embed_texts`` default
    (index-time embedding), and the two models must produce
    compatible vectors (same dimension / vector space — changing the
    index model requires re-indexing Qdrant).

    Construction is lazy: no Qdrant/Neo4j connection is made here —
    ``get_client()`` only builds the client object, and the graph
    store stays ``None`` unless passed (``expand_via_graph_query``
    resolves the default from
    ``ontology/projection/provider.py::get_graph_store`` on first
    use). The dynamic gazetteer is refreshed best-effort through
    ``runtime/retrieval_graph.py::refresh_gazetteer_codes``: on any
    exception (including the function not existing yet) a warning is
    logged and the factory continues with whatever gazetteer state
    results — the refresh itself never raises, and a failed refresh
    leaves the gazetteer in live mode with an empty cache (it links
    nothing rather than falling back to demo codes).

    Fail fast: an empty ``QDRANT_URL`` raises RuntimeError; missing
    embedding config raises ValueError from ``require_embedding_config``.

    Error contract: a graph-store failure degrades to
    ``graph_missing`` and the pure-vector fallback (the existing
    behaviour in ``expand_via_graph_query``); a Qdrant failure
    surfaces as a RuntimeError naming Qdrant (``search_documents``
    and the payload helpers raise it) and is never swallowed into
    empty results.

    NOT wired into any execution path — callers opt in explicitly.
    No scoring/fusion formula, weight, guardrail or gating threshold
    changes here: the live retriever is the same ``HybridRetriever``
    with real sources injected (doc counts / doc links / doc texts
    are read lazily from Qdrant on first use).
    """
    from app.config import settings
    from infra.qdrant_client import (
        count_documents_per_node,
        get_client,
        get_document_links,
        get_document_payloads,
    )
    from runtime.llm_factory import build_embeddings, require_embedding_config

    if not settings.qdrant_url:
        raise RuntimeError(
            "Qdrant is not configured: QDRANT_URL is empty. "
            "Set QDRANT_URL in .env to your Qdrant server URL."
        )
    require_embedding_config()
    live_embeddings = embeddings or build_embeddings(
        model=settings.resolved_embedding_model_query
    )
    live_client = client or get_client()
    if live_client is None:
        raise RuntimeError(
            "Qdrant client could not be built even though QDRANT_URL is set "
            f"({settings.qdrant_url!r}). Check the Qdrant server is reachable."
        )
    try:
        from runtime.retrieval_graph import refresh_gazetteer_codes
    except Exception as exc:  # noqa: BLE001 — refresh is best-effort, never fatal
        logger.warning("dynamic gazetteer refresh unavailable: %s", exc)
    else:
        try:
            refreshed = refresh_gazetteer_codes(store=graph_store)
        except Exception as exc:  # noqa: BLE001 — refresh is best-effort, never fatal
            logger.warning("dynamic gazetteer refresh failed: %s", exc)
        else:
            logger.info("dynamic gazetteer refreshed: %s", refreshed)
    retriever = HybridRetriever(
        embeddings=live_embeddings,
        client=live_client,
        use_live_graph=True,
        graph_store=graph_store,
        doc_counts=lambda: count_documents_per_node(client=live_client),
        doc_links=lambda: get_document_links(client=live_client),
        doc_texts=lambda doc_ids: get_document_payloads(doc_ids, client=live_client),
        phase2=phase2,
        w_ppr=w_ppr,
        **kwargs,
    )
    retriever.__name__ = "hybrid_live"
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
