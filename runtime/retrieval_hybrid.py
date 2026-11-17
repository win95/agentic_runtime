"""Hybrid retriever — the locked formula (report §4.7).

It does NOT change ``infra.qdrant_client.search_documents`` or the
``retrieval_context`` hook — those keep their current behaviour
(see ``docs/hybrid-retrieval-phase1.md`` for the follow-up wiring
step, opt-in behind mission config).

All data sources are INJECTED (see ``build_hybrid_retriever``):
real LiteLLM query embeddings, a real Qdrant client, and the live
Neo4j graph. There are no fake embeddings, no fake client, no
demo corpus and no demo adjacency anywhere in this module.

Pipeline per question:
1. Vector arm: ``search_documents`` (real code path) top-40 dense,
   via the injected embeddings + client.
2. Gazetteer seeds (``runtime/retrieval_graph.py``) from the question
   — dynamic cache only (see ``refresh_gazetteer_codes``).
3. Graph arm: 1-hop expansion over the supplied adjacency, or
   ``expand_via_graph_query`` (live); every doc linked to a seed or
   a 1-hop neighbour becomes a candidate, with G1/G2/G4/G5 path
   features plus G6 coverage, G7 PPR and G8 hub. On the
   live path G6/G7/G8 run over ``build_live_adjacency`` (real edges
   out to the second hop), degrading to the expansion-only adjacency
   if the live build fails — never over a fabricated adjacency.
4. Candidates = vector top-40 UNION graph-linked docs, deduped.
5. RRF (k=60) over vector rank + graph rank -> top-30.
6. Fusion (``runtime/retrieval_fusion.py``) -> guardrails -> top-6.
7. ``exact`` = 1 when a gazetteer code from the question appears
   verbatim in the candidate's title/content.

Graph score: the locked full formula
``0.45 * minmax_log(PPR) + 0.30 * z(path_term) + 0.25 * coverage``
(``runtime/retrieval_graph.py::graph_scores``), with the
hub penalty inside the path term. ``w_ppr=0`` is the PPR ablation
flag: it zeroes the PPR sub-weight with no renormalising.

Trace per query (logged by the harness): seeds, expansion nodes
returned / truncated, graph_missing, per-candidate features and
scores, guardrail flags.
"""

from __future__ import annotations

import logging
from enum import Enum

from runtime.retrieval_fusion import (
    W_EXACT,
    W_GRAPH,
    W_VEC,
    fuse_candidates,
    rrf_prefusion,
)
from runtime.retrieval_graph import (
    W_PPR,
    build_expansion_adjacency,
    build_live_adjacency,
    expand_one_hop,
    expand_via_graph_query,
    gazetteer_seeds,
    graph_features,
    graph_scores,
)
from runtime.retrieval_weights import (
    DEFAULT_SEED_CONFIDENCE,
    TRACE_SCORE_PRECISION,
)

logger = logging.getLogger(__name__)


class AdjacencySource(str, Enum):
    """Where a candidate subgraph's adjacency came from.

    ``LIVE``: ``build_live_adjacency`` over real Neo4j edges (out to
    the second hop). ``EXPANSION_ONLY``: the fallback adjacency built
    from the expansion alone (``build_expansion_adjacency``) when the
    live build failed or came back empty. One spelling for the trace
    value ``graph_adjacency_source`` — ``retrieval_live.py`` and the
    plan path pass these members, never raw strings.
    """

    LIVE = "live"
    EXPANSION_ONLY = "expansion-only"

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
    """Dense + 1-hop graph + deterministic fusion.

    All data sources are INJECTED — there are no built-in
    defaults, no demo corpus and no fake embeddings/client.
    ``embeddings`` and ``client`` are required (a missing one
    raises ``ValueError`` naming it). ``adjacency`` is the graph
    substrate for the non-live path; it is required unless the
    live path applies (``use_live_graph`` or a ``graph_store``),
    in which case it defaults to ``{}`` and the per-query
    substrate is built from real store data
    (``build_live_adjacency``, degrading to
    ``build_expansion_adjacency``). ``doc_counts`` /
    ``doc_links`` / ``doc_texts`` default to empty when not
    supplied; the live factory supplies callables reading the
    real Qdrant client.
    """

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
        w_ppr: float = W_PPR,
    ):
        if embeddings is None:
            raise ValueError(
                "HybridRetriever requires 'embeddings' (an object with "
                "embed_query); no default embeddings exist."
            )
        if client is None:
            raise ValueError(
                "HybridRetriever requires 'client' (a Qdrant client); "
                "no default client exists."
            )
        self.embeddings = embeddings
        self.client = client
        if adjacency is None:
            if use_live_graph or graph_store is not None:
                # Live path: the per-query substrate is built from
                # real store data in _candidate_rows; an empty
                # adjacency is only the pre-expansion placeholder.
                adjacency = {}
            else:
                raise ValueError(
                    "HybridRetriever requires 'adjacency' (the graph "
                    "substrate for expand_one_hop), or use_live_graph=True "
                    "with a graph_store so the substrate is built live."
                )
        self.adjacency = adjacency
        self.use_live_graph = use_live_graph
        self.graph_store = graph_store
        self.weights = {"w_vec": w_vec, "w_graph": w_graph, "w_exact": w_exact}
        # PPR ablation flag: w_ppr=0 zeroes the PPR sub-weight
        # (no renormalising — see graph_scores).
        self.w_ppr = w_ppr
        # Live sources for the three tables below. Each accepts
        # a precomputed dict or a callable; a callable is resolved
        # LAZILY (once, cached, on the first query that needs it)
        # and never at construction, so construction never touches
        # Qdrant. The live factory supplies callables reading the
        # real client (see ``build_hybrid_retriever``). When a
        # source is not supplied it defaults to EMPTY (see
        # ``_resolved_doc_counts`` / ``_resolved_doc_links`` /
        # ``_docs``) — there is no demo-table fallback.
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
        if isinstance(doc_texts, dict):
            self._docs = dict(doc_texts)
        else:  # callable: fetched per query, batched (see _fetch_doc_texts)
            # None means no precomputed texts: {} — the live factory
            # supplies a callable; there is no demo-corpus fallback.
            self._docs = {}

    def _resolved_doc_counts(self) -> dict[str, int]:
        """Doc counts per node, from the injected source ({} when none)."""
        if self._doc_counts_cache is not None:
            return self._doc_counts_cache
        source = self._doc_counts_source
        if source is None:
            self._doc_counts_cache = {}
        elif callable(source):
            self._doc_counts_cache = dict(source() or {})
        else:
            self._doc_counts_cache = dict(source)
        return self._doc_counts_cache

    def _resolved_doc_links(self) -> dict[str, list[str]]:
        """Doc -> node ids, from the injected source ({} when none)."""
        if self._doc_links_cache is not None:
            return self._doc_links_cache
        source = self._doc_links_source
        if source is None:
            self._doc_links_cache = {}
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

        Only the callable source fetches anything; a missing
        source and the precomputed-dict source return {} without
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
        """Candidate doc: dict texts, vector payload, fetched payload, else stub."""
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

    def _candidate_rows(
        self,
        question: str,
        seeds: list[dict] | None = None,
        *,
        expansion: dict | None = None,
        adjacency: dict | None = None,
        adjacency_source: str | None = None,
    ):
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
        B3: ``expansion`` / ``adjacency`` let a caller that already
        built them (``hybrid_documents`` on the plan path) pass
        them in instead of rebuilding the same seeds +
        expansion + live-adjacency a second time. When given,
        they are used as-is (the caller built them from the same
        question's seeds); ``adjacency_source`` names which build
        the adjacency came from (``"live"`` /
        ``"expansion-only"``) so the trace marking matches the
        build-it-here path. ``None`` (the default) rebuilds them
        here, exactly as before.
        """
        from infra.qdrant_client import search_documents

        # B14: slim vector fetch — doc_id + node_ids (+ title) only,
        # no content. The vector arm ranks 40 docs but fusion keeps
        # 6, so shipping up to 10,000 chars of content per doc for
        # all 40 is pure waste; content is fetched only for the docs
        # that survive (RRF top-30 below, then top-6 in retrieve).
        vector_docs = search_documents(
            question, limit=VECTOR_LIMIT, embeddings=self.embeddings, client=self.client,
            slim=True,
        )
        vec_by_id = {d["doc_id"]: d for d in vector_docs}
        vec_scores = {d["doc_id"]: float(d.get("score", 0.0)) for d in vector_docs}

        seeds = seeds if seeds is not None else gazetteer_seeds(question)
        if expansion is None:
            if self.use_live_graph:
                expansion = expand_via_graph_query(seeds, store=self.graph_store)
            else:
                expansion = expand_one_hop(seeds, self.adjacency)
        graph_missing = not seeds or not expansion.get("nodes")

        # Substrate for PPR (G7), coverage (G6) and hub (G8): on the
        # live path these must run over REAL graph edges — a live
        # doc's PPR computed from fabricated edges would be
        # meaningless. The non-live path uses self.adjacency as
        # given. Degradation, live path only:
        # if the live build raises or comes back empty while the
        # 1-hop expansion itself succeeded, fall back to the
        # expansion-only adjacency (seeds + expansion nodes + their
        # seed->node edges — real data, partial), never to the demo
        # adjacency and never a crash. If the expansion itself is
        # empty, the existing graph_missing path applies unchanged
        # (no live build is attempted).
        graph_adjacency = self.adjacency
        if self.use_live_graph and not graph_missing:
            if adjacency is not None:
                # B3: prebuilt by the caller (hybrid_documents plan
                # path) — reuse it instead of rebuilding the same
                # live adjacency a second time.
                graph_adjacency = adjacency
                try:
                    source = (
                        AdjacencySource(adjacency_source)
                        if adjacency_source
                        else AdjacencySource.LIVE
                    )
                except ValueError:
                    logger.warning(
                        "unknown adjacency_source %r; treating as live",
                        adjacency_source,
                    )
                    source = AdjacencySource.LIVE
                expansion["graph_adjacency_source"] = source.value
            else:
                try:
                    live_adjacency = build_live_adjacency(seeds, expansion, store=self.graph_store)
                except Exception as exc:  # noqa: BLE001 — live graph is advisory
                    logger.warning("live adjacency build failed: %s", exc)
                    live_adjacency = {}
                if live_adjacency:
                    graph_adjacency = live_adjacency
                    expansion["graph_adjacency_source"] = AdjacencySource.LIVE.value
                else:
                    graph_adjacency = build_expansion_adjacency(seeds, expansion)
                    expansion["graph_adjacency_source"] = AdjacencySource.EXPANSION_ONLY.value
                    logger.warning(
                        "live adjacency empty/failed; using expansion-only adjacency (%d nodes)",
                        len(graph_adjacency),
                    )
            expansion["graph_adjacency_nodes"] = len(graph_adjacency)

        # PPR (once per query, over the candidate subgraph —
        # the live adjacency, <=~200 nodes by design).
        # Seeds weigh confidence * specificity; a seed in no linked
        # document is maximally specific (weight 1.0).
        ppr_scores: dict = {}
        if not graph_missing:
            from runtime.retrieval_ppr import personalized_pagerank, seed_specificity

            spec = seed_specificity(seeds, self._resolved_doc_counts())
            seed_weights = {
                s["key"]: float(s.get("confidence", DEFAULT_SEED_CONFIDENCE))
                * spec[s["key"]]
                for s in seeds
            }
            ppr_scores = personalized_pagerank(graph_adjacency, seeds, seed_weights=seed_weights)

        # Graph-linked candidate docs: docs whose linked nodes touch a
        # seed or a 1-hop neighbour, plus docs that ARE expansion nodes.
        # Links come from real Qdrant payload node_ids
        # (see _resolved_doc_links).
        graph_doc_ids: set[str] = set()
        for doc_id, node_ids in self._resolved_doc_links().items():
            linked = set(node_ids)
            if linked & {s["key"] for s in seeds}:
                graph_doc_ids.add(doc_id)
            elif linked & set(expansion.get("nodes", {})):
                graph_doc_ids.add(doc_id)
        # Payloads for graph-only candidates (not in the vector
        # results): title/content/node_ids for the exact feature and
        # the graph features. Fetched in one batch; with no callable
        # source this is a no-op.
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
        if graph_missing:
            gscores = [0.0 for _ in feature_rows]
        else:
            gscores = graph_scores(feature_rows, w_ppr=self.w_ppr)
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
        # B14: the slim vector fetch carries no content, but the
        # exact feature (0.10) reads title+content — a vector doc
        # whose code lives only in its content would score exact=0
        # and fusion would change. RRF uses vec/graph ranks only,
        # so its top-30 is identical with or without content:
        # fetch content for exactly that set (one batched call,
        # cached — a doc fetched here is never re-fetched for a
        # variant or for the top-6 fill), recompute exact on the
        # filled text, and set title/content on those rows. Docs
        # outside the top-30 never reach fusion, so their exact is
        # never scored. Content is cut to 4,000 chars via
        # clean_text_fields — the same cut the plain
        # repositories.search_documents path applies.
        self._fill_candidate_contents(question, candidates, vec_by_id, seeds=seeds)
        return candidates, seeds, expansion, graph_missing

    def _fill_candidate_contents(
        self, question: str, candidates: list[dict], vec_by_id: dict, *, seeds: list[dict]
    ) -> None:
        """Fill title/content for the RRF top-30 and recompute exact (B14)."""
        from runtime.security import clean_text_fields

        top_ids = [c["doc_id"] for c in rrf_prefusion(candidates)]
        if not top_ids:
            return
        self._fetch_doc_texts(top_ids)
        by_id = {c["doc_id"]: c for c in candidates}
        for doc_id in top_ids:
            row = by_id.get(doc_id)
            if row is None:
                continue
            # The fetched payload is authoritative: _doc_for prefers
            # the slim vector row (title only) over the fetched
            # text, which would leave content empty here.
            doc = (
                self._fetched_texts.get(doc_id)
                or self._docs.get(doc_id)
                or vec_by_id.get(doc_id)
                or {}
            )
            if not doc.get("content") and not doc.get("title"):
                continue
            row["title"] = row.get("title") or doc.get("title") or ""
            row["content"] = clean_text_fields(
                {"content": doc.get("content") or ""}
            )["content"]
            row["exact"] = _exact_match(question, doc, seeds=seeds)

    def _fill_top_contents(self, final: list[dict], rows: list[dict]) -> None:
        """Fetch content for the fused top-6 and set it on their rows (B14).

        The final ranking may include a doc whose content was not
        fetched yet (merged plan rows take their content from the
        best-graph row, which may not have been in its variant's
        RRF top-30). One batched ``_fetch_doc_texts`` call,
        cached — rows that already carry content are untouched.
        Content is cut to 4,000 chars, same as the plain path.
        """
        from runtime.security import clean_text_fields

        top_ids = [c["doc_id"] for c in final]
        if not top_ids:
            return
        by_id = {row["doc_id"]: row for row in rows}
        wanted = [d for d in top_ids if not (by_id.get(d) or {}).get("content")]
        if wanted:
            self._fetch_doc_texts(wanted)
        for doc_id in top_ids:
            row = by_id.get(doc_id)
            if row is None or row.get("content"):
                continue
            doc = self._fetched_texts.get(doc_id) or self._docs.get(doc_id) or {}
            if not doc.get("content") and not doc.get("title"):
                continue
            row["title"] = row.get("title") or doc.get("title") or ""
            row["content"] = clean_text_fields(
                {"content": doc.get("content") or ""}
            )["content"]

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
        # The plan already fetched this question's candidate rows
        # for gating (build_query_plan) — reuse them instead of
        # fetching the same question twice. The cache applies only
        # to the question the plan was built for; a plan built by
        # hand (or for another question) falls through to a fresh
        # fetch with the plan's seeds.
        cached = getattr(query_plan, "base_candidate_rows", None)
        if cached is not None and getattr(query_plan, "base_question", None) == question:
            base_rows, seeds, expansion, graph_missing = cached
        else:
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
            # B14/B5: the fused top-6 carry content; stash the rows
            # so hybrid_documents reads them without re-fetching.
            self._fill_top_contents(base_final, base_rows)
            self._last_candidate_rows = base_rows
            # B8: stash the expansion so the SessionStart hook's
            # relations section can reuse these 1-hop neighbours
            # instead of re-fetching them from Neo4j.
            self._last_expansion = expansion
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
        # B14/B5: the fused top-6 carry content; stash the rows so
        # hybrid_documents reads them without re-fetching.
        self._fill_top_contents(merged_final, merged)
        self._last_candidate_rows = merged
        # B8: stash the base expansion (see above).
        self._last_expansion = expansion
        return ranked_ids, trace

    def _trace(self, candidates, final, seeds, expansion, graph_missing):
        """The standard retrieve trace (shared by both paths)."""
        trace = {
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
                    "vec": round(c["vec_score"], TRACE_SCORE_PRECISION),
                    "graph": round(c["graph_score"], TRACE_SCORE_PRECISION),
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
        # Which adjacency PPR/coverage/hub ran over. Absent
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
        # B14/B5: the fused top-6 carry content (filled in
        # _candidate_rows for the RRF top-30); stash the rows so
        # hybrid_documents reads them without re-fetching.
        self._fill_top_contents(final, candidates)
        self._last_candidate_rows = candidates
        # B8: stash the expansion so the SessionStart hook's
        # relations section can reuse these 1-hop neighbours
        # instead of re-fetching them from Neo4j.
        self._last_expansion = expansion
        return ranked_ids, trace


def trace_feature_rows(candidates: list[dict], final: list[dict]) -> list[dict]:
    """Full per-candidate feature rows for one query (Phase 3 LTR logging).

    One row per candidate in the fused set (pre-fusion — every
    fused candidate, not just the final top-6), with the
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


def  build_hybrid_retriever(
    *,
    client=None,
    graph_store=None,
    embeddings=None,
    w_ppr: float = W_PPR,
    **kwargs,
):
    """Hybrid retriever over the REAL stores — THE factory.

    Real LiteLLM query embeddings + real Qdrant + live Neo4j graph.
    This is the only factory: there is no fake/blurred variant and
    no demo corpus behind it.

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
    leaves the gazetteer with an empty cache (it links nothing
    rather than falling back to any built-in vocabulary).

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
        w_ppr=w_ppr,
        **kwargs,
    )
    retriever.__name__ = "hybrid"
    return retriever
