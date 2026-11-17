"""Live hybrid retrieval for production call sites (Seams A + B).

One shared helper — ``hybrid_documents`` — over the hybrid retriever:
build the retriever, optionally build a QueryPlan, retrieve, and fetch
the ranked docs' payloads. Both production call sites use it:

- the ``retrieval_context`` hook (Seam A): the documents section of
  the pre-retrieval block, assembled with ``assemble_context``;
- the ``knowledge_retrieval`` capability node (Seam B): the document
  evidence rows.

The helper raises on ANY failure (bad embedding config, Qdrant down,
...). It never falls back and never invents docs — the callers decide:
the hook falls back to plain vector search, the capability node
records error evidence. No plan is built unless ``query_plan=True``
(the capability node never passes it: the deterministic spine holds
no LLM call).
"""

from __future__ import annotations

import logging
import threading

from app.config import settings
from infra.qdrant_client import get_document_payloads
from runtime.retrieval_graph import (
    build_expansion_adjacency,
    build_live_adjacency,
    expand_via_graph_query,
    gazetteer_seeds,
)
from runtime.retrieval_hybrid import AdjacencySource, build_hybrid_retriever
from runtime.retrieval_query_plan import build_query_plan

logger = logging.getLogger(__name__)


# --- One retriever per execution (B2) --------------------------------
# Both retrieval seams run in the same worker process for one
# execution — the capability node first (hybrid mode: the flow
# reaches the capability graph before the deep agent) or the
# SessionStart hook first (autonomous / deep-agent-only flow).
# Whichever runs first builds the retriever and stashes it here
# keyed by execution_id; the other reuses it — including the
# resolved gazetteer (B7), the doc_counts/doc_links instance
# caches (B6) and the fetched payloads cache, so the second seam
# never re-scrolls Qdrant or re-fetches docs the first seam
# already fetched. Both seams keep their own question and their
# own evidence contract; only the SUBSTRATE is shared (candidate
# rows are reused only where the question matches — see
# HybridRetriever._last_candidate_rows / the plan's
# base_candidate_rows). The registry is bounded (32 entries,
# FIFO eviction) and dropped when an execution finishes (see
# app/worker/tasks.py::run_execution), so a long-running worker
# never grows without bound. A missing/empty execution_id builds
# a fresh retriever (unchanged behaviour for direct callers).
_EXECUTION_RETRIEVERS: dict[str, object] = {}
_EXECUTION_RETRIEVERS_LOCK = threading.Lock()
_MAX_EXECUTION_RETRIEVERS = settings.retrieval_max_execution_retrievers


def get_execution_retriever(execution_id: str | None):
    """The stashed retriever for one execution, building it on first use."""
    if not execution_id:
        return build_hybrid_retriever()
    with _EXECUTION_RETRIEVERS_LOCK:
        retriever = _EXECUTION_RETRIEVERS.get(execution_id)
        if retriever is None:
            retriever = build_hybrid_retriever()
            _EXECUTION_RETRIEVERS[execution_id] = retriever
            while len(_EXECUTION_RETRIEVERS) > _MAX_EXECUTION_RETRIEVERS:
                _EXECUTION_RETRIEVERS.pop(next(iter(_EXECUTION_RETRIEVERS)))
        return retriever


def get_stashed_execution_retriever(execution_id: str | None):
    """The stashed retriever for one execution, or None (B8).

    Unlike ``get_execution_retriever`` this NEVER builds one:
    the hook's relations section only reuses an expansion the
    documents section already built — when nothing is stashed
    it falls back to Neo4j instead of building a retriever
    (which would touch Qdrant) just to look for an expansion.
    """
    if not execution_id:
        return None
    with _EXECUTION_RETRIEVERS_LOCK:
        return _EXECUTION_RETRIEVERS.get(execution_id)


def drop_execution_retriever(execution_id: str | None) -> None:
    """Drop the stashed retriever for a finished execution (B2)."""
    if not execution_id:
        return
    with _EXECUTION_RETRIEVERS_LOCK:
        _EXECUTION_RETRIEVERS.pop(execution_id, None)


def triples_from_expansion(expansion: dict, doc_links: dict) -> list[dict]:
    """Verbalisable triples from a 1-hop expansion.

    One triple per expanded edge (seed ``-rel->`` node). The source
    doc is the first document linked to either endpoint; when neither
    endpoint has a linked doc the key is omitted and
    ``assemble_context`` falls back to the subject key.
    """
    node_to_docs: dict[str, list[str]] = {}
    for doc_id, node_ids in (doc_links or {}).items():
        for node_id in node_ids or []:
            node_to_docs.setdefault(node_id, []).append(doc_id)
    triples = []
    for to_key, node in (expansion.get("nodes") or {}).items():
        from_seed = node.get("from_seed") or ""
        rel = node.get("rel") or ""
        source = (
            node_to_docs.get(to_key) or node_to_docs.get(from_seed) or [None]
        )[0]
        triple = {
            "subject": from_seed,
            "relation": rel,
            "object": to_key,
        }
        if source is not None:
            triple["source_doc_id"] = source
        triples.append(triple)
    return triples


def hybrid_documents(
    question: str,
    *,
    query_plan: bool = False,
    retriever=None,
    execution_id: str | None = None,
) -> dict:
    """One hybrid retrieval + payload fetch. Returns ``{"docs",
    "ranked", "trace", "plan", "triples"}``.

    ``docs`` are rows shaped like ``repositories.search_documents``
    (``doc_id`` / ``title`` / ``content``) so existing consumers keep
    their contract. ``triples`` is non-empty only on the plan path
    (the no-plan path never builds an expansion to source them from).
    Raises on any failure — the caller decides the fallback.
    B2: when ``retriever`` is not given and ``execution_id`` is,
    the execution's stashed retriever is reused (built on first
    use); without an execution_id a fresh retriever is built,
    exactly as before.
    """
    live_retriever = retriever or get_execution_retriever(execution_id)
    plan = None
    expansion: dict = {"nodes": {}, "truncated": False, "nodes_returned": 0}
    doc_links: dict = {}
    if hasattr(live_retriever, "_resolved_doc_links"):
        doc_links = live_retriever._resolved_doc_links() or {}
    if query_plan:
        seeds = gazetteer_seeds(question)
        # B3: the expansion + live-adjacency built here are passed
        # into build_query_plan (and from there into the gating
        # _candidate_rows call) instead of being rebuilt there —
        # one build per question, not two. The retriever's own
        # graph store is used, so the prebuilt substrate is the
        # same one _candidate_rows would have built.
        store = getattr(live_retriever, "graph_store", None)
        expansion = expand_via_graph_query(seeds, store=store)
        try:
            adjacency = build_live_adjacency(seeds, expansion, store=store)
            adjacency_source = AdjacencySource.LIVE.value
        except Exception:  # noqa: BLE001 — degraded substrate, never fatal
            adjacency = build_expansion_adjacency(seeds, expansion)
            adjacency_source = AdjacencySource.EXPANSION_ONLY.value
        plan = build_query_plan(
            question,
            retriever=live_retriever,
            embeddings=getattr(live_retriever, "embeddings", None),
            adjacency=adjacency,
            doc_links=doc_links,
            seeds=seeds,
            expansion=expansion,
            adjacency_source=adjacency_source,
        )
    ranked, trace = live_retriever.retrieve(question, query_plan=plan)
    # B5: build docs from the retriever's candidate rows (they
    # already carry title/content for the fused top-6 — see
    # HybridRetriever._fill_candidate_contents /
    # _fill_top_contents) and fetch payloads ONLY for ranked ids
    # missing from those rows (a retriever without
    # _last_candidate_rows, or an id the rows don't cover). An
    # unknown id stays absent, exactly as before.
    rows = getattr(live_retriever, "_last_candidate_rows", None) or []
    by_id = {row["doc_id"]: row for row in rows}
    missing = [
        doc_id
        for doc_id in ranked
        if not (by_id.get(doc_id) or {}).get("content")
    ]
    payloads = get_document_payloads(missing) if missing else {}
    docs = [
        {
            "doc_id": doc_id,
            "title": (by_id.get(doc_id) or {}).get("title")
            or (payloads.get(doc_id) or {}).get("title")
            or doc_id,
            "content": (by_id.get(doc_id) or {}).get("content")
            or (payloads.get(doc_id) or {}).get("content")
            or "",
        }
        for doc_id in ranked
    ]
    triples = (
        triples_from_expansion(expansion, doc_links) if query_plan else []
    )
    logger.info(
        "hybrid_documents: %d docs, %d triples, graph_missing=%s, "
        "plan=%s",
        len(docs),
        len(triples),
        (trace or {}).get("graph_missing"),
        plan.intent if plan is not None else None,
    )
    return {
        "docs": docs,
        "ranked": ranked,
        "trace": trace,
        "plan": plan,
        "triples": triples,
    }
