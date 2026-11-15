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

from infra.qdrant_client import get_document_payloads
from runtime.retrieval_graph import (
    build_expansion_adjacency,
    build_live_adjacency,
    expand_via_graph_query,
    gazetteer_seeds,
)
from runtime.retrieval_hybrid import build_hybrid_retriever
from runtime.retrieval_query_plan import build_query_plan

logger = logging.getLogger(__name__)


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
    question: str, *, query_plan: bool = False, retriever=None
) -> dict:
    """One hybrid retrieval + payload fetch. Returns ``{"docs",
    "ranked", "trace", "plan", "triples"}``.

    ``docs`` are rows shaped like ``repositories.search_documents``
    (``doc_id`` / ``title`` / ``content``) so existing consumers keep
    their contract. ``triples`` is non-empty only on the plan path
    (the no-plan path never builds an expansion to source them from).
    Raises on any failure — the caller decides the fallback.
    """
    live_retriever = retriever or build_hybrid_retriever()
    plan = None
    expansion: dict = {"nodes": {}, "truncated": False, "nodes_returned": 0}
    doc_links: dict = {}
    if hasattr(live_retriever, "_resolved_doc_links"):
        doc_links = live_retriever._resolved_doc_links() or {}
    if query_plan:
        seeds = gazetteer_seeds(question)
        expansion = expand_via_graph_query(seeds)
        try:
            adjacency = build_live_adjacency(seeds, expansion)
        except Exception:  # noqa: BLE001 — degraded substrate, never fatal
            adjacency = build_expansion_adjacency(seeds, expansion)
        plan = build_query_plan(
            question,
            retriever=live_retriever,
            embeddings=getattr(live_retriever, "embeddings", None),
            adjacency=adjacency,
            doc_links=doc_links,
        )
    ranked, trace = live_retriever.retrieve(question, query_plan=plan)
    payloads = get_document_payloads(ranked)
    docs = [
        {
            "doc_id": doc_id,
            "title": (payloads.get(doc_id) or {}).get("title") or doc_id,
            "content": (payloads.get(doc_id) or {}).get("content") or "",
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
