"""Context assembly — Phase 1 (report §5).

Best-first fill within a fixed token budget, deduped, with graph
facts as short verbalised triples — never raw JSON. Two blocks:
``## Relevant documents`` (chunks, best-first, headed ``[D1] title —
doc_id``) and ``## Related facts from the asset graph`` (verbalised
triples, each with its source doc id in brackets).

Token counting: a cheap estimator, ``chars / 4`` (``estimate_tokens``).
It is an approximation — documented as such; no tokenizer dependency
is added in Phase 1. Live serving must re-measure with the real
tokenizer before trusting the budget edge.

Rules (all enforced here):
- total budget 3,000 tokens; ~200-token / <=20-triple graph reserve
  (triples are filled from the reserve, documents from the rest);
- dedup by chunk_id, then near-dup by normalised text hash;
- at most 2 chunks per document_id;
- truncate the lowest-ranked candidates first; never truncate the
  top-1 chunk mid-sentence — chunks are dropped whole, and the top-1
  chunk is always included whole even if it alone exceeds the doc
  budget (it never does on this corpus: ~60 tokens each).
"""

from __future__ import annotations

import hashlib
import re

TOTAL_BUDGET_TOKENS = 3000
GRAPH_RESERVE_TOKENS = 200
MAX_TRIPLES = 20
MAX_CHUNKS_PER_DOC = 2

# One short sentence per fact, template per relation type.
TRIPLE_TEMPLATES = {
    "HAS_COMPONENT": "{s} has component {o}.",
    "PART_OF": "{s} is part of {o}.",
    "LOCATED_IN": "{s} is located in {o}.",
    "DOCUMENTED_BY": "{s} is documented by {o}.",
    "RELATED_TO": "{s} is related to {o}.",
}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: chars / 4. Approximation, not a tokenizer."""
    return max(1, len(text) // 4)


def verbalise_triple(subject: str, relation: str, obj: str, source_doc_id: str) -> str:
    """One short sentence per fact, with the source doc id in brackets."""
    template = TRIPLE_TEMPLATES.get(relation, "{s} {r} {o}.")
    sentence = template.format(s=subject, o=obj, r=relation)
    return f"{sentence} [{source_doc_id}]"


def assemble_context(
    ranked_candidates: list[dict],
    triples: list[dict] | None = None,
    *,
    total_budget_tokens: int = TOTAL_BUDGET_TOKENS,
    graph_reserve_tokens: int = GRAPH_RESERVE_TOKENS,
    max_triples: int = MAX_TRIPLES,
) -> dict:
    """Assemble the LLM context for ranked candidates + graph triples.

    Candidates need ``doc_id`` (== chunk_id on this corpus),
    ``title`` and ``content``. Triples need ``subject`` / ``relation``
    / ``object`` / ``source_doc_id``. Returns ``{"text", "tokens",
    "n_documents", "n_triples", "truncated", "doc_ids"}``.
    """
    doc_budget = total_budget_tokens - graph_reserve_tokens
    seen_chunks: set[str] = set()
    seen_texts: set[str] = set()
    per_doc: dict[str, int] = {}
    chosen: list[dict] = []
    used_tokens = 0
    truncated = False

    for i, cand in enumerate(ranked_candidates):
        chunk_id = cand.get("chunk_id", cand.get("doc_id"))
        if chunk_id in seen_chunks:
            continue
        text_key = hashlib.sha256(
            re.sub(r"\s+", " ", (cand.get("content", "") or "").strip().lower()).encode()
        ).hexdigest()
        if text_key in seen_texts:
            continue
        doc_id = cand.get("doc_id", chunk_id)
        if per_doc.get(doc_id, 0) >= MAX_CHUNKS_PER_DOC:
            continue
        chunk_text = (cand.get("content", "") or "").strip()
        chunk_tokens = estimate_tokens(chunk_text) + estimate_tokens(
            cand.get("title", "") or ""
        ) + 8  # heading + citation overhead
        if i == 0:
            # Top-1 chunk: always included whole, never mid-sentence.
            pass
        elif used_tokens + chunk_tokens > doc_budget:
            truncated = True
            continue
        seen_chunks.add(chunk_id)
        seen_texts.add(text_key)
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        chosen.append(cand)
        used_tokens += chunk_tokens

    doc_lines = ["## Relevant documents"]
    for i, cand in enumerate(chosen, start=1):
        doc_lines.append(
            f"[D{i}] {cand.get('title', '')} — {cand.get('doc_id', '')}\n"
            f"{(cand.get('content', '') or '').strip()}"
        )
    doc_block = "\n\n".join(doc_lines)

    triple_lines = ["## Related facts from the asset graph"]
    n_triples = 0
    triple_tokens = 0
    for triple in triples or []:
        if n_triples >= max_triples:
            break
        sentence = verbalise_triple(
            triple["subject"], triple["relation"], triple["object"],
            triple.get("source_doc_id", triple["subject"]),
        )
        cost = estimate_tokens(sentence)
        if triple_tokens + cost > graph_reserve_tokens:
            break
        triple_lines.append(sentence)
        triple_tokens += cost
        n_triples += 1
    if n_triples == 0:
        triple_lines.append("(No graph facts linked to the question.)")
    triple_block = "\n".join(triple_lines)

    text = f"{doc_block}\n\n{triple_block}"
    return {
        "text": text,
        "tokens": estimate_tokens(text),
        "n_documents": len(chosen),
        "n_triples": n_triples,
        "truncated": truncated,
        "doc_ids": [c.get("doc_id") for c in chosen],
    }
