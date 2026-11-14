"""Retrieval metrics — pure functions, no infra (Phase 0, report §6).

Binary relevance: a ranked id is relevant iff it is in the gold set.
All metrics return floats in [0, 1]. Edge cases:

- empty ranking -> 0.0
- empty gold set -> 0.0 (callers exclude unanswerable cases from means
  instead; see runtime/retrieval_harness.py)
- k > len(ranked) -> the whole ranking is used
- duplicate ids in the ranking count once, at first occurrence
"""

from __future__ import annotations

import math

DEFAULT_K_CONTEXT = 6
DEFAULT_K_POOL = 20


def _dedup(ranked_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in ranked_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def recall_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Fraction of gold ids present in the top-k of the ranking."""
    if not gold_ids or k <= 0:
        return 0.0
    top_k = set(_dedup(ranked_ids)[:k])
    return len(top_k & set(gold_ids)) / len(set(gold_ids))


def precision_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Precision@k: fraction of the top-k ranking that is gold (binary).

    Denominator is ``k`` (the context size), not the number of gold
    ids — so on this corpus a single-gold case scores at most 1/6.
    That makes Precision@6 a *set* metric here (did the top-6 keep
    the gold docs and exclude distractors), complementary to
    Recall@6 (same numerator, denominator = |gold|). Edge cases
    mirror the other metrics: empty ranking / empty gold / k <= 0
    -> 0.0; duplicates count once, at first occurrence.
    """
    if not gold_ids or k <= 0:
        return 0.0
    top_k = _dedup(ranked_ids)[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & set(gold_ids)) / k


def ndcg_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Binary-relevance nDCG@k: DCG of the top-k over the ideal DCG."""
    if not gold_ids or k <= 0:
        return 0.0
    gold = set(gold_ids)
    ranked = _dedup(ranked_ids)[:k]
    relevances = [1 if doc_id in gold else 0 for doc_id in ranked]
    ideal = [1] * min(len(gold), k) + [0] * max(0, k - len(gold))
    denom = _dcg(ideal)
    return _dcg(relevances) / denom if denom else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_ids: list[str]) -> float:
    """1 / rank of the first gold id, or 0.0 when no gold id is ranked."""
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)
    for i, doc_id in enumerate(_dedup(ranked_ids)):
        if doc_id in gold:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ranking(
    ranked_ids: list[str],
    gold_ids: list[str],
    k_context: int = DEFAULT_K_CONTEXT,
    k_pool: int = DEFAULT_K_POOL,
) -> dict[str, float]:
    """All Phase 0 metrics for one query's ranking."""
    return {
        f"recall@{k_context}": recall_at_k(ranked_ids, gold_ids, k_context),
        f"ndcg@{k_context}": ndcg_at_k(ranked_ids, gold_ids, k_context),
        "mrr": reciprocal_rank(ranked_ids, gold_ids),
        f"recall@{k_pool}": recall_at_k(ranked_ids, gold_ids, k_pool),
    }
