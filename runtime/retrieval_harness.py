"""MLflow retrieval harness — Phase 0 (report §6, §8).

Runs a retriever callable over the golden set and aggregates retrieval
metrics. A retriever is ``question -> ranked chunk/doc ids``; it may
also return ``(ranked_ids, trace_dict)`` where the trace carries extra
per-query info (candidate scores, latency, seed ids). The harness
measures wall-clock latency itself when the retriever does not report
one.

Unanswerable cases (empty ``gold_chunk_ids``) are EXCLUDED from the
recall / nDCG / MRR means — there is nothing to recall — and reported
separately (count + mean candidates returned). That keeps the headline
numbers comparable across phases while still showing whether a
retriever floods unanswerable questions with confident-looking docs.

MLflow: ``log_to_mlflow`` follows the scripts/eval_mission.py pattern
(set_tracking_uri / set_experiment / start_run / log_params /
log_metrics / log artifact) but NEVER raises — an unreachable server
or a disabled MLflow only warns and the eval result is still
returned. Pass ``tracking_uri=None`` to skip MLflow entirely (local
run, results still printed / returned).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from runtime.retrieval_metrics import (
    DEFAULT_K_CONTEXT,
    DEFAULT_K_POOL,
    evaluate_ranking,
    precision_at_k,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_SET_PATH = ROOT / "scripts" / "golden_set_fault_analysis.json"
EXPERIMENT = "agentic-runtime-retrieval"
QUERY_TYPES = ("lookup", "multi-hop", "aggregate", "comparison", "unanswerable")

# Learning-to-rank (Phase 3, report §4.4d): a labelled pair is one
# (query, candidate) row with a binary relevance label. LambdaMART /
# LightGBM listwise rankers need on the order of ~3,000 such pairs
# before training is worth attempting (report §4.4d: ~100 cases x
# ~30 candidates). The golden set yields far fewer — see
# ``ltr_pair_count`` — so training is PENDING DATA, not built.
LTR_PAIR_THRESHOLD = 3000


def load_golden_set(path: str | Path | None = None) -> dict:
    """Load the golden set file. Returns {"version", "mission", "cases"}."""
    golden_path = Path(path) if path is not None else DEFAULT_GOLDEN_SET_PATH
    with golden_path.open() as fh:
        data = json.load(fh)
    problems = validate_golden_set(data)
    if problems:
        raise ValueError(f"golden set {golden_path} invalid: " + "; ".join(problems))
    return data


def validate_golden_set(data: dict) -> list[str]:
    """Schema check for a loaded golden set. Returns a list of problems."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["golden set root must be a JSON object"]
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        return ["golden set has no 'cases' list"]
    ids: set[str] = set()
    for i, case in enumerate(cases):
        where = f"case {i} ({case.get('id', '?')})"
        if not isinstance(case, dict):
            problems.append(f"{where}: not an object")
            continue
        case_id = case.get("id")
        if not case_id or not isinstance(case_id, str):
            problems.append(f"{where}: missing string 'id'")
        elif case_id in ids:
            problems.append(f"{where}: duplicate id {case_id!r}")
        else:
            ids.add(case_id)
        for field in ("question", "expected_answer"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                problems.append(f"{where}: missing/empty {field!r}")
        for field in ("must_include_facts", "gold_chunk_ids", "gold_node_ids"):
            if not isinstance(case.get(field), list) or not all(
                isinstance(v, str) for v in case[field]
            ):
                problems.append(f"{where}: {field!r} must be a list of strings")
        if case.get("query_type") not in QUERY_TYPES:
            problems.append(f"{where}: unknown query_type {case.get('query_type')!r}")
        if "sibling_confusion" in case and not isinstance(case["sibling_confusion"], bool):
            problems.append(f"{where}: 'sibling_confusion' must be a bool")
        if case.get("query_type") == "unanswerable" and case.get("gold_chunk_ids"):
            problems.append(f"{where}: unanswerable case must have empty gold_chunk_ids")
        if case.get("query_type") != "unanswerable" and not case.get("gold_chunk_ids"):
            problems.append(f"{where}: answerable case needs non-empty gold_chunk_ids")
    return problems


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _breakdown(per_query: list[dict]) -> dict[str, dict]:
    """Per-query-type (or subset) means over answerable cases only."""
    out: dict[str, dict] = {}
    types = sorted({row["query_type"] for row in per_query})
    recall_key = f"recall@{DEFAULT_K_CONTEXT}"
    ndcg_key = f"ndcg@{DEFAULT_K_CONTEXT}"
    precision_key = f"precision@{DEFAULT_K_CONTEXT}"
    pool_key = f"recall@{DEFAULT_K_POOL}"
    for query_type in types:
        rows = [r for r in per_query if r["query_type"] == query_type]
        answerable = [r for r in rows if r["answerable"]]
        out[query_type] = {
            "n": len(rows),
            "n_answerable": len(answerable),
            recall_key: _mean([r[recall_key] for r in answerable]),
            ndcg_key: _mean([r[ndcg_key] for r in answerable]),
            precision_key: _mean([r[precision_key] for r in answerable]),
            "mrr": _mean([r["mrr"] for r in answerable]),
            pool_key: _mean([r[pool_key] for r in answerable]),
            "mean_latency_seconds": _mean([r["latency_seconds"] for r in rows]),
        }
    return out


def evaluate_retriever(
    retriever,
    cases: list[dict],
    *,
    k_context: int = DEFAULT_K_CONTEXT,
    k_pool: int = DEFAULT_K_POOL,
    retriever_name: str = "retriever",
    golden_set_version: str = "unknown",
) -> dict:
    """Evaluate one retriever over golden-set cases. Returns full results."""
    per_query: list[dict] = []
    for case in cases:
        start = time.monotonic()
        outcome = retriever(case["question"])
        if isinstance(outcome, tuple):
            ranked_ids, trace = outcome
        else:
            ranked_ids, trace = outcome, {}
        latency = trace.get("latency_seconds")
        if latency is None:
            latency = time.monotonic() - start
        ranked_ids = [str(doc_id) for doc_id in (ranked_ids or [])]
        gold_ids = list(case.get("gold_chunk_ids") or [])
        answerable = bool(gold_ids)
        metrics = evaluate_ranking(ranked_ids, gold_ids, k_context, k_pool) \
            if answerable else {
                f"recall@{k_context}": None,
                f"ndcg@{k_context}": None,
                f"precision@{k_context}": None,
                "mrr": None,
                f"recall@{k_pool}": None,
            }
        if answerable:
            # Precision@6 (Phase 3) is computed here, not inside
            # evaluate_ranking: a Phase 0 test pins that function's
            # exact key set, so the metric stays additive.
            metrics[f"precision@{k_context}"] = precision_at_k(
                ranked_ids, gold_ids, k_context)
        per_query.append({
            "id": case["id"],
            "query_type": case["query_type"],
            "sibling_confusion": bool(case.get("sibling_confusion", False)),
            "answerable": answerable,
            "question": case["question"],
            "gold_chunk_ids": gold_ids,
            "gold_node_ids": list(case.get("gold_node_ids") or []),
            "ranked_ids": ranked_ids[:k_pool],
            "latency_seconds": round(float(latency), 4),
            "trace": trace,
            **metrics,
            "n_candidates_returned": len(ranked_ids),
        })

    answerable_rows = [r for r in per_query if r["answerable"]]
    unanswerable_rows = [r for r in per_query if not r["answerable"]]
    sibling_rows = [r for r in answerable_rows if r["sibling_confusion"]]
    aggregate = {
        "n_cases": len(per_query),
        "n_answerable": len(answerable_rows),
        "n_unanswerable": len(unanswerable_rows),
        f"recall@{k_context}": _mean([r[f"recall@{k_context}"] for r in answerable_rows]),
        f"ndcg@{k_context}": _mean([r[f"ndcg@{k_context}"] for r in answerable_rows]),
        f"precision@{k_context}": _mean(
            [r[f"precision@{k_context}"] for r in answerable_rows]),
        "mrr": _mean([r["mrr"] for r in answerable_rows]),
        f"recall@{k_pool}": _mean([r[f"recall@{k_pool}"] for r in answerable_rows]),
        "mean_latency_seconds": _mean([r["latency_seconds"] for r in per_query]),
        "unanswerable_mean_candidates": _mean(
            [r["n_candidates_returned"] for r in unanswerable_rows]
        ),
    }
    by_type = _breakdown(per_query)
    sibling = {
        "n": len(sibling_rows),
        f"recall@{k_context}": _mean([r[f"recall@{k_context}"] for r in sibling_rows]),
        f"ndcg@{k_context}": _mean([r[f"ndcg@{k_context}"] for r in sibling_rows]),
        f"precision@{k_context}": _mean(
            [r[f"precision@{k_context}"] for r in sibling_rows]),
        "mrr": _mean([r["mrr"] for r in sibling_rows]),
        f"recall@{k_pool}": _mean([r[f"recall@{k_pool}"] for r in sibling_rows]),
    }
    return {
        "retriever": retriever_name,
        "golden_set_version": golden_set_version,
        "k_context": k_context,
        "k_pool": k_pool,
        "aggregate": aggregate,
        "by_query_type": by_type,
        "sibling_confusion": sibling,
        "per_query": per_query,
    }


def ltr_feature_rows(result: dict) -> list[dict]:
    """Full §4.1 feature row per (query, candidate) — LTR LOGGING ONLY.

    One row per candidate the retriever considered, built from the
    trace's ``feature_rows`` (every fused candidate — see
    ``runtime/retrieval_hybrid.trace_feature_rows``). Rows carry
    G1–G9 (G3 = the best-path relation list, G9 = the exact-code
    feature, BM25 absent — no BM25 in Phase 1, so ``bm25_rank`` is
    ``None`` rather than invented), vec / graph / final scores,
    ranks before (RRF) and after (fusion) reranking, provenance
    (arms + variants), the expansion metadata for the query, and
    the binary relevance label (1 iff the doc is gold).

    Fallback: a trace without ``feature_rows`` (pre-Phase-3
    retrievers) contributes its top-6 ``candidates`` dict instead,
    with ``rrf_rank`` / ``rank_before`` unknown (``None``) — the
    row is still logged, and the gap is visible in the fields.

    NO training happens here: there is no model, no trainer, no
    learned weights anywhere in this repo. These rows are the
    training data a future LTR job would consume once the pair
    count reaches ``LTR_PAIR_THRESHOLD`` (see ``ltr_pair_count``).
    """
    rows: list[dict] = []
    for query in result.get("per_query", []):
        trace = query.get("trace") or {}
        expansion = trace.get("expansion") or {}
        gold = set(query.get("gold_chunk_ids") or [])
        feature_rows = trace.get("feature_rows")
        if feature_rows:
            source = [
                {
                    "doc_id": r.get("doc_id"),
                    "vec_score": r.get("vec_score"),
                    "graph_score": r.get("graph_score"),
                    "exact": r.get("exact"),
                    "g1_min_hops": r.get("g1_min_hops"),
                    "g2_path_score": r.get("g2_path_score"),
                    "g3_best_path_rels": r.get("g3_best_path_rels"),
                    "g4_type_match": r.get("g4_type_match"),
                    "g5_direct": r.get("g5_direct"),
                    "g5_n_seeds_linked": r.get("g5_n_seeds_linked"),
                    "g6_coverage": r.get("g6_coverage"),
                    "g7_ppr": r.get("g7_ppr"),
                    "g8_hub": r.get("g8_hub"),
                    "g8_transit_key": r.get("g8_transit_key"),
                    "g9_exact": r.get("g9_exact"),
                    "bm25_rank": None,
                    "arms": r.get("arms"),
                    "provenance_variants": r.get("provenance_variants"),
                    "rrf_rank": r.get("rrf_rank"),
                    "rank_before": r.get("rank_before", r.get("rrf_rank")),
                    "rank_after": r.get("rank"),
                    "final_score": r.get("final_score"),
                    "z_vec": r.get("z_vec"),
                    "z_graph": r.get("z_graph"),
                }
                for r in feature_rows
            ]
        else:
            source = [
                {
                    "doc_id": doc_id,
                    "vec_score": c.get("vec"),
                    "graph_score": c.get("graph"),
                    "exact": c.get("exact"),
                    "g1_min_hops": c.get("g1"),
                    "g2_path_score": c.get("g2"),
                    "g3_best_path_rels": None,
                    "g4_type_match": c.get("g4"),
                    "g5_direct": c.get("g5"),
                    "g5_n_seeds_linked": None,
                    "g6_coverage": c.get("g6_coverage"),
                    "g7_ppr": c.get("g7_ppr"),
                    "g8_hub": c.get("g8_hub"),
                    "g8_transit_key": c.get("g8_transit"),
                    "g9_exact": c.get("exact"),
                    "bm25_rank": None,
                    "arms": None,
                    "provenance_variants": None,
                    "rrf_rank": None,
                    "rank_before": None,
                    "rank_after": None,
                    "final_score": c.get("final"),
                    "z_vec": None,
                    "z_graph": None,
                }
                for doc_id, c in (trace.get("candidates") or {}).items()
            ]
        for item in source:
            rows.append({
                "case_id": query.get("id"),
                "query_type": query.get("query_type"),
                "question": query.get("question"),
                "answerable": bool(query.get("answerable")),
                "gold_chunk_ids": sorted(gold),
                "label": 1 if item["doc_id"] in gold else 0,
                **item,
                "expansion_triggered": bool(expansion.get("triggered")),
                "expansion_kind": expansion.get("kind", "none"),
                "expansion_variants": list(expansion.get("variants") or []),
                "expansion_reasons": list(
                    (expansion.get("gating") or {}).get("reasons") or [])
                + list((expansion.get("multi_part") or {}).get("reasons") or []),
            })
    return rows


def ltr_pair_count(result: dict) -> dict:
    """Labelled-pair counts from an eval result vs the LTR threshold.

    ``pairs`` counts every (query, candidate) row; ``answerable_pairs``
    counts only rows from answerable cases (the only rows with a
    meaningful positive label — unanswerable cases label every
    candidate 0). Both are reported against ``LTR_PAIR_THRESHOLD``
    (~3,000, report §4.4d). Neither is close on this golden set,
    which is the point: LTR training is PENDING DATA.
    """
    rows = ltr_feature_rows(result)
    answerable = [r for r in rows if r["answerable"]]
    return {
        "pairs": len(rows),
        "answerable_pairs": len(answerable),
        "threshold": LTR_PAIR_THRESHOLD,
        "meets_threshold": len(answerable) >= LTR_PAIR_THRESHOLD,
    }


def _apply_fail_fast_http_defaults() -> None:
    """Mirror infra/mlflow_setup.py: a dead MLflow server must fail fast."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")


def log_to_mlflow(
    result: dict,
    *,
    tracking_uri: str | None = None,
    experiment: str = EXPERIMENT,
    run_name: str | None = None,
    params: dict | None = None,
    ltr_rows: list[dict] | None = None,
) -> str | None:
    """Log eval results to MLflow. Returns the run id, or None on any failure.

    Never raises: MLflow is observability, and a dead server must never
    fail an eval run. ``ltr_rows`` (Phase 3) logs the per-candidate
    feature rows as a ``ltr_feature_rows.json`` artifact alongside
    the per-query traces, following the same pattern.
    """
    if tracking_uri is None:
        logger.info("MLflow logging skipped (no tracking URI configured).")
        return None
    _apply_fail_fast_http_defaults()
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        aggregate = result["aggregate"]
        k_context = result["k_context"]
        k_pool = result["k_pool"]
        with mlflow.start_run(run_name=run_name or result["retriever"]) as run:
            mlflow.log_params({
                "retriever": result["retriever"],
                "golden_set_version": result["golden_set_version"],
                "n_cases": result["aggregate"]["n_cases"],
                "n_answerable": result["aggregate"]["n_answerable"],
                "n_unanswerable": result["aggregate"]["n_unanswerable"],
                "k_context": k_context,
                "k_pool": k_pool,
                **(params or {}),
            })
            # MLflow metric names may not contain '@'.
            metrics = {
                f"recall_at_{k_context}": aggregate[f"recall@{k_context}"],
                f"ndcg_at_{k_context}": aggregate[f"ndcg@{k_context}"],
                f"precision_at_{k_context}": aggregate[f"precision@{k_context}"],
                "mrr": aggregate["mrr"],
                f"recall_at_{k_pool}": aggregate[f"recall@{k_pool}"],
                "mean_latency_seconds": aggregate["mean_latency_seconds"],
                "unanswerable_mean_candidates": aggregate["unanswerable_mean_candidates"],
            }
            for query_type, row in result["by_query_type"].items():
                safe = query_type.replace("-", "_")
                metrics[f"recall_at_{k_context}_{safe}"] = row[f"recall@{k_context}"]
                metrics[f"ndcg_at_{k_context}_{safe}"] = row[f"ndcg@{k_context}"]
                metrics[f"precision_at_{k_context}_{safe}"] = row[f"precision@{k_context}"]
                metrics[f"mrr_{safe}"] = row["mrr"]
            sibling = result["sibling_confusion"]
            metrics[f"recall_at_{k_context}_sibling_confusion"] = sibling[f"recall@{k_context}"]
            metrics[f"ndcg_at_{k_context}_sibling_confusion"] = sibling[f"ndcg@{k_context}"]
            metrics[f"precision_at_{k_context}_sibling_confusion"] = sibling[
                f"precision@{k_context}"]
            metrics["mrr_sibling_confusion"] = sibling["mrr"]
            mlflow.log_metrics(metrics)
            # Full per-query trace (candidate ids, ranks, latency, gold
            # ids/nodes) + the golden set itself, so any later phase is a
            # like-for-like rerun of this one.
            mlflow.log_dict(
                {"per_query": result["per_query"]}, "per_query_traces.json"
            )
            if ltr_rows is not None:
                mlflow.log_dict(
                    {"rows": ltr_rows}, "ltr_feature_rows.json")
            golden_path = DEFAULT_GOLDEN_SET_PATH
            if golden_path.exists():
                mlflow.log_artifact(str(golden_path), "golden_set")
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001 — MLflow must never break eval
        logger.warning("MLflow logging failed (%s); eval results above unaffected.", exc)
        return None


def run_retrieval_eval(
    retriever,
    cases: list[dict] | None = None,
    *,
    retriever_name: str = "retriever",
    golden_set_version: str = "unknown",
    tracking_uri: str | None = None,
    run_name: str | None = None,
    params: dict | None = None,
) -> dict:
    """Evaluate + optionally log to MLflow. Always returns the results dict."""
    if cases is None:
        golden = load_golden_set()
        cases = golden["cases"]
        golden_set_version = golden.get("version", golden_set_version)
    result = evaluate_retriever(
        retriever, cases, retriever_name=retriever_name,
        golden_set_version=golden_set_version,
    )
    result["mlflow_run_id"] = log_to_mlflow(
        result, tracking_uri=tracking_uri, run_name=run_name, params=params
    )
    return result


def format_results_table(result: dict) -> str:
    """One-screen summary of the baseline numbers."""
    agg = result["aggregate"]
    k_context = result["k_context"]
    k_pool = result["k_pool"]
    lines = [
        f"Retriever: {result['retriever']}  "
        f"(golden set {result['golden_set_version']}, "
        f"n={agg['n_cases']}, answerable={agg['n_answerable']}, "
        f"unanswerable={agg['n_unanswerable']})",
        f"  Recall@{k_context}: {agg[f'recall@{k_context}']:.3f}   "
        f"nDCG@{k_context}: {agg[f'ndcg@{k_context}']:.3f}   "
        f"Precision@{k_context}: {agg[f'precision@{k_context}']:.3f}   "
        f"MRR: {agg['mrr']:.3f}   Recall@{k_pool}: {agg[f'recall@{k_pool}']:.3f}",
        f"  Mean latency: {agg['mean_latency_seconds']:.4f}s/query   "
        f"Unanswerable mean candidates returned: "
        f"{agg['unanswerable_mean_candidates']:.1f}",
        "  By query type:",
    ]
    for query_type, row in result["by_query_type"].items():
        if row["n_answerable"] == 0:
            lines.append(
                f"    {query_type:<12} n={row['n']:<3} "
                f"(no answerable cases — excluded from means)"
            )
            continue
        lines.append(
            f"    {query_type:<12} n={row['n']:<3} "
            f"Recall@{k_context}: {row[f'recall@{k_context}']:.3f}  "
            f"nDCG@{k_context}: {row[f'ndcg@{k_context}']:.3f}  "
            f"Precision@{k_context}: {row[f'precision@{k_context}']:.3f}  "
            f"MRR: {row['mrr']:.3f}  Recall@{k_pool}: {row[f'recall@{k_pool}']:.3f}"
        )
    sib = result["sibling_confusion"]
    lines.append(
        f"  Sibling-confusion subset (n={sib['n']}): "
        f"Recall@{k_context}: {sib[f'recall@{k_context}']:.3f}  "
        f"nDCG@{k_context}: {sib[f'ndcg@{k_context}']:.3f}  "
        f"Precision@{k_context}: {sib[f'precision@{k_context}']:.3f}  "
        f"MRR: {sib['mrr']:.3f}  Recall@{k_pool}: {sib[f'recall@{k_pool}']:.3f}"
    )
    return "\n".join(lines)
