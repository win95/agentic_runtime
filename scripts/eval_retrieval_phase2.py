"""Phase 2 hybrid retrieval eval — PPR + coverage + hub, re-tune, ablation.

Fake-only: no LiteLLM proxy, no Qdrant/Neo4j server. Two setups:

(a) Phase 0 BoW fake — expected to saturate; reported for
    completeness, NOT the exit criterion.
(b) Code-blurred fake — vectors that ignore/confuse asset codes
    (``runtime/retrieval_blurred.py``); the exit criterion is judged
    here.

Runs, per setup: plain-vector baseline, Phase 1 hybrid
(``phase2=False``) and Phase 2 hybrid (default). Also grid-searches
w_graph in {0.30, 0.40, 0.50} (renormalising vec:exact 5:1) on the
blurred setup, and runs the PPR ablation — the hybrid retriever
with the PPR sub-weight set to 0 (``w_ppr=0``, a real code path)
vs the full Phase 2 retriever — reporting the delta on nDCG@6
overall and on the multi-hop subset.

Exit criterion (code-blurred setup):
  1. multi-hop nDCG@6 improves over Phase 1's hybrid number
     (Phase 1 blurred multi-hop nDCG@6 = 0.954), and
  2. the PPR ablation shows a measurable delta (overall nDCG@6
     differs AND at least one case's ranking actually changes —
     the flag must be a live code path, not a dead parameter), and
  3. no regression vs Phase 1 overall, and no regression vs the
     baseline on the BoW setup.
Exit code 0 = exit criterion passed.

Usage:
    python -m scripts.eval_retrieval_phase2 [--tracking-uri URI] [--output FILE]
    make eval-retrieval-phase2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.retrieval_fusion import renormalise_weights  # noqa: E402
from runtime.retrieval_harness import (  # noqa: E402
    format_results_table,
    load_golden_set,
    run_retrieval_eval,
)
from runtime.retrieval_hybrid import (  # noqa: E402
    build_blurred_baseline_retriever,
    build_hybrid_retriever,
)


def _grid_search(cases, golden_version: str) -> list[dict]:
    """Grid-search w_graph on the blurred setup; returns rows per weight."""
    rows = []
    for w_graph in (0.30, 0.40, 0.50):
        weights = renormalise_weights(w_graph)
        retriever = build_hybrid_retriever(
            blurred=True,
            w_vec=weights["w_vec"], w_graph=weights["w_graph"], w_exact=weights["w_exact"],
        )
        result = run_retrieval_eval(
            retriever, cases, retriever_name=f"hybrid2_blurred_w{w_graph:.2f}",
            golden_set_version=golden_version,
        )
        agg, sib = result["aggregate"], result["sibling_confusion"]
        multi = result["by_query_type"]["multi-hop"]
        rows.append({
            "w_graph": w_graph, "w_vec": weights["w_vec"], "w_exact": weights["w_exact"],
            "recall@6": round(agg["recall@6"], 4),
            "ndcg@6": round(agg["ndcg@6"], 4),
            "mrr": round(agg["mrr"], 4),
            "multi_hop_ndcg@6": round(multi["ndcg@6"], 4),
            "sibling_ndcg@6": round(sib["ndcg@6"], 4),
            "sibling_recall@6": round(sib["recall@6"], 4),
        })
    return rows


def _ablation(cases, golden_version: str) -> dict:
    """PPR ablation on the blurred setup: w_ppr=0 vs full (0.45)."""
    full = run_retrieval_eval(
        build_hybrid_retriever(blurred=True), cases,
        retriever_name="hybrid2_blurred_full", golden_set_version=golden_version,
    )
    ablated = run_retrieval_eval(
        build_hybrid_retriever(blurred=True, w_ppr=0.0), cases,
        retriever_name="hybrid2_blurred_no_ppr", golden_set_version=golden_version,
    )
    changed = [
        row["id"]
        for a, row in zip(full["per_query"], ablated["per_query"])
        if a["ranked_ids"] != row["ranked_ids"]
    ]
    per_case_delta = {
        row["id"]: round((row["ndcg@6"] or 0) - (other["ndcg@6"] or 0), 4)
        for row, other in zip(full["per_query"], ablated["per_query"])
        if abs((row["ndcg@6"] or 0) - (other["ndcg@6"] or 0)) > 1e-9
    }
    return {
        "full": full,
        "ablated": ablated,
        "overall_ndcg_delta": round(
            full["aggregate"]["ndcg@6"] - ablated["aggregate"]["ndcg@6"], 4),
        "multi_hop_ndcg_delta": round(
            full["by_query_type"]["multi-hop"]["ndcg@6"]
            - ablated["by_query_type"]["multi-hop"]["ndcg@6"], 4),
        "changed_case_ids": changed,
        "per_case_ndcg_delta": per_case_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 hybrid retrieval eval.")
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking URI (default: skip MLflow).")
    parser.add_argument("--output", default=None, help="Write full results JSON.")
    parser.add_argument("--golden-set", default=None,
                        help="Path to the golden set JSON.")
    args = parser.parse_args()

    golden = load_golden_set(args.golden_set)
    cases = golden["cases"]
    version = golden.get("version", "unknown")

    from runtime.retrieval_baseline import build_baseline_retriever  # noqa: E402

    setups = {
        "bow (Phase 0 fake)": (
            build_baseline_retriever(),
            build_hybrid_retriever(blurred=False, phase2=False),
            build_hybrid_retriever(blurred=False)),
        "code-blurred": (
            build_blurred_baseline_retriever(),
            build_hybrid_retriever(blurred=True, phase2=False),
            build_hybrid_retriever(blurred=True)),
    }
    results: dict = {}
    for setup_name, (baseline, hybrid1, hybrid2) in setups.items():
        for label, retriever in (
            ("baseline", baseline), ("hybrid-phase1", hybrid1), ("hybrid-phase2", hybrid2),
        ):
            key = f"{setup_name} / {label}"
            results[key] = run_retrieval_eval(
                retriever, cases,
                retriever_name=f"{label}_{setup_name}",
                golden_set_version=version,
                tracking_uri=args.tracking_uri,
                params={"embedding": "deterministic-bow" if "bow" in setup_name
                        else "code-blurred-bow (digit tokens dropped)"},
            )
            print(f"--- {key} ---")
            print(format_results_table(results[key]))
            print()

    grid = _grid_search(cases, version)
    print("Grid search (code-blurred setup, Phase 2, w_graph varied, vec:exact 5:1):")
    for row in grid:
        print(
            f"  w_graph={row['w_graph']:.2f} w_vec={row['w_vec']:.3f} "
            f"w_exact={row['w_exact']:.3f}  Recall@6={row['recall@6']:.3f}  "
            f"nDCG@6={row['ndcg@6']:.3f}  multi-hop nDCG@6={row['multi_hop_ndcg@6']:.3f}  "
            f"sibling nDCG@6={row['sibling_ndcg@6']:.3f} "
            f"sibling Recall@6={row['sibling_recall@6']:.3f}"
        )
    best = max(grid, key=lambda r: (r["ndcg@6"], r["recall@6"]))
    print(f"Best grid row: w_graph={best['w_graph']:.2f} "
          f"(nDCG@6 {best['ndcg@6']:.3f}); priors are w_graph=0.40.")

    ablation = _ablation(cases, version)
    full, ablated = ablation["full"], ablation["ablated"]
    print("\nPPR ablation (code-blurred setup, w_ppr=0 vs full 0.45):")
    print(f"  overall nDCG@6: {ablated['aggregate']['ndcg@6']:.3f} (w_ppr=0) -> "
          f"{full['aggregate']['ndcg@6']:.3f} (full), "
          f"delta {ablation['overall_ndcg_delta']:+.3f}")
    print(f"  multi-hop nDCG@6: "
          f"{ablated['by_query_type']['multi-hop']['ndcg@6']:.3f} (w_ppr=0) -> "
          f"{full['by_query_type']['multi-hop']['ndcg@6']:.3f} (full), "
          f"delta {ablation['multi_hop_ndcg_delta']:+.3f}")
    print(f"  cases whose ranking changes: {ablation['changed_case_ids'] or 'none'}")
    print(f"  per-case nDCG@6 deltas (full - ablated): "
          f"{ablation['per_case_ndcg_delta'] or 'none'}")

    blurred_p1 = results["code-blurred / hybrid-phase1"]
    blurred_p2 = results["code-blurred / hybrid-phase2"]
    bow_base = results["bow (Phase 0 fake) / baseline"]["aggregate"]
    bow_p2 = results["bow (Phase 0 fake) / hybrid-phase2"]["aggregate"]
    mh_p1 = blurred_p1["by_query_type"]["multi-hop"]["ndcg@6"]
    mh_p2 = blurred_p2["by_query_type"]["multi-hop"]["ndcg@6"]
    multi_hop_improves = mh_p2 > mh_p1
    ablation_measurable = (
        ablation["overall_ndcg_delta"] != 0 and bool(ablation["changed_case_ids"])
    )
    no_regression = (
        blurred_p2["aggregate"]["ndcg@6"] >= blurred_p1["aggregate"]["ndcg@6"]
        and bow_p2["ndcg@6"] >= bow_base["ndcg@6"]
    )
    exit_ok = multi_hop_improves and ablation_measurable and no_regression
    print("\nEXIT CRITERION (code-blurred setup): "
          f"multi-hop nDCG@6 {mh_p1:.3f} (Phase 1) -> {mh_p2:.3f} (Phase 2) "
          f"[{'improves' if multi_hop_improves else 'NO IMPROVEMENT'}]; "
          f"PPR ablation overall delta {ablation['overall_ndcg_delta']:+.3f} "
          f"over {len(ablation['changed_case_ids'])} changed case(s) "
          f"[{'measurable' if ablation_measurable else 'NOT MEASURABLE'}]; "
          f"overall {blurred_p1['aggregate']['ndcg@6']:.3f} -> "
          f"{blurred_p2['aggregate']['ndcg@6']:.3f}, BoW "
          f"{bow_base['ndcg@6']:.3f} -> {bow_p2['ndcg@6']:.3f} "
          f"[{'no regression' if no_regression else 'REGRESSION'}] — "
          + ("PASS" if exit_ok else "FAIL"))
    print("NOTE: fake-setup numbers only. Live numbers must be re-measured "
          "on the user's machine with the real embedding model + seeded Qdrant.")

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"results": results, "grid_search": grid, "ablation": ablation,
             "exit_criterion_pass": exit_ok},
            indent=2, ensure_ascii=False, default=str,
        ))
        print(f"Full results written to {args.output}")
    return 0 if exit_ok else 1


if __name__ == "__main__":
    sys.exit(main())
