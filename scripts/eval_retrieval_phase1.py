"""Phase 1 hybrid retrieval eval — baseline vs hybrid, both fake setups.

Fake-only: no LiteLLM proxy, no Qdrant/Neo4j server. Two setups:

(a) Phase 0 BoW fake — expected to saturate (Recall@6 1.000, sibling
    nDCG@6 1.000); reported for completeness, NOT the exit criterion.
(b) Code-blurred fake — vectors that ignore/confuse asset codes
    (``runtime/retrieval_blurred.py``); the exit criterion is judged
    here: hybrid must beat the plain-vector baseline on nDCG@6 and
    Recall@6 overall AND on the sibling-confusion subset.

Also grid-searches w_graph in {0.30, 0.40, 0.50} (renormalising
vec:exact 5:1) on the blurred setup and reports the runner-up.

Phase 2 exists now (``scripts/eval_retrieval_phase2.py``) and is
the default in ``runtime/retrieval_hybrid.py``; this script pins
``phase2=False`` everywhere so it keeps measuring Phase 1 exactly
as documented in ``docs/hybrid-retrieval-phase1.md``.

Usage:
    python -m scripts.eval_retrieval_phase1 [--tracking-uri URI] [--output FILE]
    make eval-retrieval-phase1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.retrieval_baseline import build_baseline_retriever  # noqa: E402
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
            blurred=True, phase2=False,
            w_vec=weights["w_vec"], w_graph=weights["w_graph"], w_exact=weights["w_exact"],
        )
        result = run_retrieval_eval(
            retriever, cases, retriever_name=f"hybrid_blurred_w{w_graph:.2f}",
            golden_set_version=golden_version,
        )
        agg, sib = result["aggregate"], result["sibling_confusion"]
        rows.append({
            "w_graph": w_graph, "w_vec": weights["w_vec"], "w_exact": weights["w_exact"],
            "recall@6": round(agg["recall@6"], 4),
            "ndcg@6": round(agg["ndcg@6"], 4),
            "mrr": round(agg["mrr"], 4),
            "sibling_ndcg@6": round(sib["ndcg@6"], 4),
            "sibling_recall@6": round(sib["recall@6"], 4),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 hybrid retrieval eval.")
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking URI (default: skip MLflow).")
    parser.add_argument("--output", default=None, help="Write full results JSON.")
    parser.add_argument("--golden-set", default=None,
                        help="Path to the golden set JSON.")
    args = parser.parse_args()

    golden = load_golden_set(args.golden_set)
    cases = golden["cases"]
    version = golden.get("version", "unknown")

    setups = {
        "bow (Phase 0 fake)": (
            build_baseline_retriever(), build_hybrid_retriever(blurred=False, phase2=False)),
        "code-blurred": (
            build_blurred_baseline_retriever(), build_hybrid_retriever(blurred=True, phase2=False)),
    }
    results: dict = {}
    for setup_name, (baseline, hybrid) in setups.items():
        for label, retriever in (("baseline", baseline), ("hybrid", hybrid)):
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
    print("Grid search (code-blurred setup, w_graph varied, vec:exact 5:1):")
    for row in grid:
        print(
            f"  w_graph={row['w_graph']:.2f} w_vec={row['w_vec']:.3f} "
            f"w_exact={row['w_exact']:.3f}  Recall@6={row['recall@6']:.3f}  "
            f"nDCG@6={row['ndcg@6']:.3f}  sibling nDCG@6={row['sibling_ndcg@6']:.3f} "
            f"sibling Recall@6={row['sibling_recall@6']:.3f}"
        )
    best = max(grid, key=lambda r: (r["ndcg@6"], r["recall@6"]))
    print(f"Best grid row: w_graph={best['w_graph']:.2f} "
          f"(nDCG@6 {best['ndcg@6']:.3f}); priors are w_graph=0.40.")

    blurred_base = results["code-blurred / baseline"]["aggregate"]
    blurred_hyb = results["code-blurred / hybrid"]["aggregate"]
    sib_base = results["code-blurred / baseline"]["sibling_confusion"]
    sib_hyb = results["code-blurred / hybrid"]["sibling_confusion"]
    exit_ok = (
        blurred_hyb["ndcg@6"] > blurred_base["ndcg@6"]
        and blurred_hyb["recall@6"] >= blurred_base["recall@6"]
        and sib_hyb["ndcg@6"] > sib_base["ndcg@6"]
    )
    print("\nEXIT CRITERION (code-blurred setup): "
          f"nDCG@6 {blurred_base['ndcg@6']:.3f} -> {blurred_hyb['ndcg@6']:.3f}, "
          f"Recall@6 {blurred_base['recall@6']:.3f} -> {blurred_hyb['recall@6']:.3f}, "
          f"sibling nDCG@6 {sib_base['ndcg@6']:.3f} -> {sib_hyb['ndcg@6']:.3f} — "
          + ("PASS" if exit_ok else "FAIL"))
    print("NOTE: fake-setup numbers only. Live numbers must be re-measured "
          "on the user's machine with the real embedding model + seeded Qdrant.")

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"results": results, "grid_search": grid, "exit_criterion_pass": exit_ok},
            indent=2, ensure_ascii=False, default=str,
        ))
        print(f"Full results written to {args.output}")
    return 0 if exit_ok else 1


if __name__ == "__main__":
    sys.exit(main())
