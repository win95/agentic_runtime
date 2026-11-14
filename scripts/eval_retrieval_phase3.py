"""Phase 3 hybrid retrieval eval — gated expansion A/B + LTR logging.

Fake-only: no LiteLLM proxy, no Qdrant/Neo4j server. The expansion
generators are the DETERMINISTIC STUBS
(``runtime/retrieval_expansion.stub_paraphrase_variants`` /
``stub_decompose_question``) — no live LLM exists in this VM, so the
A/B measures the merge/rerank plumbing and the gating, not the
quality of model-written paraphrases. Two setups:

(a) Phase 0 BoW fake — expected to saturate; reported for
    completeness, NOT the exit criterion.
(b) Code-blurred fake — the exit-criterion setup.

Runs, per setup: hybrid Phase 2 with expansion OFF (the base
retriever) vs ON (``ExpandedRetriever``). Both use the same fusion,
so any difference is the expansion alone.

Exit criterion (code-blurred setup AND BoW setup — the criterion
is "no loss anywhere", judged on the blurred setup first):
  1. Precision@6 (on) >= Precision@6 (off),
  2. Recall@6 (on) >= Recall@6 (off),
  3. nDCG@6 (on) >= nDCG@6 (off), and
  4. on the GATED subset (cases where expansion triggered),
     nDCG@6 (on) >= nDCG@6 (off) — reported separately, same ids.
Recall@20 is SATURATED at 1.000 on this 7-doc corpus by
construction (the pool holds all 7 docs), so no Recall@20 gain is
claimed or expected; the exit is judged on the @6 metrics above.

The script also reports the LTR labelled-pair count the golden set
yields (cases x candidates) against the ~3,000-pair LambdaMART
threshold, and states plainly that LTR training is PENDING DATA.

Exit code 0 = exit criterion passed.

Usage:
    python -m scripts.eval_retrieval_phase3 [--tracking-uri URI] [--output FILE]
    make eval-retrieval-phase3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.retrieval_expansion import build_expanded_retriever  # noqa: E402
from runtime.retrieval_harness import (  # noqa: E402
    evaluate_retriever,
    format_results_table,
    load_golden_set,
    log_to_mlflow,
    ltr_feature_rows,
    ltr_pair_count,
    run_retrieval_eval,
)
from runtime.retrieval_hybrid import build_hybrid_retriever  # noqa: E402


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _gated_subset(result: dict) -> dict:
    """Metrics over the gated subset: answerable cases where the
    retriever's trace says expansion triggered (kind != 'none').

    Returns the case ids plus Precision@6 / Recall@6 / nDCG@6 means
    over those cases. The OFF run has no expansion key in its trace,
    so its gated subset is computed on the SAME case ids as the ON
    run (passed in by the caller via ``case_ids``) — see
    ``_gated_subset_for_ids``.
    """
    rows = [
        r for r in result["per_query"]
        if r["answerable"] and (r.get("trace") or {}).get("expansion", {}).get("triggered")
    ]
    return _summarise(rows)


def _gated_subset_for_ids(result: dict, case_ids: list[str]) -> dict:
    rows = [r for r in result["per_query"] if r["id"] in set(case_ids)]
    return _summarise(rows)


def _summarise(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "case_ids": [r["id"] for r in answerable],
        "precision@6": round(_mean(
            [r["precision@6"] for r in answerable]), 4),
        "recall@6": round(_mean([r["recall@6"] for r in answerable]), 4),
        "ndcg@6": round(_mean([r["ndcg@6"] or 0 for r in answerable]), 4),
    }


def _overlap_stats(result: dict) -> dict:
    """Per-variant overlap stats from the expanded run's traces."""
    triggered = [
        r for r in result["per_query"]
        if (r.get("trace") or {}).get("expansion", {}).get("triggered")
    ]
    kinds: dict[str, int] = {}
    jaccards: list[float] = []
    cand_jaccards: list[float] = []
    n_variants: list[int] = []
    for row in triggered:
        exp = row["trace"]["expansion"]
        kinds[exp.get("kind", "?")] = kinds.get(exp.get("kind", "?"), 0) + 1
        overlap = exp.get("variant_overlap") or {}
        if "mean_top6_jaccard" in overlap:
            jaccards.append(overlap["mean_top6_jaccard"])
        if "mean_candidate_jaccard" in overlap:
            cand_jaccards.append(overlap["mean_candidate_jaccard"])
        n_variants.append(len(exp.get("variants") or []))
    return {
        "n_triggered": len(triggered),
        "n_total": len(result["per_query"]),
        "triggered_by_kind": kinds,
        "mean_variants_per_triggered": round(_mean(n_variants), 2),
        "mean_top6_jaccard_vs_base": round(_mean(jaccards), 4),
        "mean_candidate_jaccard_vs_base": round(_mean(cand_jaccards), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 hybrid retrieval eval.")
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
        "bow (Phase 0 fake)": (False, "deterministic-bow"),
        "code-blurred": (True, "code-blurred-bow (digit tokens dropped)"),
    }
    results: dict = {}
    gated: dict = {}
    overlaps: dict = {}
    for setup_name, (blurred, embedding) in setups.items():
        off = run_retrieval_eval(
            build_hybrid_retriever(blurred=blurred), cases,
            retriever_name=f"hybrid2_off_{setup_name}",
            golden_set_version=version,
            tracking_uri=args.tracking_uri,
            params={"embedding": embedding, "expansion": "off"},
        )
        on_retriever = build_expanded_retriever(blurred=blurred)
        # The blurred ON run also carries the LTR feature rows to
        # MLflow (log_to_mlflow's ltr_rows artifact); the other runs
        # go through run_retrieval_eval as usual.
        if blurred and args.tracking_uri:
            on = evaluate_retriever(
                on_retriever, cases, retriever_name=f"hybrid2_on_{setup_name}",
                golden_set_version=version,
            )
            on["mlflow_run_id"] = log_to_mlflow(
                on, tracking_uri=args.tracking_uri,
                run_name=f"hybrid2_on_{setup_name}",
                params={"embedding": embedding, "expansion": "on"},
                ltr_rows=ltr_feature_rows(on),
            )
        else:
            on = run_retrieval_eval(
                on_retriever, cases,
                retriever_name=f"hybrid2_on_{setup_name}",
                golden_set_version=version,
                tracking_uri=args.tracking_uri,
                params={"embedding": embedding, "expansion": "on"},
            )
        results[f"{setup_name} / off"] = off
        results[f"{setup_name} / on"] = on
        for label, result in (("off", off), ("on", on)):
            print(f"--- {setup_name} / expansion {label} ---")
            print(format_results_table(result))
            print()
        gated_on = _gated_subset(on)
        gated_off = _gated_subset_for_ids(off, gated_on["case_ids"])
        gated[setup_name] = {"on": gated_on, "off": gated_off}
        overlaps[setup_name] = _overlap_stats(on)
        print(f"Gated subset ({setup_name}, n={gated_on['n_answerable']} answerable, "
              f"cases {gated_on['case_ids']}):")
        print(f"  Precision@6: {gated_off['precision@6']:.3f} (off) -> "
              f"{gated_on['precision@6']:.3f} (on)")
        print(f"  Recall@6:    {gated_off['recall@6']:.3f} (off) -> "
              f"{gated_on['recall@6']:.3f} (on)")
        print(f"  nDCG@6:      {gated_off['ndcg@6']:.3f} (off) -> "
              f"{gated_on['ndcg@6']:.3f} (on)")
        ov = overlaps[setup_name]
        print(f"Expansion overlap ({setup_name}): triggered "
              f"{ov['n_triggered']}/{ov['n_total']} "
              f"{ov['triggered_by_kind']}, mean variants "
              f"{ov['mean_variants_per_triggered']}, mean top-6 Jaccard "
              f"vs base {ov['mean_top6_jaccard_vs_base']:.3f}, mean "
              f"candidate Jaccard vs base "
              f"{ov['mean_candidate_jaccard_vs_base']:.3f}")
        print()

    # LTR labelled-pair count (blurred ON run — the exit-criterion run).
    ltr_rows = ltr_feature_rows(results["code-blurred / on"])
    ltr = ltr_pair_count(results["code-blurred / on"])
    print("LTR feature logging (no training — see docs/hybrid-retrieval-phase3.md §4):")
    print(f"  labelled pairs: {ltr['pairs']} total "
          f"({ltr['answerable_pairs']} from answerable cases) vs "
          f"~{ltr['threshold']} needed for LambdaMART — "
          f"{'MEETS threshold' if ltr['meets_threshold'] else 'PENDING DATA'}")
    print("  rows logged per candidate: G1–G9, vec, exact, ranks "
          "before/after, final score, provenance, expansion metadata")
    print("  NOTE: Recall@20 is 1.000 by construction on this 7-doc "
          "corpus (the pool holds all 7 docs); no Recall@20 gain is "
          "claimed.")

    def _agg(result):
        return result["aggregate"]

    exit_ok = True
    verdict_lines = []
    for setup_name in setups:
        off, on = _agg(results[f"{setup_name} / off"]), _agg(results[f"{setup_name} / on"])
        g_off, g_on = gated[setup_name]["off"], gated[setup_name]["on"]
        checks = {
            "Precision@6 no loss": on["precision@6"] >= off["precision@6"],
            "Recall@6 no regression": on["recall@6"] >= off["recall@6"],
            "nDCG@6 no regression": on["ndcg@6"] >= off["ndcg@6"],
            "gated nDCG@6 no regression": g_on["ndcg@6"] >= g_off["ndcg@6"],
        }
        for name, ok in checks.items():
            exit_ok = exit_ok and ok
            verdict_lines.append(
                f"  [{setup_name}] {name}: {'PASS' if ok else 'FAIL'}")
        print(f"EXIT CHECKS ({setup_name}): "
              f"Precision@6 {off['precision@6']:.3f} -> {on['precision@6']:.3f}; "
              f"Recall@6 {off['recall@6']:.3f} -> {on['recall@6']:.3f}; "
              f"nDCG@6 {off['ndcg@6']:.3f} -> {on['ndcg@6']:.3f}; "
              f"gated nDCG@6 {g_off['ndcg@6']:.3f} -> {g_on['ndcg@6']:.3f}")
    print("\nEXIT CRITERION (expansion off vs on, both fake setups):")
    print("\n".join(verdict_lines))
    print("— " + ("PASS" if exit_ok else "FAIL"))
    print("NOTE: fake-setup numbers only, stub generators only. Live "
          "numbers (real paraphrases from the 3B model, real embeddings) "
          "must be re-measured on the user's machine.")

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"results": results, "gated_subset": gated,
             "overlap_stats": overlaps, "ltr": ltr,
             "ltr_feature_rows": ltr_rows,
             "exit_criterion_pass": exit_ok},
            indent=2, ensure_ascii=False, default=str,
        ))
        print(f"Full results written to {args.output}")
    return 0 if exit_ok else 1


if __name__ == "__main__":
    sys.exit(main())
