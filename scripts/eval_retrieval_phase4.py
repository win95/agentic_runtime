"""Phase 4 hybrid retrieval eval — QueryPlan: link + intent + gated expansion.

Fake-only: no LiteLLM proxy, no Qdrant/Neo4j server. The generators
are the DETERMINISTIC STUBS (``runtime/retrieval_expansion.
stub_paraphrase_variants`` / ``stub_decompose_question``) — no live
LLM exists in this VM, so the A/B measures the plan plumbing, the
gating and the guardrails, not the quality of model-written
paraphrases. Two setups:

(a) Phase 0 BoW fake — reported for completeness, NOT the exit
    criterion.
(b) Code-blurred fake — the exit-criterion setup.

Runs, per setup: hybrid Phase 2 with NO plan (the base retriever)
vs WITH plan (``HybridRetriever.retrieve(question, query_plan)``
where the plan is built per question by
``runtime/retrieval_query_plan.build_query_plan`` with the stub
generators). Both use the same fusion, so any difference is the
plan alone.

What is measured (see ``docs/hybrid-retrieval-phase4.md`` for the
definitions):

* entity-link accuracy — per case, the fraction of the gold
  ENTITY nodes (``gold_node_ids`` minus ``doc-*`` entries) that
  the plan's linked nodes cover; mean over cases, plus the
  exact-set match rate. Link labels are entity-only because the
  gold ``doc-*`` entries are documents, not linked entities.
* intent accuracy — the plan's scored intent vs the golden
  ``intent`` label (a deliberate per-case judgement of what the
  question ASKS, NOT a copy of ``query_type``; unanswerable
  cases are labelled by their ask, never ``unanswerable`` —
  answerability is a separate dimension). Overall, per
  ``query_type``, and on the gated subset.
* rewrite faithfulness — over the paraphrase-generated cases
  (gate open, no decomposition, seeds present): the preservation
  rate of the stub variants, AND a re-check of every KEPT
  variant (re-running the guardrails on the plan's kept list —
  any violation is a FAIL, not a warning).
* downstream nDCG@6 delta — with-plan vs without, overall, per
  query type, and on the gated subset (cases where the plan's
  gate opened).

Exit criterion (code-blurred setup AND BoW setup — "no loss
anywhere", judged on the blurred setup first):
  1. link accuracy + intent accuracy measured and reported per
     query type (no threshold — the numbers ARE the result);
  2. rewrite faithfulness 100% on the preservation checks
     (every kept variant passes the re-check);
  3. nDCG@6 (on) >= nDCG@6 (off), overall AND on the gated
     subset.
Exit code 0 = exit criterion passed. A FAIL stops here — do not
force a pass.

Usage:
    python -m scripts.eval_retrieval_phase4 [--tracking-uri URI] [--output FILE]
    make eval-retrieval-phase4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.retrieval_expansion import (  # noqa: E402
    MAX_VARIANTS,
    stub_decompose_question,
    stub_paraphrase_variants,
)
from runtime.retrieval_graph import build_demo_adjacency  # noqa: E402
from runtime.retrieval_harness import (  # noqa: E402
    format_results_table,
    load_golden_set,
    run_retrieval_eval,
)
from runtime.retrieval_hybrid import build_hybrid_retriever  # noqa: E402
from runtime.retrieval_query_plan import (  # noqa: E402
    build_query_plan,
    check_rewrite_variants,
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _entity_nodes(case: dict) -> list[str]:
    """Gold ENTITY nodes only: gold_node_ids minus doc-* entries."""
    return [n for n in (case.get("gold_node_ids") or []) if not n.startswith("doc-")]


def _summarise_ndcg(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "case_ids": [r["id"] for r in answerable],
        "ndcg@6": round(_mean([r["ndcg@6"] or 0 for r in answerable]), 4),
        "recall@6": round(_mean([r["recall@6"] for r in answerable]), 4),
        "precision@6": round(_mean([r["precision@6"] for r in answerable]), 4),
    }


def _make_planned_retriever(base, plans: dict):
    """A retriever callable that builds a QueryPlan per question.

    ``plans`` is filled in as a side effect (question -> plan) so
    the caller can measure link / intent / faithfulness from the
    exact plans the retrieval used.
    """

    def retrieve(question: str):
        plan = build_query_plan(
            question,
            retriever=base,
            embeddings=base.embeddings,
            adjacency=base.adjacency,
            paraphrase_generator=stub_paraphrase_variants,
            decomposition_generator=stub_decompose_question,
        )
        plans[question] = plan
        return base.retrieve(question, query_plan=plan)

    retrieve.__name__ = getattr(base, "__name__", "hybrid") + "_with_plan"
    return retrieve


def _link_accuracy(cases: list[dict], plans: dict) -> dict:
    """Entity-link accuracy: per-case gold-entity recall + exact rate."""
    per_case: dict[str, dict] = {}
    recalls: list[float] = []
    exact = 0
    for case in cases:
        gold_ent = set(_entity_nodes(case))
        plan = plans[case["question"]]
        linked = {s["key"] for s in plan.linked_nodes}
        recall = len(linked & gold_ent) / len(gold_ent) if gold_ent else 1.0
        recalls.append(recall)
        is_exact = linked == gold_ent
        exact += int(is_exact)
        per_case[case["id"]] = {
            "gold_entity_nodes": sorted(gold_ent),
            "linked_nodes": sorted(linked),
            "recall": round(recall, 4),
            "exact_set_match": is_exact,
        }
    by_type: dict[str, dict] = {}
    for query_type in sorted({c["query_type"] for c in cases}):
        rows = [per_case[c["id"]] for c in cases if c["query_type"] == query_type]
        by_type[query_type] = {
            "n": len(rows),
            "mean_recall": round(_mean([r["recall"] for r in rows]), 4),
            "exact_set_match_rate": round(
                _mean([1.0 if r["exact_set_match"] else 0.0 for r in rows]), 4),
        }
    return {
        "mean_recall": round(_mean(recalls), 4),
        "exact_set_match_rate": round(exact / len(cases), 4) if cases else 0.0,
        "by_query_type": by_type,
        "per_case": per_case,
    }


def _intent_accuracy(cases: list[dict], plans: dict) -> dict:
    """Intent accuracy: scored intent vs the golden intent label."""
    per_case: dict[str, dict] = {}
    correct = 0
    for case in cases:
        plan = plans[case["question"]]
        hit = plan.intent == case.get("intent")
        correct += int(hit)
        per_case[case["id"]] = {
            "query_type": case["query_type"],
            "gold_intent": case.get("intent"),
            "scored_intent": plan.intent,
            "margin": plan.confidence,
            "correct": hit,
        }
    by_type: dict[str, dict] = {}
    for query_type in sorted({c["query_type"] for c in cases}):
        rows = [per_case[c["id"]] for c in cases if c["query_type"] == query_type]
        by_type[query_type] = {
            "n": len(rows),
            "accuracy": round(_mean([1.0 if r["correct"] else 0.0 for r in rows]), 4),
            "misses": [c_id for c_id, r in
                       ((c["id"], per_case[c["id"]]) for c in cases
                        if c["query_type"] == query_type)
                       if not r["correct"]],
        }
    gated_ids = [c["id"] for c in cases if not plans[c["question"]].gated_off]
    gated_rows = [per_case[c_id] for c_id in gated_ids]
    return {
        "accuracy": round(correct / len(cases), 4) if cases else 0.0,
        "n_correct": correct,
        "n": len(cases),
        "by_query_type": by_type,
        "gated_subset": {
            "n": len(gated_rows),
            "case_ids": gated_ids,
            "accuracy": round(
                _mean([1.0 if r["correct"] else 0.0 for r in gated_rows]), 4
            ) if gated_rows else 0.0,
        },
        "per_case": per_case,
    }


def _rewrite_faithfulness(cases: list[dict], plans: dict) -> dict:
    """Faithfulness of the paraphrase path.

    Only cases where the plan actually generated paraphrases
    count: gate open, no decomposition sub-questions, seeds
    present. For each, the stub generator is re-run and the
    guardrails re-applied (the plan's kept list must equal the
    recomputed kept list), and every KEPT variant is re-checked
    on its own — any violation there is a FAIL. The generated
    pass rate is reported as measured; the exit check is the
    kept-variant re-check (100%).
    """
    generated = 0
    kept_total = 0
    recompute_mismatch: list[str] = []
    recheck_violations: list[dict] = []
    per_case: dict[str, dict] = {}
    for case in cases:
        plan = plans[case["question"]]
        if plan.gated_off or plan.sub_questions or not plan.linked_nodes:
            continue
        raw = list(stub_paraphrase_variants(case["question"]) or [])
        raw = [v.strip() for v in raw
               if v and v.strip()
               and v.strip().lower() != case["question"].strip().lower()]
        raw = list(dict.fromkeys(raw))[:MAX_VARIANTS]
        if not raw:
            continue
        kept, dropped, _reasons = check_rewrite_variants(
            case["question"], raw, linked_nodes=plan.linked_nodes,
            adjacency=build_demo_adjacency(),
        )
        generated += len(raw)
        kept_total += len(kept)
        if kept != list(plan.rewrite_variants):
            recompute_mismatch.append(case["id"])
        # Re-check every kept variant on its own: it must pass.
        _re_kept, re_dropped, re_reasons = check_rewrite_variants(
            case["question"], list(plan.rewrite_variants),
            linked_nodes=plan.linked_nodes,
            adjacency=build_demo_adjacency(),
        )
        if re_dropped:
            recheck_violations.append({
                "case_id": case["id"],
                "dropped": re_dropped,
                "reasons": re_reasons,
            })
        per_case[case["id"]] = {
            "generated": len(raw),
            "kept": len(kept),
            "dropped": dropped,
        }
    return {
        "cases_with_paraphrases": len(per_case),
        "generated_variants": generated,
        "kept_variants": kept_total,
        "preservation_rate": round(kept_total / generated, 4) if generated else 1.0,
        "kept_recheck_violations": recheck_violations,
        "kept_recheck_pass": not recheck_violations and not recompute_mismatch,
        "recompute_mismatch_case_ids": recompute_mismatch,
        "per_case": per_case,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 hybrid retrieval eval.")
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
    measurements: dict = {}
    for setup_name, (blurred, embedding) in setups.items():
        base = build_hybrid_retriever(blurred=blurred)
        off = run_retrieval_eval(
            base, cases,
            retriever_name=f"hybrid2_noplan_{setup_name}",
            golden_set_version=version,
            tracking_uri=args.tracking_uri,
            params={"embedding": embedding, "query_plan": "off"},
        )
        planned_base = build_hybrid_retriever(blurred=blurred)
        plans: dict = {}
        on = run_retrieval_eval(
            _make_planned_retriever(planned_base, plans), cases,
            retriever_name=f"hybrid2_plan_{setup_name}",
            golden_set_version=version,
            tracking_uri=args.tracking_uri,
            params={"embedding": embedding, "query_plan": "on"},
        )
        results[f"{setup_name} / off"] = off
        results[f"{setup_name} / on"] = on
        for label, result in (("off", off), ("on", on)):
            print(f"--- {setup_name} / plan {label} ---")
            print(format_results_table(result))
            print()

        link = _link_accuracy(cases, plans)
        intent = _intent_accuracy(cases, plans)
        faith = _rewrite_faithfulness(cases, plans)
        gated_ids = intent["gated_subset"]["case_ids"]
        gated_on = _summarise_ndcg(
            [r for r in on["per_query"] if r["id"] in set(gated_ids)])
        gated_off = _summarise_ndcg(
            [r for r in off["per_query"] if r["id"] in set(gated_ids)])
        measurements[setup_name] = {
            "link_accuracy": link,
            "intent_accuracy": intent,
            "rewrite_faithfulness": faith,
            "gated_subset_ndcg": {"on": gated_on, "off": gated_off},
            "n_gated_open": len(gated_ids),
            "gated_open_case_ids": gated_ids,
        }
        print(f"Entity-link accuracy ({setup_name}): mean recall "
              f"{link['mean_recall']:.3f}, exact-set match rate "
              f"{link['exact_set_match_rate']:.3f}")
        for query_type, row in link["by_query_type"].items():
            print(f"  link {query_type:<12} n={row['n']:<3} recall "
                  f"{row['mean_recall']:.3f} exact {row['exact_set_match_rate']:.3f}")
        print(f"Intent accuracy ({setup_name}): {intent['n_correct']}/"
              f"{intent['n']} = {intent['accuracy']:.3f} "
              f"(gated subset {intent['gated_subset']['accuracy']:.3f} "
              f"on n={intent['gated_subset']['n']})")
        for query_type, row in intent["by_query_type"].items():
            print(f"  intent {query_type:<12} n={row['n']:<3} accuracy "
                  f"{row['accuracy']:.3f} misses {row['misses']}")
        print(f"Rewrite faithfulness ({setup_name}): "
              f"{faith['kept_variants']}/{faith['generated_variants']} variants kept "
              f"(preservation rate {faith['preservation_rate']:.3f} over "
              f"{faith['cases_with_paraphrases']} paraphrase cases); kept-variant "
              f"re-check: {'PASS' if faith['kept_recheck_pass'] else 'FAIL'}")
        print(f"Gated subset ({setup_name}, n={gated_on['n_answerable']} answerable, "
              f"cases {gated_on['case_ids']}):")
        print(f"  nDCG@6: {gated_off['ndcg@6']:.3f} (off) -> "
              f"{gated_on['ndcg@6']:.3f} (on)")
        print()

    exit_ok = True
    verdict_lines = []
    for setup_name in setups:
        off = results[f"{setup_name} / off"]["aggregate"]
        on = results[f"{setup_name} / on"]["aggregate"]
        m = measurements[setup_name]
        g_off = m["gated_subset_ndcg"]["off"]
        g_on = m["gated_subset_ndcg"]["on"]
        checks = {
            "rewrite faithfulness 100% (kept re-check)":
                m["rewrite_faithfulness"]["kept_recheck_pass"],
            "nDCG@6 no regression": on["ndcg@6"] >= off["ndcg@6"],
            "gated nDCG@6 no regression": g_on["ndcg@6"] >= g_off["ndcg@6"],
        }
        for name, ok in checks.items():
            exit_ok = exit_ok and ok
            verdict_lines.append(
                f"  [{setup_name}] {name}: {'PASS' if ok else 'FAIL'}")
        print(f"EXIT CHECKS ({setup_name}): nDCG@6 {off['ndcg@6']:.3f} -> "
              f"{on['ndcg@6']:.3f}; gated nDCG@6 {g_off['ndcg@6']:.3f} -> "
              f"{g_on['ndcg@6']:.3f}; faithfulness kept re-check "
              f"{'PASS' if checks['rewrite faithfulness 100% (kept re-check)'] else 'FAIL'}; "
              f"link recall {m['link_accuracy']['mean_recall']:.3f}, "
              f"intent accuracy {m['intent_accuracy']['accuracy']:.3f}")
    print("\nEXIT CRITERION (plan off vs on, both fake setups):")
    print("\n".join(verdict_lines))
    print("— " + ("PASS" if exit_ok else "FAIL"))
    print("NOTE: fake-setup numbers only, stub generators only. Live "
          "numbers (real paraphrases from the 3B model, real embeddings) "
          "must be re-measured on the user's machine.")

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"results": results, "measurements": measurements,
             "exit_criterion_pass": exit_ok},
            indent=2, ensure_ascii=False, default=str,
        ))
        print(f"Full results written to {args.output}")
    return 0 if exit_ok else 1


if __name__ == "__main__":
    sys.exit(main())
