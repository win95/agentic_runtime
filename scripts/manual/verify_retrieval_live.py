"""Manual verification: live retrieval + Question Understanding, combined.

Runs real questions through the COMBINED path — a QueryPlan first
(``runtime/retrieval_query_plan.build_query_plan``), then the hybrid
retriever with that plan (``HybridRetriever.retrieve(question,
query_plan=plan)``) — against the REAL stores (Qdrant + Neo4j +
LiteLLM query embeddings). Per question it prints what QU decided
(intent, linked nodes, gating, sub-questions / rewrites), what
retrieval returned (top-6 with vec / graph / exact / final scores),
the graph-arm state, and — when the question comes from the golden
set — whether the gold chunk is in the top-6. It then re-runs the
same question WITHOUT a plan so the QU effect is visible side by side.

What this verifies that ``make eval-retrieval-live`` does not: the
live eval runs the golden set through the retriever WITHOUT a plan.
This script is the only runner that exercises retrieval + QU together
on the live stores.

Checks (honest verdicts, not observations):

  FAIL
    - a question raised (plumbing bug, listed at the end);
    - an ANSWERABLE question (gold_chunk_ids non-empty) did not put
      any gold chunk in the top-6;
    - the plan turned an answerable gold HIT into a MISS that the
      no-plan path found (QU regression);
    - retrieval returned zero documents for a question;
    - ``trace["graph_missing"]`` disagrees with the seeds / expansion
      the retrieval actually used (trace/reporting bug);
    - a reachable Neo4j produced an empty gazetteer (linking is dead).

  WARN (signal, not a gate)
    - an UNANSWERABLE question (gold_chunk_ids empty) returned
      documents: the retriever has no abstention layer;
    - a seeded question expanded to 0 graph nodes (``no_expansion``);
    - questions that linked no gazetteer seed at all (``no_seed``) —
      vector-only by design, but a linking-coverage gap;
    - intent accuracy below ``--min-intent-accuracy`` is reported as
      WARN by default; pass the flag to turn it into a FAIL.

  SKIP
    - gold / intent checks for ``--question`` (no expectation);
    - the plan-vs-no-plan comparison under ``--no-compare``.

``graph_missing`` here is the RETRIEVAL flag — ``not seeds or not
expansion["nodes"]`` (``runtime/retrieval_hybrid.py``). It is NOT the
same as preflight's ``inventory["graph_missing"]``, which means Neo4j
is unreachable. The per-question ``graph:`` line separates the two:
``state=no_seed`` / ``state=no_expansion`` vs ``state=unreachable``.

What the new retrieval + QU touches (2026-11-14), so you know what
you are verifying:
  TOUCHED (library + its live seams):
    - ``runtime/retrieval_*.py``: the retriever itself (hybrid,
      fusion, PPR, expansion, query plan, intent, decompose,
      graph, linking, context, metrics, harness).
    - ``infra/qdrant_client.py``: the vector arm
      (``search_documents``) plus the live helpers
      (``count_documents_per_node``, ``get_document_links``,
      ``get_document_payloads``).
    - ``ontology/knowledge/graph_query.py``: the graph arm
      (``neighbors`` for live expansion / adjacency,
      ``list_entity_keys`` for the dynamic gazetteer).
    - ``scripts/eval_retrieval_live.py``: the golden-set eval
      against the real stores (retriever WITHOUT a plan).
    - ``scripts/golden_set_fault_analysis.json``: the 36-case
      golden set this script samples from.
  NOT TOUCHED (the running system does not call any of it):
    - No mission, agent, API route, worker task or session code
      calls the hybrid retriever or builds a QueryPlan. The
      fault_analysis capability and the ``retrieval_context``
      hook still use plain ``repositories.search_documents``
      (vector only). The opt-in wiring
      (``execution.retrieval.hybrid`` / ``query_plan`` on
      ``fault_analysis``) is NOT done — see
      ``docs/hybrid-retrieval.md`` section 12. Until that lands,
      this library changes nothing in production behaviour;
      it is only reachable through this script and the live eval.

Prerequisites (same stack as the other manual scripts):
  - Docker backing services up: ``docker compose up -d qdrant neo4j``
    (Postgres/Redis/worker are NOT needed — this script talks to
    the stores directly, not through the API).
  - ``EMBEDDING_MODEL`` in the repo-root ``.env`` set to the model
    name your LiteLLM proxy serves (``qwen3-embedding-0.6b`` on
    the connected proxy), and the proxy reachable.
  - Qdrant seeded (``make seed`` for the demo docs, or
    ``make ingest`` for real documents) and the Neo4j projection
    drained, so the stores hold real data.
  - Run from the repo root with the project venv.

Usage:
    .venv/bin/python scripts/manual/verify_retrieval_live.py
    .venv/bin/python scripts/manual/verify_retrieval_live.py --question "O21 quá nhiệt?"
    .venv/bin/python scripts/manual/verify_retrieval_live.py --golden-set
    .venv/bin/python scripts/manual/verify_retrieval_live.py --no-compare
    .venv/bin/python scripts/manual/verify_retrieval_live.py --min-intent-accuracy 0.5
    .venv/bin/python scripts/manual/verify_retrieval_live.py --output reports/retrieval.json

Exit codes: 0 = every check passed (WARNs allowed); 1 = at least
one FAIL (the failure labels are listed at the end); 2 = preflight
failed (missing embedding config or Qdrant, same as the live eval).

A gated-open question may call the LLM (live paraphrase /
decomposition generators), so ``--golden-set`` costs real proxy
calls; the 6-question default keeps that cheap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_config_check import Report  # noqa: E402

from runtime.retrieval_graph import (  # noqa: E402
    build_expansion_adjacency,
    build_live_adjacency,
    expand_via_graph_query,
    gazetteer_seeds,
    refresh_gazetteer_codes,
)
from runtime.retrieval_harness import load_golden_set  # noqa: E402
from runtime.retrieval_hybrid import build_hybrid_retriever  # noqa: E402
from runtime.retrieval_query_plan import build_query_plan  # noqa: E402
from scripts.eval_retrieval_live import (  # noqa: E402
    LivePreflightError,
    format_inventory,
    preflight_live_stores,
)

# Six sample questions from the golden set, one per shape. The gold
# intent / chunk ids are the expectation the checks compare against.
SAMPLE_IDS = ["L01", "L06", "M01", "C01", "A01", "U05"]

K = 6


def _print_top6(label: str, ranked: list[str], trace: dict) -> None:
    """Top-6 with the fused scores, from the retrieve trace."""
    print(f"  {label}:")
    candidates = trace.get("candidates", {})
    for rank, doc_id in enumerate(ranked[:6], 1):
        scores = candidates.get(doc_id, {})
        print(
            f"    {rank}. {doc_id} "
            f"vec={scores.get('vec')} graph={scores.get('graph')} "
            f"exact={scores.get('exact')} final={scores.get('final')}"
        )
    if not ranked:
        print("    (no documents returned)")


def _graph_state(
    seeds: list[dict],
    expansion: dict,
    *,
    graph_reachable: bool,
) -> str:
    """Which graph-arm state this question hit.

    ``unreachable`` is infra (preflight also warns); ``no_seed`` means
    linking found nothing; ``no_expansion`` means the seeds have no
    usable neighbours. Only ``ok`` means the graph arm ran.
    """
    if not graph_reachable:
        return "unreachable"
    if not seeds:
        return "no_seed"
    if not expansion.get("nodes"):
        return "no_expansion"
    return "ok"


def _adjacency_for(seeds: list[dict], expansion: dict) -> tuple[dict, str]:
    """Live adjacency, degrading to expansion-only then empty."""
    try:
        return build_live_adjacency(seeds, expansion), "live"
    except Exception as exc:  # noqa: BLE001 — degraded path, reported
        print(f"  [note] live adjacency failed ({exc}); "
              "falling back to expansion-only adjacency")
        try:
            return build_expansion_adjacency(seeds, expansion), "expansion-only"
        except Exception as exc2:  # noqa: BLE001 — last resort
            print(f"  [note] expansion adjacency failed ({exc2}); "
                  "using empty adjacency")
            return {}, "empty"


def run_one(
    retriever,
    question: str,
    *,
    gold: dict | None = None,
    compare: bool = True,
    graph_reachable: bool = True,
    report: Report | None = None,
) -> dict:
    """One question through QU + retrieval. Returns the verdict dict."""
    started = time.perf_counter()
    report = report or Report()
    case_id = gold.get("id") if gold else question[:32]
    verdict: dict = {"id": case_id, "question": question}

    gazetteer_links = gazetteer_seeds(question)
    gaz_expansion = expand_via_graph_query(gazetteer_links)
    adjacency, adjacency_source = _adjacency_for(gazetteer_links, gaz_expansion)

    plan = build_query_plan(
        question,
        retriever=retriever,
        embeddings=retriever.embeddings,
        adjacency=adjacency,
        doc_links=retriever._resolved_doc_links(),
    )
    ranked, trace = retriever.retrieve(question, query_plan=plan)

    # The plan's linked nodes are what retrieval actually seeded from
    # (``retrieve`` passes them to ``_candidate_rows``). Recompute the
    # expansion only when they differ from the gazetteer lookup.
    plan_seeds = list(plan.linked_nodes or [])
    gaz_keys = [s["key"] for s in gazetteer_links]
    plan_keys = [s["key"] for s in plan_seeds]
    if plan_keys == gaz_keys:
        seeds, expansion = gazetteer_links, gaz_expansion
    else:
        seeds = plan_seeds
        expansion = expand_via_graph_query(seeds) if seeds else {
            "nodes": {}, "truncated": False, "nodes_returned": 0,
        }

    state = _graph_state(seeds, expansion, graph_reachable=graph_reachable)
    trace_graph_missing = bool(trace.get("graph_missing"))
    expansion_nodes = expansion.get("nodes_returned", len(expansion.get("nodes", {})))

    print(f"Q: {question}")
    print(f"  seeds (gazetteer): {gaz_keys}")
    if plan_keys != gaz_keys:
        print(f"  seeds (plan):      {plan_keys}")
    print(f"  graph: state={state} expansion={expansion_nodes} "
          f"adjacency={adjacency_source}/{len(adjacency)} "
          f"graph_missing={trace_graph_missing}")
    print(f"  plan: intent={plan.intent} confidence={plan.confidence:.3f} "
          f"gated_off={plan.gated_off} "
          f"fallback={plan.fallback_reason}")
    print(f"  plan linked: {plan_keys}")
    if plan.sub_questions:
        print("  sub-questions:")
        for sub in plan.sub_questions:
            print(f"    - {sub['text']} "
                  f"(links: {sub.get('linked_nodes', [])})")
    if plan.rewrite_variants:
        print("  rewrite variants:")
        for variant in plan.rewrite_variants:
            print(f"    - {variant}")
    if plan.gating:
        print(f"  gating: {plan.gating}")
    _print_top6("top-6 (QU + hybrid)", ranked, trace)

    ranked_np: list[str] = []
    if compare:
        ranked_np, trace_np = retriever.retrieve(question)
        same = ranked_np[:K] == ranked[:K]
        print(f"  top-6 without plan: "
              f"{'SAME' if same else 'DIFFERS'}")
        if not same:
            _print_top6("top-6 (no plan)", ranked_np, trace_np)
        verdict["no_plan_same"] = same

    verdict.update({
        "ranked": ranked[:K],
        "ranked_without_plan": ranked_np[:K] if compare else None,
        "intent": plan.intent,
        "confidence": round(plan.confidence, 4),
        "gated_off": plan.gated_off,
        "fallback_reason": plan.fallback_reason,
        "seeds": gaz_keys,
        "plan_seeds": plan_keys,
        "expansion_nodes": expansion_nodes,
        "adjacency_source": adjacency_source,
        "graph_state": state,
        "graph_missing": trace_graph_missing,
    })

    print("  checks:")

    # 1. Retrieval returned something at all.
    report.check(
        bool(ranked),
        f"{case_id}: retrieval returned documents",
        f"{len(ranked)} doc(s)" if ranked else "top-6 is empty",
    )

    # 2. Trace/reporting consistency: graph_missing must describe the
    #    seeds + expansion retrieval actually used.
    expected_graph_missing = not seeds or not expansion.get("nodes")
    report.check(
        trace_graph_missing == expected_graph_missing,
        f"{case_id}: graph_missing matches seeds/expansion",
        f"trace={trace_graph_missing} expected={expected_graph_missing} "
        f"(seeds={len(seeds)}, expansion={expansion_nodes})",
    )

    # 3. Graph-arm state. no_expansion is the suspicious one (a seed
    #    with no edges); no_seed is aggregated below, not repeated.
    if state == "ok":
        report.check(
            True,
            f"{case_id}: graph arm live",
            f"{expansion_nodes} expansion nodes over {len(adjacency)} adjacency nodes",
        )
    elif state == "no_expansion":
        report.warn(
            f"{case_id}: seeds expanded to 0 graph nodes",
            f"seeds={plan_keys}",
        )

    # 4. Gold expectations.
    if gold is not None:
        gold_chunks = set(gold.get("gold_chunk_ids") or [])
        gold_intent = gold.get("intent")
        intent_match = plan.intent == gold_intent
        hit = bool(gold_chunks & set(ranked[:K]))
        print(f"  gold: intent={gold_intent} "
              f"{'MATCH' if intent_match else 'MISMATCH'}; "
              f"gold chunk in top-6: {'HIT' if hit else 'MISS'} "
              f"(gold: {sorted(gold_chunks)})")
        verdict.update({
            "gold_intent": gold_intent,
            "gold_chunks": sorted(gold_chunks),
            "intent_match": intent_match,
            "gold_hit": hit if gold_chunks else None,
            "answerable": bool(gold_chunks),
        })

        if gold_chunks:
            report.check(
                hit,
                f"{case_id}: answerable gold chunk in top-6",
                f"gold={sorted(gold_chunks)} top6={ranked[:K]}",
            )
        else:
            verdict["gold_hit"] = False
            if ranked:
                report.warn(
                    f"{case_id}: no abstention on unanswerable question",
                    f"gold=[] but retrieval returned {len(ranked)} doc(s)",
                )

        # 5. Did the plan help or hurt on an answerable question?
        if compare and gold_chunks:
            hit_np = bool(gold_chunks & set(ranked_np[:K]))
            if hit_np and not hit:
                report.check(
                    False,
                    f"{case_id}: plan lost a gold hit the no-plan path found",
                    f"plan={ranked[:K]} no_plan={ranked_np[:K]}",
                )
            elif hit and not hit_np:
                report.check(
                    True,
                    f"{case_id}: plan recovered a gold hit",
                    f"plan={ranked[:K]} no_plan={ranked_np[:K]}",
                )
    elif compare:
        verdict["gold_hit"] = None

    verdict["seconds"] = round(time.perf_counter() - started, 2)
    print(f"  ({verdict['seconds']}s)")
    print()
    return verdict


def _summarise(verdicts: list[dict]) -> dict:
    """Aggregate counts over the run (used by both print and JSON)."""
    with_gold = [v for v in verdicts if v.get("gold_intent") is not None]
    answerable = [v for v in with_gold if v.get("answerable")]
    unanswerable = [v for v in with_gold if not v.get("answerable")]
    states = {"ok": 0, "no_seed": 0, "no_expansion": 0, "unreachable": 0}
    for v in verdicts:
        states[v.get("graph_state", "unreachable")] += 1
    compared = [v for v in verdicts if v.get("no_plan_same") is not None]
    regressions = [
        v for v in answerable
        if v.get("gold_hit") and v.get("no_plan_same") is not None
        and not (set(v.get("gold_chunks") or []) & set(v.get("ranked_without_plan") or []))
    ]
    recoveries = [
        v for v in answerable
        if not v.get("gold_hit") and v.get("no_plan_same") is not None
        and (set(v.get("gold_chunks") or []) & set(v.get("ranked_without_plan") or []))
    ]
    return {
        "questions": len(verdicts),
        "gold_cases": len(with_gold),
        "answerable": len(answerable),
        "answerable_hit": sum(1 for v in answerable if v.get("gold_hit")),
        "unanswerable": len(unanswerable),
        "unanswerable_returned": sum(
            1 for v in unanswerable if v.get("ranked")
        ),
        "intent_match": sum(1 for v in with_gold if v.get("intent_match")),
        "intent_accuracy": (
            sum(1 for v in with_gold if v.get("intent_match")) / len(with_gold)
            if with_gold else None
        ),
        "graph_states": states,
        "compared": len(compared),
        "no_plan_same": sum(1 for v in compared if v.get("no_plan_same")),
        "plan_regressions": [v["id"] for v in regressions],
        "plan_recoveries": [v["id"] for v in recoveries],
        "errors": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manual verification: live retrieval + Question Understanding."
    )
    parser.add_argument("--question", default=None,
                        help="One ad-hoc question (no gold expectation).")
    parser.add_argument("--golden-set", action="store_true",
                        help="Run all 36 golden-set cases.")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip the without-plan comparison retrieval.")
    parser.add_argument("--min-intent-accuracy", type=float, default=None,
                        help="Gate intent accuracy at this ratio (default: WARN only).")
    parser.add_argument("--output", default=None,
                        help="Write full verdicts JSON.")
    args = parser.parse_args()

    report = Report()

    try:
        inventory = preflight_live_stores()
    except LivePreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}")
        return 2

    print(format_inventory(inventory))
    print()
    print("Preflight checks:")
    report.check(
        inventory["qdrant_points"] >= 1,
        "Qdrant collection readable and non-empty",
        f"{inventory['qdrant_collection']!r} dim={inventory['qdrant_dim']} "
        f"points={inventory['qdrant_points']}",
    )
    report.check(
        bool(inventory["embedding_model_query"]),
        "embedding query model resolved",
        f"index={inventory['embedding_model_index']!r} "
        f"query={inventory['embedding_model_query']!r}",
    )
    graph_reachable = not inventory["graph_missing"]
    if graph_reachable:
        report.check(
            True,
            "Neo4j reachable",
            f"{inventory['graph_node_count']} nodes",
        )
    else:
        report.warn(
            "Neo4j unreachable — graph arm degrades to vector-only",
            "start Neo4j for the full hybrid numbers",
        )

    retriever = build_hybrid_retriever()
    gazetteer = refresh_gazetteer_codes()
    print(f"Gazetteer refresh: {gazetteer}")
    key_count = sum(
        int(gazetteer.get(k, 0))
        for k in ("assets", "components", "buildings", "documents")
    )
    if key_count:
        report.check(True, "gazetteer loaded", f"{key_count} keys")
    elif graph_reachable:
        report.check(
            False,
            "gazetteer is empty on a reachable graph",
            "linking would never seed — check the ontology projection",
        )
    else:
        report.warn("gazetteer empty (Neo4j unreachable)")

    cases: list[dict] = []
    if args.question:
        cases = [{"question": args.question}]
    elif args.golden_set:
        cases = load_golden_set()["cases"]
    else:
        golden = {c["id"]: c for c in load_golden_set()["cases"]}
        cases = [golden[cid] for cid in SAMPLE_IDS]

    compare = not args.no_compare
    if not args.question:
        print()
        print("Question checks:")
    verdicts: list[dict] = []
    errors: list[str] = []
    for case in cases:
        try:
            verdicts.append(run_one(
                retriever, case["question"],
                gold=case if "id" in case else None,
                compare=compare,
                graph_reachable=graph_reachable,
                report=report,
            ))
        except Exception as exc:  # noqa: BLE001 — one bad case must not kill the run
            case_id = case.get("id", case["question"][:32])
            print(f"  [ERROR] {case_id}: {exc!r}")
            print()
            report.check(False, f"{case_id}: question raised", repr(exc))
            errors.append(f"{case_id}: {exc!r}")

    summary = _summarise(verdicts)
    summary["errors"] = errors

    print("Summary:")
    if summary["gold_cases"]:
        print(f"  answerable gold: {summary['answerable_hit']}/"
              f"{summary['answerable']} hit in top-6")
        print(f"  unanswerable: {summary['unanswerable']} case(s), "
              f"{summary['unanswerable_returned']} returned documents "
              "(no abstention layer)")
        print(f"  intent: {summary['intent_match']}/{summary['gold_cases']} "
              f"= {summary['intent_accuracy']:.3f}")
    else:
        report.skip("gold / intent checks", "no gold expectation for --question")
    states = summary["graph_states"]
    print(f"  graph states: ok={states['ok']} no_seed={states['no_seed']} "
          f"no_expansion={states['no_expansion']} "
          f"unreachable={states['unreachable']}")
    if summary["compared"]:
        print(f"  plan vs no-plan: {summary['no_plan_same']}/"
              f"{summary['compared']} SAME; "
              f"recoveries={summary['plan_recoveries']} "
              f"regressions={summary['plan_regressions']}")
    else:
        report.skip("plan vs no-plan comparison", "--no-compare")

    # Aggregate WARNs: linking coverage and the intent observation.
    if states["no_seed"]:
        report.warn(
            "linking coverage",
            f"{states['no_seed']}/{summary['questions']} question(s) linked "
            "no gazetteer seed (vector-only)",
        )
    if summary["intent_accuracy"] is not None:
        intent_detail = (
            f"{summary['intent_match']}/{summary['gold_cases']} = "
            f"{summary['intent_accuracy']:.3f}"
        )
        if args.min_intent_accuracy is not None:
            report.check(
                summary["intent_accuracy"] >= args.min_intent_accuracy,
                "intent accuracy >= threshold",
                f"{intent_detail} (threshold {args.min_intent_accuracy:.3f})",
            )
        else:
            report.warn("intent accuracy (observation only)", intent_detail)

    total = len(report.failures)
    print(
        f"\nchecks: {'PASS' if not total else 'FAIL'} — "
        f"{report.passed} passed, {report.skipped} skipped, "
        f"{len(report.warnings)} warning(s), {total} failure(s)"
    )
    if report.warnings:
        print("WARN:")
        for label in report.warnings:
            print(f"  - {label}")
    if report.failures:
        print("FAIL:")
        for label in report.failures:
            print(f"  - {label}")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "inventory": inventory,
            "gazetteer": gazetteer,
            "summary": summary,
            "checks": {
                "passed": report.passed,
                "skipped": report.skipped,
                "warnings": report.warnings,
                "failures": report.failures,
            },
            "verdicts": verdicts,
        }, indent=2, ensure_ascii=False, default=str))
        print(f"Full verdicts written to {args.output}")

    print("PASS: live retrieval + QU" if not total
          else "FAIL: live retrieval + QU")
    return 0 if not total else 1


if __name__ == "__main__":
    sys.exit(main())
