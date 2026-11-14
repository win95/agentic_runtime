"""Manual verification: live retrieval + Question Understanding, combined.

Runs real questions through the COMBINED path — a QueryPlan first
(``runtime/retrieval_query_plan.build_query_plan``), then the hybrid
retriever with that plan (``HybridRetriever.retrieve(question,
query_plan=plan)``) — against the REAL stores (Qdrant + Neo4j +
LiteLLM query embeddings). Prints, per question, what QU decided
(intent, linked nodes, gating, sub-questions / rewrites) and what
retrieval returned (top-6 with vec / graph / exact / final scores),
plus the same question WITHOUT a plan, so the QU effect is visible
side by side.

What this verifies that ``make eval-retrieval-live`` does not: the
live eval runs the golden set through the retriever WITHOUT a plan.
This script is the only runner that exercises retrieval + QU together
on the live stores.

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
    .venv/bin/python scripts/manual/verify_retrieval_live.py --output reports/retrieval.json

Exit codes: 0 = the run completed (plumbing OK); 1 = a question
raised unexpectedly (listed at the end); 2 = preflight failed
(missing embedding config or Qdrant, same as the live eval).
Per-question MATCH / MISS lines are OBSERVATIONS, not a gate:
intent accuracy on the 3B model was 0.500 on the old fake setup,
so a mismatch is a model-quality signal, not a code failure.
The numbers live in ``make eval-retrieval-live``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
# intent / chunk ids are the expectation to compare against, not a
# gate (see the module docstring).
SAMPLE_IDS = ["L01", "L06", "M01", "C01", "A01", "U05"]


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


def run_one(retriever, question: str, *, gold: dict | None = None,
            compare: bool = True) -> dict:
    """One question through QU + retrieval. Returns the verdict dict."""
    started = time.perf_counter()
    verdict: dict = {"question": question}

    seeds = gazetteer_seeds(question)
    expansion = expand_via_graph_query(seeds)
    try:
        adjacency = build_live_adjacency(seeds, expansion)
        adjacency_source = "live"
    except Exception as exc:  # noqa: BLE001 — degraded path, reported
        print(f"  [note] live adjacency failed ({exc}); "
              "falling back to expansion-only adjacency")
        try:
            adjacency = build_expansion_adjacency(seeds, expansion)
            adjacency_source = "expansion-only"
        except Exception as exc2:  # noqa: BLE001 — last resort
            print(f"  [note] expansion adjacency failed ({exc2}); "
                  "using empty adjacency")
            adjacency = {}
            adjacency_source = "empty"

    plan = build_query_plan(
        question,
        retriever=retriever,
        embeddings=retriever.embeddings,
        adjacency=adjacency,
        doc_links=retriever._resolved_doc_links(),
    )
    ranked, trace = retriever.retrieve(question, query_plan=plan)

    print(f"Q: {question}")
    print(f"  seeds: {[s['key'] for s in seeds]} "
          f"(expansion nodes: {expansion.get('nodes_returned', 0)}, "
          f"adjacency: {adjacency_source} {len(adjacency)} nodes, "
          f"graph_missing: {trace.get('graph_missing')})")
    print(f"  plan: intent={plan.intent} confidence={plan.confidence:.3f} "
          f"gated_off={plan.gated_off} "
          f"fallback={plan.fallback_reason}")
    print(f"  plan linked: {[n['key'] for n in plan.linked_nodes]}")
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

    if compare:
        ranked_np, trace_np = retriever.retrieve(question)
        same = ranked_np[:6] == ranked[:6]
        print(f"  top-6 without plan: "
              f"{'SAME' if same else 'DIFFERS'}")
        if not same:
            _print_top6("top-6 (no plan)", ranked_np, trace_np)
        verdict["no_plan_same"] = same
    verdict["ranked"] = ranked[:6]
    verdict["intent"] = plan.intent
    verdict["gated_off"] = plan.gated_off
    verdict["graph_missing"] = trace.get("graph_missing")

    if gold is not None:
        gold_intent = gold.get("intent")
        intent_match = plan.intent == gold_intent
        gold_chunks = set(gold.get("gold_chunk_ids", []))
        hit = bool(gold_chunks & set(ranked[:6]))
        print(f"  gold: intent={gold_intent} "
              f"{'MATCH' if intent_match else 'MISMATCH'}; "
              f"gold chunk in top-6: {'HIT' if hit else 'MISS'} "
              f"(gold: {sorted(gold_chunks)})")
        verdict["gold_intent"] = gold_intent
        verdict["intent_match"] = intent_match
        verdict["gold_hit"] = hit

    verdict["seconds"] = round(time.perf_counter() - started, 2)
    print(f"  ({verdict['seconds']}s)")
    print()
    return verdict


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
    parser.add_argument("--output", default=None,
                        help="Write full verdicts JSON.")
    args = parser.parse_args()

    try:
        inventory = preflight_live_stores()
    except LivePreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}")
        return 2
    print(format_inventory(inventory))
    print()

    retriever = build_hybrid_retriever()
    gazetteer = refresh_gazetteer_codes()
    print(f"Gazetteer refresh: {gazetteer}")
    print()

    cases: list[dict] = []
    if args.question:
        cases = [{"question": args.question}]
    elif args.golden_set:
        cases = load_golden_set()["cases"]
    else:
        golden = {c["id"]: c for c in load_golden_set()["cases"]}
        cases = [golden[cid] for cid in SAMPLE_IDS]

    compare = not args.no_compare
    verdicts: list[dict] = []
    errors: list[str] = []
    for case in cases:
        try:
            verdicts.append(run_one(
                retriever, case["question"],
                gold=case if "id" in case else None,
                compare=compare,
            ))
        except Exception as exc:  # noqa: BLE001 — one bad case must not kill the run
            print(f"  [ERROR] {case.get('id', '?')}: {exc!r}")
            print()
            errors.append(f"{case.get('id', '?')}: {exc!r}")

    with_gold = [v for v in verdicts if "gold_intent" in v]
    if with_gold:
        intent_ok = sum(1 for v in with_gold if v["intent_match"])
        hits = sum(1 for v in with_gold if v["gold_hit"])
        print(f"Summary: {len(verdicts)} questions, "
              f"intent MATCH {intent_ok}/{len(with_gold)}, "
              f"gold chunk HIT {hits}/{len(with_gold)}")
    else:
        print(f"Summary: {len(verdicts)} questions (no gold expectations)")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for line in errors:
            print(f"  - {line}")
        return 1

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"inventory": inventory, "verdicts": verdicts},
            indent=2, ensure_ascii=False, default=str,
        ))
        print(f"Full verdicts written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
