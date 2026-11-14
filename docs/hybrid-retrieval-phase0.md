# Hybrid retrieval — Phase 0: golden set + harness + baseline

Report: `~/workspace/research_notes/hybrid-vector-graph-retrieval-20261114-0430/report.md`
(§8 Phase 0, §6 evaluation). Pilot mission: `fault_analysis`.

## Files

| File | What it is |
|---|---|
| `scripts/golden_set_fault_analysis.json` | Golden set, version `phase0-v1`, 36 cases. Chosen home: next to the other eval script (`scripts/eval_mission.py`), because it is eval data, not a seed — it is *verified against* `seeds/demo_data.py`, never loaded by the seeders. |
| `runtime/retrieval_metrics.py` | Pure metrics: recall@k, nDCG@k (binary relevance), MRR. No infra. |
| `runtime/retrieval_harness.py` | `load_golden_set` / `validate_golden_set`, `evaluate_retriever`, `log_to_mlflow` (never raises), `run_retrieval_eval`, `format_results_table`. |
| `runtime/retrieval_baseline.py` | Deterministic plain-vector baseline, fakes only. |
| `scripts/eval_retrieval.py` | CLI: `python -m scripts.eval_retrieval [--tracking-uri URI] [--output FILE]`. Also `make eval-retrieval-phase0`. |
| `tests/test_retrieval_phase0.py` | Main-feature tests only. |

## Golden-set grounding rules (how each case was built)

- Every case is grounded in a real row of `seeds/demo_data.py`: a fault
  (`FAULTS`), maintenance row, sensor reading, one of the 7 Qdrant
  documents (`DOCUMENTS`), or an ontology relation (`RELATIONS`).
- `gold_chunk_ids` reference the 7 seeded Qdrant document ids. Note the
  corpus really has **7** documents, not 6: the report says 6 in one
  place, but `DOCUMENTS` lists 7 (the 7th is `doc-o23-overview`).
- `gold_node_ids` reference real ontology node keys (endpoints of
  `RELATIONS`). Only 3 documents are ontology `Document` nodes
  (`doc-door-sensor`, `doc-motor-overheat`, `doc-brake-inspection`,
  all `DOCUMENTED_BY` from O21); the other 4 documents have no Document
  node, so their cases name the Asset/Component instead. Case M05
  deliberately records that asymmetry: the motor-overheating doc is the
  right chunk for O22's F-206, but its only graph link is O21.
- Stratification (36 cases): lookup 12, multi-hop 8, comparison 6,
  aggregate 4, unanswerable 6 (16.7%, inside the 15–20% band).
  4 cases are tagged `sibling_confusion: true` — near-identical wording
  on different assets (O21 vs O22: F-204 92C vs F-206 88.5C; F-410 vs
  F-207, whose seed descriptions are word-for-word identical). Phase 1's
  exit criterion measures this subset specifically.
- Unanswerable cases are genuinely unanswerable from the demo corpus
  (warranty period, fire alarm on healthy O23, brake-pad part number,
  B2 floor count — an ontology property, not document text — hydraulic
  leak on retired O20, firmware update procedure). Each states why in
  `notes`. They carry empty `gold_chunk_ids` and are **excluded** from
  the recall/nDCG/MRR means; the harness reports them separately
  (count + mean candidates returned).

## Harness decisions

- Retriever contract: `question -> ranked ids`, or
  `(ranked ids, trace_dict)`. The trace may carry `latency_seconds`;
  otherwise the harness measures wall time.
- Aggregate means are over **answerable cases only**. Per-query-type
  breakdown and the sibling-confusion subset use the same rule.
- MLflow logging follows `scripts/eval_mission.py` (params + metrics +
  `per_query_traces.json` artifact + the golden set file as an
  artifact, experiment `agentic-runtime-retrieval`). It never raises:
  no tracking URI → skipped; unreachable server → warning, results
  still returned. Fail-fast HTTP defaults mirror
  `infra/mlflow_setup.py` (`MLFLOW_HTTP_REQUEST_MAX_RETRIES=0`,
  10 s timeout).
- `k_context=6` (the context size) and `k_pool=20` (pre-rerank pool).

## Baseline decisions (all fakes — no live LLM/Qdrant/Neo4j in this VM)

- `DeterministicFakeEmbeddings`: bag-of-words vectors over a fixed
  vocabulary built **from the seeded document texts only**, never from
  the golden-set questions, so the baseline cannot be tuned to the
  questions it is scored on. L2-normalised; hyphenated codes (`f-204`)
  stay whole tokens. This is a lexical-overlap proxy for a real
  embedding model, not a measurement of one.
- `FakeQdrantClient`: in-memory points for the 7 seeded docs, cosine
  search, ties broken by document id (deterministic — e.g. the
  all-zero query vector for out-of-vocabulary questions).
- The baseline calls the **real** `infra.qdrant_client.search_documents`
  with injected fake embeddings + fake client, so it exercises the
  production search code path instead of reimplementing search.
- It retrieves `limit=20` (the whole 7-doc corpus), so Recall@20
  saturates by construction on this corpus; **Recall@6 / nDCG@6 / MRR
  are the discriminating metrics** until the corpus grows.
- Baseline numbers are fake-setup numbers. Live numbers must be
  re-measured on the user's machine with the real embedding model and
  a seeded Qdrant (see "Live verification" below).

## Baseline numbers (fake setup)

Run `python -m scripts.eval_retrieval` to reproduce. Recorded
2026-11-14 (golden set `phase0-v1`, n=36, answerable=30,
unanswerable=6; mean latency ~0.01–0.02 s/query — fake wall-clock,
varies run to run, not a serving measurement):

| Slice | Recall@6 | nDCG@6 | MRR | Recall@20 |
|---|---|---|---|---|
| Overall (answerable only) | 1.000 | 0.942 | 0.922 | 1.000 |
| lookup (n=12) | 1.000 | 1.000 | 1.000 | 1.000 |
| multi-hop (n=8) | 1.000 | 0.954 | 0.938 | 1.000 |
| comparison (n=6) | 1.000 | 0.893 | 0.861 | 1.000 |
| aggregate (n=4) | 1.000 | 0.822 | 0.750 | 1.000 |
| sibling-confusion subset (n=4) | 1.000 | 1.000 | 1.000 | 1.000 |

Unanswerable cases (n=6): excluded from the means above; the
baseline returned all 7 docs for each (mean 7.0 candidates) — a
vector-only retriever cannot abstain, which is expected, not a bug.

Read these as a ceiling, not a measurement: with only 7 docs in the
corpus, Recall@6 is 1.000 whenever the gold docs are anywhere in the
top 6, and the BoW stand-in aces sibling cases because the questions
literally contain the distinguishing codes (`F-204` vs `F-206`).
Real embeddings blur codes, so the live sibling-subset numbers are
where Phase 1 must actually show its gain. Do not quote these as live
retrieval quality.

## What Phase 1 reuses

- `evaluate_retriever` takes any retriever callable, so the Phase 1
  fused retriever (dense + BM25 + 1-hop expansion + weighted fusion)
  drops in unchanged and its numbers are a direct diff against the
  Phase 0 baseline. `gold_node_ids` in the per-query traces are already
  recorded so graph-signal scoring (G1–G7) can be added without
  changing the golden set. The `sibling_confusion` subset is the
  Phase 1 exit-criterion slice.

## Live verification (user's machine)

1. Seed Qdrant with the real embedding model (`make seed` path), then
   run the real vector retriever through the same harness — the
   baseline numbers above must be replaced by live ones.
2. Point the harness at a live MLflow server:
   `python -m scripts.eval_retrieval --tracking-uri <uri>` and confirm
   the `agentic-runtime-retrieval` experiment shows the run with the
   per-query trace artifact.
3. Grow the corpus before trusting Recall@20: with 7 docs it is
   saturated by construction.
