# Hybrid retrieval — Phase 3: gated expansion, decomposition, LTR logging

Report: `~/workspace/research_notes/hybrid-vector-graph-retrieval-20261114-0430/report.md`
(§2 expansion methods, §4.1 feature rows, §4.4d LTR, §8 Phase 3).
Phase 0: `docs/hybrid-retrieval-phase0.md`. Phase 1:
`docs/hybrid-retrieval-phase1.md`. Phase 2:
`docs/hybrid-retrieval-phase2.md`. Consolidated as-built design:
`docs/hybrid-retrieval.md`. Pilot mission: `fault_analysis`.

## What was built

| File | What it is |
|---|---|
| `runtime/retrieval_expansion.py` (new) | Gating rule, decomposition detection, pluggable generators (live LLM + stub), mean-merge + rerank through the existing fusion, `ExpandedRetriever`. |
| `runtime/retrieval_metrics.py` | `precision_at_k` (binary relevance). `evaluate_ranking` untouched — a Phase 0 test pins its exact key set. |
| `runtime/retrieval_harness.py` | Precision@6 in per-query rows, aggregates, per-type breakdown, sibling subset, `format_results_table`, MLflow metrics; `ltr_feature_rows` + `ltr_pair_count`; `log_to_mlflow(..., ltr_rows=...)` logs `ltr_feature_rows.json`. |
| `runtime/retrieval_hybrid.py` | Refactor only: `retrieve` split into `_candidate_rows` (pre-fusion rows) + `_rank_candidates` (RRF → fusion → guardrails), `trace_feature_rows` (full per-candidate rows for LTR logging), `arms` on each candidate. No score changed — Phase 1/2 evals reproduce their documented numbers exactly. |
| `scripts/eval_retrieval_phase3.py` (new) | Expansion A/B off vs on, both fake setups + per-type + gated-subset breakdown, Precision@6, per-variant overlap stats, LTR pair count. `make eval-retrieval-phase3`. |
| `tests/test_retrieval_phase3.py` (new) | Main-feature tests only (23 tests). |
| `Makefile` | New `eval-retrieval-phase3` target. |

Changed nothing else. Untouched on purpose: `app/config.py`,
`runtime/llm_factory.py`, the embedding paths in
`infra/qdrant_client.py`, `search_documents`, the
`retrieval_context` hook, `services/user_service/`, and every
other mission. No new dependency was added.

## 1. Gating rule — exact, as built

Expand iff ANY of (measured on the base retriever's PRE-fusion
candidate rows — the vector arm's top hits before reranking):

| Trigger | Threshold (named constant) | Measured basis (fake setups) |
|---|---|---|
| Low top-1 | `top1_vec < LOW_TOP1_VEC = 0.35` | Confident lookups 0.50–0.66; weak cases L06 0.21, A02 0.23, A03 0.17, M01/C01/C06 0.29–0.32 |
| Small margin | `top1_vec − top2_vec < SMALL_MARGIN_VEC = 0.15` | M01 0.10, M04 0.08, M05 0.03, C01 0.03, A03 0.01 (blurred) |
| Gazetteer miss | `gazetteer_seeds(question)` empty | L06, L11, A02, A03, U04 — no code for the graph arm to seed on |

The decision (triggered, reasons, top-1/top-2/margin, exact-match
flag) is in every expanded trace under `trace["expansion"]
["gating"]`, so WHY each query expanded is auditable. Thresholds
are cosine values on the configured embedding model — re-tune
them live; the fake BoW scores are not a live calibration.

**Known limit:** gating sees the vector arm only. A01/A04 have a
confident vector top-1 (0.39–0.46) but gold docs that rank low
AFTER fusion (aggregate nDCG@6 0.838), so they do not expand
(A01 expands on BoW only, where its top-1 is 0.32). A fusion-aware
gate (e.g. top-1 final-score margin) is a live follow-up, not
built here.

## 2. Multi-query — pluggable generator, live prompt written, stubs verified

`ExpandedRetriever` takes a `paraphrase_generator` callable
(`question -> list[str]`, ≤3 variants). The DEFAULT is the live
implementation, `build_live_variant_generator`: it calls the chat
model (`runtime/llm_factory.build_chat_model`, LiteLLM → vLLM 3B)
with the constrained few-shot prompt in
`runtime/retrieval_expansion.py::EXPANSION_PROMPT` ("paraphrase
only; do not add facts or asset names not in the question", 3
paraphrases, one per line). The model is built LAZILY on first
use — constructing the generator never touches LiteLLM config —
and the output parser strips numbering/bullets, drops lines
identical to the original, dedups, and caps at 3.

**Verified only with stub generators in tests/eval** (no live LLM
in this VM): `stub_paraphrase_variants` — a synonym-swap variant
(word-boundary swaps only, codes/numbers copied verbatim) plus a
code-stripped variant (parenthetical codes removed). The eval and
the tests pass the stubs explicitly; `build_expanded_retriever`
is the stub-wired fake convenience, live use must construct
`ExpandedRetriever` directly so the live generators apply.

## 3. Decomposition — deterministic heuristic first

Detection fires iff ANY of: (a) ≥2 distinct asset codes
(`O21`+`O22`); (b) ≥2 distinct fault codes; (c) a comparison word
(vs/versus/compare(d)/comparison/hotter/colder/higher/lower/
difference(s)); (d) ≥2 distinct wh-words AND ≥2 distinct codes.
Reasons are logged per query (`trace["expansion"]["multi_part"]`).

Documented limits (tested, not fixed by widening the rule —
widening would fire on every two-clause lookup):

- UNDER-triggers when the second part names no code: M06
  ("Which building is O21 in, and which document covers its motor
  overheating?") is genuinely two-part but is NOT detected — it
  falls back to gated multi-query instead.
- OVER-triggers on multi-hop questions naming asset + component +
  fault (M01–M04: "Which document …, and what …") — defensible,
  since they ask for two facts, but they are not comparisons.
- "List all …" aggregates (A01) are NOT decomposed: one list
  request, one retrieval.
- The STUB splitter (`stub_decompose_question`, clause split on
  "and"/";") cannot split "Both X and Y …" (C02) or "Compare X
  with Y …" (C04, C05) — no "and" boundary to split on. When the
  decomposition generator returns the question unchanged, the
  retriever falls back to gated multi-query (C04 does; C02/C05
  are confident, so they do not expand at all). The live LLM
  generator is the fix for these wordings, not a smarter splitter.

Decomposed sub-questions are retrieved in sequence through the
same base retriever (parallel retrieval is a live optimisation —
the calls are independent and could be concurrent; the fake
corpus is too small for it to matter) and merged exactly like
paraphrase variants.

## 4. LTR feature logging — structure only, NO training

Every evaluated query logs every candidate's FULL §4.1 feature
row via `runtime/retrieval_harness.ltr_feature_rows`: G1–G9
(G3 = best-path relation list, G9 = exact-code feature,
`bm25_rank: null` — no BM25 in Phase 1, recorded as absent rather
than invented), vec / graph / final scores, ranks before (RRF)
and after (fusion) reranking, provenance (arms + variants), the
expansion metadata (triggered, kind, variants, reasons), and the
binary label (1 iff gold). Logged to MLflow as the
`ltr_feature_rows.json` artifact (blurred ON run) and included in
the eval `--output` JSON.

**Labelled-pair count: 252 total (210 from answerable cases) vs
~3,000 needed for LambdaMART — LTR training is PENDING DATA.**
36 cases × 7 candidates is the whole yield of this golden set;
reaching ~3,000 needs either ~430 golden cases or a larger corpus
with more candidates per query (the live corpus, not this 7-doc
demo). There is NO training code in this repo: no model, no
trainer, no learned weights — the tests assert `train_ltr`
exists nowhere on the Phase 3 surface.

## 5. Precision@k

`runtime/retrieval_metrics.precision_at_k` (binary relevance,
denominator = k). On this corpus a single-gold case scores at
most 1/6, so Precision@6 is a *set* metric (did the top-6 keep
the gold docs and exclude distractors), complementary to
Recall@6 (same numerator, denominator = |gold|). It is reported
alongside Recall@6/nDCG@6 in the Phase 3 eval and in the harness
tables. `evaluate_ranking` was NOT extended (a Phase 0 test pins
its key set); Precision@6 is computed in the harness instead.

## 6. Merge rule — mean, not max (prototyped)

Per candidate: `vec_score` / `graph_score` = MEAN across the
original question + all variants; `exact` = MAX; the §4.1
feature row, best path and arms come from the variant row with
the highest `graph_score`. The mean is deliberate: a max lets
one drifting paraphrase single-handedly promote a doc, and on
the blurred setup max-merge regressed nDCG@6 0.958 → 0.957
while mean-merge held it at 0.958 (BoW: 0.958 → 0.965). The
merged set is reranked by the SAME fusion (RRF k=60 → top-30 →
weighted fusion → guardrails → top-6). Expansion never invents
candidates and never bypasses the fusion.

## 7. Numbers — expansion A/B, both fake setups (golden set phase0-v1, n=36/30)

Run: `python -m scripts.eval_retrieval_phase3` (exit code 0 =
exit criterion passed). All numbers are fake-setup numbers with
STUB generators — they measure the merge/rerank plumbing and the
gating, not model-written paraphrases.

| Setup | Expansion | Precision@6 | Recall@6 | nDCG@6 | MRR |
|---|---|---|---|---|---|
| BoW | off | 0.217 | 1.000 | 0.958 | 0.940 |
| BoW | **on** | 0.217 | 1.000 | **0.965** | 0.957 |
| Code-blurred | off | 0.217 | 1.000 | 0.958 | 0.940 |
| Code-blurred | **on** | 0.217 | 1.000 | **0.958** | 0.940 |

Per-type, code-blurred (off → on nDCG@6): lookup 1.000 → 1.000,
multi-hop 1.000 → 1.000, comparison 0.898 → 0.898, aggregate
0.838 → 0.838 — identical everywhere. BoW per-type: aggregate
0.838 → 0.890 is the whole BoW gain (MRR 0.750 → 0.875); the
rest identical.

Gated subset (cases where expansion triggered — same ids off vs on):

| Setup | Gated cases (answerable) | Precision@6 off → on | Recall@6 | nDCG@6 off → on |
|---|---|---|---|---|
| BoW | 11: M01, M04, M05, M06, M07, C01, C03, C04, C06, A01, A02 | 0.242 → 0.242 | 1.000 → 1.000 | 0.917 → **0.936** |
| Code-blurred | 10: M01, M04, M05, M06, M07, C01, C03, C04, C06, A02 | 0.183 → 0.183 | 1.000 → 1.000 | 0.931 → **0.931** |

Trigger counts: BoW 13/36 (decomposition 6, multi-query 7),
blurred 12/36 (decomposition 6, multi-query 6). Mean variants
per triggered query 1.62 (BoW) / 1.67 (blurred). Mean top-6
Jaccard vs base 0.886 / 0.877 — variants DO move the ranking
(the A/B is not a no-op: ~12% of top-6 slots change). Mean
candidate Jaccard vs base 1.000 — by construction on this corpus
(the vector arm returns all 7 docs for every variant), reported
as measured, not as a finding.

**Exit criterion: PASS.** No Precision@6 loss and no Recall@6 /
nDCG@6 regression on either setup, overall and on the gated
subset. **Verdict, stated plainly: no measurable gain on this
7-doc corpus; expansion value unproven until a larger corpus /
live data.** On the exit-criterion (blurred) setup the numbers
are IDENTICAL off vs on (nDCG@6 0.958, gated 0.931); the only
movement anywhere is BoW aggregate 0.838 → 0.890 (+0.052, one
ranking swap on 4 cases) and gated BoW nDCG 0.917 → 0.936.
Recall@20 is 1.000 by construction and is not claimed as a gain.
The plumbing is verified (gating fires on the right cases,
variants move rankings, merge + fusion hold the line); whether
real paraphrases from the 3B model HELP on real data is still
unmeasured — that is the live A/B in §8.

## 8. Live verification (user's machine)

Same standing items as Phases 1–2, plus Phase 3 specifics — the
full consolidated checklist is in `docs/hybrid-retrieval.md` §9;
the Phase 3 items are:

1. Run the golden set live through each phase's eval script
   (`make eval-retrieval-phase1/2/3`) with the real embeddings +
   Qdrant + Neo4j, and replace every fake number above with live
   ones. Re-run the w_graph grid live with **0.50 first** (the
   first-to-recheck-live runner-up in Phases 1–2).
2. Swap the stub generators for the live ones
   (`ExpandedRetriever(base)` with no generator arguments) and
   re-run the Phase 3 A/B: real 3B paraphrases may drift where
   the stubs cannot, so the live exit check is the same
   criterion (no Precision@6 loss, no Recall@6/nDCG@6
   regression, gated subset reported separately) — and a live
   gain, if any, should show up on the gated subset first.
3. Re-tune the gating thresholds live (LOW_TOP1_VEC 0.35 /
   SMALL_MARGIN_VEC 0.15 are fake-BoW cosine values): log the
   top-1/margin distribution on live traffic, pick thresholds
   that expand ~10–30% of queries, and confirm confident
   lookups stay unexpanded (expansion costs 1 LLM call + N
   embeddings + N searches per query).
4. Confirm the live decomposition generator splits the wordings
   the stub cannot (C02 "Both X and Y", C04/C05 "Compare X with
   Y") and that its sub-questions never add asset names/codes
   not in the question (spot-check the prompt's constraint live).
5. Grow the corpus / golden set before trusting any expansion
   claim — and before LTR: 252 pairs is 8% of the ~3,000-pair
   threshold. Do NOT train a ranker on this golden set.
6. Point the eval at the live MLflow server
   (`python -m scripts.eval_retrieval_phase3 --tracking-uri <uri>`)
   and confirm the `agentic-runtime-retrieval` experiment shows
   the runs, including the `ltr_feature_rows.json` artifact.
