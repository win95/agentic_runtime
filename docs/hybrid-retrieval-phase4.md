# Hybrid retrieval — Phase 4: Query Understanding (QueryPlan)

> **REMOVED IN PART 2026-11-14 (user option B — delete the
> fakes entirely).** This phase's eval script
> (`scripts/eval_retrieval_phase4.py`,
> `make eval-retrieval-phase4`) is DELETED. The QueryPlan /
> intent / guardrail code survives. The numbers in §6 are
> HISTORICAL fake-setup measurements and are NOT
> reproducible; the only retrieval eval left is
> `make eval-retrieval-live`.

Report: `~/workspace/research_notes/hybrid-vector-graph-retrieval-20261114-0430/report.md`
(§2 expansion methods, §8 Phase 4).
Phase 0: `docs/hybrid-retrieval-phase0.md`. Phase 1:
`docs/hybrid-retrieval-phase1.md`. Phase 2:
`docs/hybrid-retrieval-phase2.md`. Phase 3:
`docs/hybrid-retrieval-phase3.md`. Consolidated as-built design:
`docs/hybrid-retrieval.md`. Pilot mission: `fault_analysis`.

## What was built

| File | What it is |
|---|---|
| `runtime/retrieval_query_plan.py` (new) | `QueryPlan` dataclass, `build_query_plan`, rewrite guardrails `check_rewrite_variants`. No LLM, no Qdrant/Neo4j; live generators are lazy (see §2). |
| `runtime/retrieval_hybrid.py` | Plan seam: `HybridRetriever.retrieve(question, query_plan=None)` + `_candidate_rows(question, seeds=None)` + `_exact_match(..., seeds=None)` + the shared `_trace` helper. The no-plan path is the old `retrieve` body moved verbatim into `_retrieve_no_plan` — byte-identical (see §5). |
| `runtime/retrieval_graph.py` | (Worker A) gazetteer normalisation only — diacritics-insensitive, separator variants at confidence 0.8, exact hits byte-identical. |
| `runtime/retrieval_intent.py` | (Worker B) `score_intents` — fusion-style intent scoring, deterministic, no LLM. |
| `runtime/retrieval_decompose.py` | (Worker C) `decompose_question` + `relatedness_verdict` — RELATED iff BFS distance ≤2 between parts' seed sets; `related=None` when no verdict exists. |
| `scripts/golden_set_fault_analysis.json` | Every case gains an `"intent"` label (deliberate per-case judgement, §4). No other field changed; `load_golden_set` ignores the new field, so Phase 0–3 evals still load the set. |
| `scripts/eval_retrieval_phase4.py` (new) | ~~Plan off vs on, both fake setups + link accuracy, intent accuracy, rewrite faithfulness, downstream nDCG@6 delta. `make eval-retrieval-phase4`.~~ **DELETED 2026-11-14** (fake removal). Use `make eval-retrieval-live`. |
| `tests/test_retrieval_phase4.py` (new) | Main-feature tests only (21 tests). |
| `Makefile` | ~~New `eval-retrieval-phase4` target (+ `.PHONY`).~~ Target removed 2026-11-14 (fake removal). |

Changed nothing else. Untouched on purpose: `search_documents`,
the `retrieval_context` hook, the route node, skill selection,
`app/config.py`, `runtime/llm_factory.py`, the embedding paths in
`infra/qdrant_client.py`, `services/user_service/`, and every other
mission. No new dependency was added. Opt-in live wiring is NOT
done — it is the follow-up in §8.

## 1. QueryPlan — what it is

One object per question, built in four steps
(`runtime/retrieval_query_plan.py::build_query_plan`):

1. **Link** — `gazetteer_seeds(question)` (Worker A).
2. **Intent** — `score_intents(question, seeds, embeddings,
   adjacency)` (Worker B): six intents, fusion-style
   `final = 0.50·z(vec) + 0.30·graph + 0.20·exact`, ties break in
   `INTENT_ORDER` (lookup first, unanswerable last). Inputs are
   INJECTED since the 2026-11-14 fake removal — `score_intents`
   now takes `embeddings`, `asset_data`, `doc_links`,
   `vocabulary` (plus `seeds` / `adjacency`); with no
   embeddings the vector signal is 0 for every intent and
   graph + exact carry the score.
3. **Decompose** — `decompose_question` (Worker C), only when the
   gate opens (see §2).
4. **Gate** — `gating_decision` on the base retriever's PRE-fusion
   candidate rows (the Phase 3 rule: top-1 < 0.35, or margin <
   0.15, or gazetteer miss).

Fields: `intent`, `linked_nodes` (seeds with confidences),
`sub_questions` (`[{"text", "linked_nodes": [keys]}]`),
`rewrite_variants` (guardrail-kept paraphrases), `confidence`
(the intent margin), `gated_off`, `fallback_reason`, plus audit
fields `intent_scores`, `gating`, `is_multi_part` /
`multi_part_reasons`, `related` / `shared_nodes`.

## 2. Gating interpretation — the decision this phase documents

Gating controls the expensive / risky parts ONLY — rewrite
generation and decomposition-driven expansion. Intent scoring and
linking are cheap and deterministic, so they run ALWAYS: a
gated-off plan still carries the scored intent and the linked
nodes as-is, plus NO rewrite variants (`gated_off=True`,
`fallback_reason="confident — no expansion"`). This satisfies
"confident lookups pass through untouched" (no rewrite, no
expansion) WITHOUT mislabelling a confident question: a
confident comparison keeps its scored intent (`comparison`), it
is not relabelled `lookup` just because it did not expand.

Two consequences, both deliberate and different from Phase 3:

* **Confident multi-part questions no longer decompose.** In
  Phase 3, decomposition bypassed gating: a detected multi-part
  question was decomposed even when confident (C02/C03/C05,
  M07). In Phase 4 the gate is single: a gated-off plan has NO
  sub-questions either — the decomposition generator (a live LLM
  call) is not invoked for a question the vector arm already
  answers confidently. Detection itself (`detect_multi_part`,
  cheap regex) still runs always and is recorded on the plan
  (`is_multi_part` / `multi_part_reasons`), so the audit trail
  shows what was NOT expanded and why. Measured effect,
  verified per case: the four questions' rankings are IDENTICAL
  with or without decomposition on this corpus — skipping them
  costs nothing here. BoW plan-on nDCG@6 is 0.962 vs Phase 3's
  expanded 0.965, and the whole gap is a different case: A02
  (see the linking-failure note below), not these four.
* **Decomposition takes precedence when the gate opens** (the
  Phase 3 pattern): if the generator splits the question, the
  plan carries the sub-questions and no rewrite variants;
  paraphrases are the fallback when the question is not
  multi-part or the generator cannot split it (C04 under the
  stub splitter: the stub returns the question unchanged, so
  C04 falls back to paraphrases).

**No retriever, no vector gate.** When `build_query_plan` is
called with `retriever=None`, gating falls back to
gazetteer-miss-only: the plan expands iff linking found nothing.
The unmeasured vector values are `None` in `plan.gating` (never
invented zeros).

**Linking failure.** When linking fails entirely
(`linked_nodes == []`) the plan says so explicitly
(`fallback_reason="no confident link"`) and carries NO rewrite
variants, ever (guardrail (c)). Decomposition sub-questions MAY
still be present in that case — they are retrieved as written
through the vector arm; the linking-failure rule covers
rewrites only, not splits. (No golden case hits the
seedless-plus-multi-part combination, so this is untested on
the golden set — a live edge to watch.) Measured consequence
on the golden set: seedless A02 expands in Phase 3 (multi-query
variants retrieved) but not in Phase 4 (no variants generated
at all), and it is the ONLY case whose ranking differs between
Phase 3-on and Phase 4-on on either setup — BoW nDCG@6 1.000 →
0.920 (the rope doc drops from 2nd to 3rd), which is the whole
BoW gap 0.965 → 0.962. L06 and A03 are also seedless, but their
stub generates zero variants anyway, so Phase 3 did not expand
them either.

## 3. Rewrite guardrails — pure checks, HyDE excluded

`check_rewrite_variants(question, variants, *, linked_nodes,
adjacency) -> (kept, dropped, reasons)`, reason codes
`missing_entities`, `missing_fault_codes`, `missing_numbers`,
`drift`, `no_seeds`, `over_cap`. A variant is KEPT only if it
passes all of:

* **(a) Exact-preservation floor** — it links to every seed the
  original linked to (`gazetteer_seeds` on both sides, so a
  normalised re-mention counts), it keeps every fault code the
  original names (`FAULT_CODE_RE` from `retrieval_expansion`,
  imported not copied), and it keeps every number-with-unit the
  original names (`NUMBER_RE` from `retrieval_intent`, imported
  not copied). Known limit: bare numbers with no unit ("20
  open/close cycles", "8 faults") are NOT in the floor —
  `NUMBER_RE` only matches a number with a unit — so a variant
  dropping one passes (a).
* **(b) Drift cap** — it adds no entity outside the original
  question plus its 1-hop neighbourhood:
  `gazetteer_seeds(variant) ⊆ original seeds ∪
  expand_one_hop(original seeds)`. The same subset rule applies
  to fault codes and numbers-with-unit: a variant may not name
  a fault code or a threshold the original did not name (a
  variant inventing "92C" is `drift`, even though "92C" is not
  a gazetteer node — this is an extension of (b) beyond
  gazetteer entities, same reason code).
* **(c) Linking-failure rule** — no seeds, every variant
  dropped, original used verbatim.
* **(d) Cap** — at most `MAX_VARIANTS` (3) kept, in generator
  order; the rest are dropped as `over_cap`.

**HyDE is EXPLICITLY EXCLUDED** — no hypothetical-document
generation anywhere. Variants come only from the existing
stub / live paraphrase generators. A structural test pins this
(the word "HyDE" appears exactly once in the module, in the
docstring sentence excluding it; there is no HyDE generator
and no `llm_factory` import).

Concrete stub behaviour (measured): the code-stripping stub
variant is the one that gets dropped — C04's stripped variant
("Compare the motor overheating remedy for O21 with O22 — is
the documented remedy the same?") drops the "(92C)" / "(88.5C)"
parentheticals and is dropped for `missing_numbers`. The
synonym-swap variant is kept wherever it preserves codes.

## 4. Golden-set intent labels — the labelling rule

**Rule:** the label is the question's primary ASK — what the
user wants done — judged from the wording alone, ignoring
`query_type` and ignoring whether the corpus can answer it.
`unanswerable` is NEVER used as a label: answerability is a
separate dimension from intent, and labelling the six U cases
`unanswerable` would just copy `query_type`. The five labels
used are `lookup` (a single fact / what a guide says),
`diagnosis` (cause / why / what usually causes a fault),
`procedure` (how-to / what must happen / steps), `comparison`
(compare two subjects / which is more / hotter / same?),
`aggregate` (list / enumerate / count a set).

| Cases | Label | Note |
|---|---|---|
| L01, L04, L09, L10, M07, U02, U05 | diagnosis | "What causes…", "How do you diagnose…", "What usually causes it", "what does the guide say it points to" (L10), "what indicates a worn encoder" (M07), "What caused the fire alarm" (U02), "root cause of the leak" (U05) |
| L02, L03, L05, M03, U06 | procedure | "What should be done…", "What must happen…", "What is the fix", "what must be measured after replacement" (M03), "firmware update procedure" (U06) |
| L06, L07, L08, L11, L12, M01, M02, M04, M05, M06, M08, U01, U03 | lookup | "What does the guide say…", "What is the normal range…", "Which document…", "What is the warranty period" (U01), "What is the part number" (U03) |
| C01–C06 | comparison | All six, explicit compare / hotter / more |
| A01–A04, U04 | aggregate | "List all…", "Which documents…", "all the documented causes…", "How many floors" (U04) |

**Ambiguous cases (judgement calls, documented not hidden):**
L01 asks for causes AND how to verify the fix — diagnosis
wins (the causes are the question). L10 asks what the guide
says high vibration "points to" — diagnosis (cause
attribution), though the scorer says procedure. L11 ("What
should be logged…") is lookup (a single fact), not procedure.
M03 ("Which document covers the inspection limits, and what
must be measured…") is procedure — the "what must be measured"
clause parallels L03 — though "which document" alone would be
lookup. M07 is diagnosis ("what indicates a worn encoder"),
M08 is lookup ("which document explains CAN bus diagnosis" —
"diagnosis" is the document's topic, not the ask). U04 is
aggregate ("How many floors…"), U05 is diagnosis (the scorer's
`unanswerable` prediction is the confusion, §6).

## 5. The retriever seam — plan in, same fusion out

`HybridRetriever.retrieve(question, query_plan=None)`:

* **No plan:** the old body, moved verbatim into
  `_retrieve_no_plan` — byte-identical (proved by the
  Phase 0–3 evals reproducing their documented numbers
  exactly, §7, and by a test asserting
  `retrieve(q) == retrieve(q, query_plan=None)` against a
  fresh retriever).
* **With plan:** the plan's `linked_nodes` seed the original
  question's rows (`_candidate_rows(question,
  seeds=plan.linked_nodes)` — the new optional `seeds`
  parameter; `None` re-derives from the question, `[]` means
  "no confident link", vector-only). Expansion, only when the
  plan is not gated off: sub-questions first, else rewrite
  variants — each retrieved via `_candidate_rows`, merged with
  `merge_candidate_rows` (the Phase 3 merge path: mean
  vec/graph, max exact), ranked by the same
  `_rank_candidates`. Sub-question rows re-derive their seeds
  from the sub-question text (identical to the keys the plan
  stores for them, confidences preserved); the original row
  uses the plan's seeds directly. The trace gains a
  `query_plan` block (intent, confidence, gating, seeds,
  sub-questions, variants, fallback reason) and an `expansion`
  block; every other trace key is as before.

Untouched on purpose: `search_documents`, the
`retrieval_context` hook, the route node, skill selection.

## 6. Numbers — plan off vs on, both fake setups (golden set phase0-v1, n=36/30) — HISTORICAL, not reproducible

~~Run: `python -m scripts.eval_retrieval_phase4` (exit code 0 =
exit criterion passed).~~ The eval script is deleted. All numbers are fake-setup numbers
with STUB generators — they measure the plan plumbing, the
gating and the guardrails, not model-written paraphrases.

### 6.1 Entity-link accuracy (entity nodes only)

Link accuracy is defined on ENTITY nodes only:
`gold_node_ids` minus `doc-*` entries (the `doc-*` entries are
documents, not linked entities). Per case it is the fraction of
gold entity nodes the plan's linked nodes cover; the headline
is the mean over the 36 cases, with the exact-set match rate
(linked set == gold entity set) alongside.

| Query type | n | Mean recall | Exact-set rate |
|---|---|---|---|
| lookup | 12 | 0.708 | 0.583 |
| multi-hop | 8 | 0.917 | 0.750 |
| comparison | 6 | 0.806 | 0.500 |
| aggregate | 4 | 0.500 | 0.500 |
| unanswerable | 6 | 0.792 | 0.667 |
| **Overall** | **36** | **0.762** | **0.611** |

Identical on BoW and code-blurred (linking is deterministic
and embedding-independent). Where recall is lost, the cause is
visible in the labels: L06/A02/A03 link nothing at all (no
code in the question — recall 0.0); L08/L10/L12 link the
component but not its asset (the question never names O21 —
recall 0.5); M06/C05/U04 link one of three/four gold entities
(the rest are ANSWER nodes — building-B1, motor-M2, brake-BK1
— that the question never mentions). **Ceiling note:** link
recall cannot reach 1.0 on this golden set, because gold
entity nodes include nodes the answer names, not just nodes
the question names. 0.762 is mention coverage, not a linker
failure rate.

### 6.2 Intent accuracy (overall + per query_type + gated subset)

Scored intent vs the §4 labels. Overall **18/36 = 0.500**,
identical on both fake setups (the prototypes are digit-free,
so blurring changes nothing).

| Query type | n | Accuracy | Misses (gold → scored) |
|---|---|---|---|
| lookup | 12 | 0.500 | L02 procedure→diagnosis, L03 procedure→lookup, L05 procedure→lookup, L07 lookup→diagnosis, L10 diagnosis→procedure, L11 lookup→comparison |
| multi-hop | 8 | 0.500 | M01 lookup→diagnosis, M05 lookup→comparison, M06 lookup→diagnosis, M08 lookup→diagnosis |
| comparison | 6 | 0.667 | C02, C03 comparison→diagnosis |
| aggregate | 4 | 0.500 | A02 aggregate→comparison, A03 aggregate→diagnosis |
| unanswerable | 6 | 0.333 | U01 lookup→diagnosis, U02 diagnosis→lookup, U05 diagnosis→unanswerable, U06 procedure→lookup |
| **Overall** | **36** | **0.500** | |
| Gated subset | 14 (BoW) / 12 (blurred) | 0.500 / 0.417 | |

**The unanswerable confusion, stated plainly:** only 2 of 6
unanswerable cases are scored as their ask (U03 lookup, U04
aggregate). The other four are exactly the confusion Worker B
documented: U01 (warranty) scores diagnosis — the data exists,
the asked detail does not, and the scorer only detects absence,
not silence; U02 (fire alarm on faultless O23) scores lookup —
its unanswerable graph is 1.0 but the lookup vector z
outweighs it; U05 (leak on O20) is the one case the scorer
calls unanswerable, but its gold label is diagnosis (the ask);
U06 (firmware procedure) scores lookup. Intent ≠
answerability: a question can ask for a diagnosis and be
unanswerable, and the scorer's `unanswerable` prediction is a
detection, not an intent. Do not read the 0.333 as "the scorer
finds unanswerable questions" — on this golden set it finds
one (U05) and that one is labelled diagnosis.

The other systematic miss pattern: **diagnosis swallows
lookup/procedure.** Ten of the eighteen misses are scored
diagnosis over a lookup/procedure/aggregate gold label
(L02, L07, M01, M06, M08, C02, C03, A03, U01, plus L10's
reverse). The BoW vector arm is lexical overlap over 174 doc
words, so fault-code / temperature topic words dominate it —
the same mechanism Worker B documented. Intent at 0.500 on
this corpus is a measurement of the fake setup, not a claim
that half of real questions are misrouted: no routing decision
in this repo consumes the intent yet (see §8).

For reference: only 14/36 golden cases match `query_type`
when scored (the brief's figure, counting multi-hop cases as
lookup — their document-seeking shape, per Worker B) — the
multi-hop cases are mostly diagnosis by intent, which is why
the labels in §4 were assigned deliberately per case instead
of copied from `query_type`.

### 6.3 Rewrite faithfulness

| Setup | Paraphrase cases | Generated | Kept | Preservation rate | Kept-variant re-check |
|---|---|---|---|---|---|
| BoW | 6 | 7 | 6 | 0.857 | PASS (100%) |
| Code-blurred | 4 | 5 | 4 | 0.800 | PASS (100%) |

"Paraphrase cases" are the cases where the plan actually
generated paraphrases (gate open, no decomposition, seeds
present). Seedless gated-open cases (L06, A02, A03) generate
no variants at all — the linking-failure rule fires before
any generator is called — so they are not in the denominator.
The one dropped variant on each setup is C04's code-stripped
variant (§3). The exit check is the kept-variant re-check:
every kept variant re-passes all guardrails, so faithfulness
on the preservation checks is 100% — the 0.857 / 0.800 is the
GENERATED pass rate (the stub's code-stripping variant being
caught, which is the guardrail working), not a kept-variant
violation.

### 6.4 Downstream nDCG@6 delta (with plan vs without)

| Setup | nDCG@6 off → on | Gated nDCG@6 off → on | Precision@6 | Recall@6 |
|---|---|---|---|---|
| BoW | 0.958 → **0.962** | 0.917 → **0.928** | 0.217 → 0.217 | 1.000 → 1.000 |
| Code-blurred | 0.958 → **0.958** | 0.930 → **0.930** | 0.217 → 0.217 | 1.000 → 1.000 |

Gate opened on 14/36 (BoW) / 12/36 (blurred) questions:
L06, L11, M01, M04, M05, M06, C01, C04, C06, A02, A03, U01
on both, plus A01 and U04 on BoW only (A01's blurred top-1 is
0.386, above the 0.35 threshold; U04's blurred top-1 is 0.356
with a seed now linked — see §7).

**Exit criterion: PASS.** Rewrite faithfulness 100% on the
kept-variant re-check, no nDCG@6 regression overall or on the
gated subset, on either setup.

**Verdict, stated plainly: no measurable downstream gain on
this 7-doc corpus — same as Phase 3.** On the exit-criterion
(blurred) setup the numbers are IDENTICAL off vs on (nDCG@6
0.958, gated 0.930). The only movement anywhere is BoW
(+0.004 overall, +0.011 gated, one ranking swap on the
aggregate type 0.838 → 0.870). The plan plumbing is verified
(gating fires on the right cases, guardrails catch the
code-dropping variant, the seam uses plan seeds, confident
questions pass through untouched with their intent intact);
whether QueryPlan HELP on real data is still unmeasured — that
is the live A/B in §8. Recall@20 is 1.000 by construction and
is not claimed as a gain.

## 7. Phase 0–3 reproduction (no-plan path byte-identical)

Re-ran all four phase evals after the Phase 4 changes
(`make eval-retrieval-phase0/1/2/3` — then runnable; the phase
eval scripts were deleted 2026-11-14, so this reproduction
cannot be re-run):

* Phase 0 baseline: nDCG@6 0.942 — as documented.
* Phase 1: blurred 0.906 → 0.942, sibling 0.908 → 1.000;
  grid 0.30 → 0.943, 0.40 → 0.943, 0.50 → 0.958 — exact.
* Phase 2: multi-hop 0.954 → 1.000, PPR ablation delta
  +0.003, overall 0.942 → 0.958 — exact.
* Phase 3: all headline and gated numbers reproduce exactly
  (BoW off → on 0.958 → 0.965, gated 0.917 → 0.936; blurred
  0.958 → 0.958, gated 0.931 → 0.931; LTR 252 pairs) —
  **except one delta, caused by Worker A's gazetteer change:**

  Blurred trigger count is now **11/36** (decomposition 6,
  multi-query 5), not the documented 12/36 (6 + 6); mean
  variants per triggered query 1.73 (was 1.67), mean top-6
  Jaccard vs base 0.866 (was 0.877). Cause, verified: U04
  ("How many floors does building B2 have…") is the ONLY
  golden case whose entire seed set comes from the variant
  pass — "building B2" (space-separated) now links to
  building-B2 at confidence 0.8, where the exact gazetteer
  linked nothing. Before, U04 was a gazetteer miss and
  triggered expansion; now it links, and its blurred top-1
  (0.356) and margin (0.188) are above the thresholds, so it
  does not expand. The other variant-only seeds (M04
  controller-C1, M05 motor-M3, U01 motor-M2, U06
  controller-C2) only add a second seed to questions that
  already linked — they change no gating decision and no
  headline number. U04 is unanswerable, so it is excluded
  from the nDCG means; the answerable gated set is identical,
  which is why every Phase 3 headline number still
  reproduces. The Phase 3 doc's trigger counts are now stale
  by this one case; the counts above are the current ones.

## 8. Live verification (user's machine)

Same standing items as Phases 1–3, plus Phase 4 specifics —
the full consolidated checklist is in
`docs/hybrid-retrieval.md` §11; the Phase 4 items are:

1. Run the golden set live via `make eval-retrieval-live`
   (live baseline + live hybrid through the same harness) and
   replace every fake number above with live ones. Expect link
   recall to be HIGHER live (real questions name codes in more
   spellings than the golden set's) and intent accuracy to move
   — the BoW prototype overlap that drives the diagnosis-swallow
   in §6.2 is a fake-setup artefact; a real embedding model may
   separate lookup / diagnosis / procedure better, or worse in
   new ways. Re-measure, do not assume. (The per-phase eval
   scripts are deleted, so there is no
   `make eval-retrieval-phase0/1/2/3/4` live run anymore.)
2. Swap the stub generators for the live ones
   (`build_query_plan` with no generator arguments, and
   `HybridRetriever.retrieve(question, query_plan=plan)`) and
   re-run the Phase 4 A/B: real 3B paraphrases may drift where
   the stubs cannot (invented thresholds, added entities), so
   the live exit check is the same criterion (kept-variant
   re-check 100%, no nDCG@6 regression, gated subset reported
   separately) — and the guardrail drop rate on live
   paraphrases is itself a finding (if the 3B drops codes
   often, the prompt, not the guardrail, needs fixing).
3. Re-tune the gating thresholds live (LOW_TOP1_VEC 0.35 /
   SMALL_MARGIN_VEC 0.15 are fake-BoW cosine values): log the
   top-1/margin distribution on live traffic, pick thresholds
   that expand ~10–30% of queries, and confirm confident
   lookups stay unexpanded (a plan costs 1 LLM call + N
   embeddings + N searches per expanded query; a gated-off
   plan costs one intent scoring + one linking, both
   deterministic and cheap).
4. Grow the golden set before trusting any QueryPlan claim —
   and specifically grow the QU (query understanding) part:
   36 cases with 5 intent labels and 12 link-miss cases is
   enough to measure the plumbing, not to tune intent weights
   or the 0.35/0.15 thresholds. Do NOT re-tune `W_VEC /
   W_GRAPH / W_EXACT` on this golden set (the 0.500 intent
   accuracy would invite exactly that overfit).
5. Opt-in wiring follow-up (NOT done here): add an opt-in
   mission-config flag, e.g. `execution.retrieval.query_plan:
   true` on `fault_analysis` only, read by the
   `retrieval_context` hook: when set, the hook builds a
   QueryPlan (live generators, real LiteLLM embeddings +
   Qdrant client + live Neo4j store), calls
   `build_hybrid_retriever().retrieve(question,
   query_plan=plan)`, assembles context with
   `runtime/retrieval_context.py::assemble_context`, and logs
   the plan (intent, margin, seeds, gating, fallback reason,
   kept/dropped variants with reasons) to MLflow. When unset,
   the hook behaves exactly as today. No change to
   `search_documents` / the tool surface is needed.
6. Point the live eval at the live MLflow server
   (`python -m scripts.eval_retrieval_live --tracking-uri
   <uri>`) and confirm the `agentic-runtime-retrieval`
   experiment shows the runs. (The old
   `python -m scripts.eval_retrieval_phase4` command is
   deleted with the fake eval.)
