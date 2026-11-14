# Hybrid retrieval — design AS BUILT (Phases 0–4)

Research report:
`~/workspace/research_notes/hybrid-vector-graph-retrieval-20261114-0430/report.md`
Per-phase docs: `docs/hybrid-retrieval-phase0.md` (golden set +
harness + baseline), `docs/hybrid-retrieval-phase1.md` (link +
fused rerank), `docs/hybrid-retrieval-phase2.md` (PPR + coverage +
hub), `docs/hybrid-retrieval-phase3.md` (gated expansion,
decomposition, LTR logging), `docs/hybrid-retrieval-phase4.md`
(QueryPlan: link + intent + gated expansion). Pilot mission:
`fault_analysis`.

This document is the consolidated as-built record. Where it
disagrees with the research report, **this document wins** — it
describes what the code actually does, including the locked
decisions that overruled the report's defaults (no cross-encoder,
no BM25, the 0.50/0.40/0.10 fusion). Do not quote the report's
§4.7 weights (0.50 ce / 0.30 graph / 0.15 vec / 0.05 exact) as
the system's weights: there is no cross-encoder, so the system
runs the adapted formula in §3 below.

Status: **evaluated on fakes only, NOT wired live.**
`search_documents` and the `retrieval_context` hook are untouched
in all five phases — no mission's behaviour changes. The opt-in
wiring follow-up is §12; the live checklist is §11.

## 1. Pipeline (one query)

```
question
  │  Phase 4 (QueryPlan — §6a): link (gazetteer, incl. normalised
  │  variants) + intent (fusion-style, deterministic) ALWAYS;
  │  then the single gate (Phase 3 rule) decides whether the
  │  expensive parts run: decomposition (sub-questions first) or
  │  paraphrase variants (≤3, guardrail-checked, HyDE excluded).
  │  A gated-off plan carries intent + links as-is, no variants.
  ▼
  Phase 3 (gated — §6): detect multi-part → decompose (≤3 sub-questions),
  else gate on vector confidence / gazetteer miss → paraphrase (≤3 variants).
  Each variant is retrieved like the original; rows are merged (mean)
  and reranked by the same fusion. No expansion by default.
  ▼
vector arm:  Qdrant top-40 dense (real LiteLLM embeddings live,
             deterministic BoW / code-blurred fakes in this VM)
graph arm:   gazetteer seeds (§4) → Neo4j 1-hop expansion (§5)
             → every doc linked to a seed or a 1-hop neighbour
candidates:  vector top-40 ∪ graph-linked docs, deduped by chunk_id,
             provenance kept per candidate (arms: vector / graph)
pre-fusion:  RRF (k=60) over vector rank + graph rank → top-30
rerank:      deterministic fusion (§3) → guardrails (§3) → top-6
context:     assemble_context (§7) — 3,000-token budget, best-first,
             ≤20 verbalised triples, never raw JSON
```

## 2. Vector-to-graph linking (Phase 1)

Every Qdrant point carries `document_id`, `chunk_id`,
`node_ids[]` (+ `asset_id` / `component_id` / `building_id`
where known) in its payload. Coverage on the demo corpus: **7/7
= 100%** (target ≥80%, enforced by `seeds/demo_data.py::validate()`).
`upsert_documents` passes extra payload keys through, so the seed
path just attaches them; the ingest path fills missing links
(`with_document_links`); existing points are backfilled
payload-only, no re-embedding
(`python -m scripts.backfill_qdrant_links`, idempotent).

`DOCUMENTED_BY` direction (locked): **Asset → Document** (the
asset is documented by the document), per the report's §4.2
direction note. Distance features (G1/G2) are computed on the
underlying undirected graph; the relation weight carries the
semantics. Grounded link table: `runtime/retrieval_linking.py::
DOCUMENT_LINKS`. Deliberate asymmetry: `doc-motor-overheat`
links to O21 / motor-M2 only — never to O22 / motor-M3 — even
though the remedy text fits O22's fault (golden case M05).

## 3. Fusion — the exact formula as built (Phases 1–2)

Per-candidate §4.1 features (G3 is the relation weight inside
G2; G9 is the exact-code feature):

| # | Signal | Definition as built |
|---|---|---|
| G1 | Min hop distance | 0 = linked entity IS a seed; 1 = 1-hop neighbour; None = unreachable within 1 hop |
| G2 | Best-path score | max over paths of `∏ w_rel × 0.6^hops`, §4.2 weights (DOCUMENTED_BY 1.00, HAS_COMPONENT/PART_OF 0.90, LOCATED_IN 0.40 / 0.15 via hub, RELATED_TO 0.25) |
| G4 | Entity-type match | +1 link to a seed component/asset, −1 link to a *different* component/asset, else 0 |
| G5 | Direct link | 1 = linked to a seed, 0.5 = linked to a 1-hop neighbour, 0 else; + `n_seeds_linked` |
| G6 | Seed coverage | \|{seeds within 2 hops}\| / \|S\|, raw [0, 1] (NOT z-scored) |
| G7 | PPR | Personalized PageRank, pure-Python power iteration, `restart_prob = 0.5`, seeds weigh `confidence × specificity` (spec = 1 / docs containing the seed); **log → min-max to [0, 1], NOT z-scored**; a document's PPR is its document node's score |
| G8 | Hub penalty | `log(1 + degree)` of the best path's transit node, ONLY when the transit node is a Building / generic Document; lives INSIDE the path term (`HUB_LAMBDA = 1.0`), so it saturates at z = −2 and can never dominate the fusion |
| G9 | Exact code match | 1 if a code the question names appears verbatim in the doc's title/content |

Phase 1 graph score (G1/G2/G4/G5):

```
graph_score = clip(path_score + 0.10*G4 + 0.10*(G5_direct − 0.5)
                   + hop_adj + 0.02*min(max(n_seeds_linked − 1, 0), 2),
                   −0.3, 1.2)
# hop_adj: G1 0 → +0.05, 1 → 0, None → −0.05. Floor −0.3 (not 0):
# a G4 mismatch must score BELOW an unlinked doc.
```

Phase 2 full graph score (locked):

```
path_term   = graph_score − 1.0 * g8_hub
graph_score = 0.45 * minmax_log(PPR) + 0.30 * z(path_term) + 0.25 * coverage
# z = per-query z-score clipped to [−2, 2]; w_ppr=0 is the PPR ablation
# (no renormalising — path:coverage stays 0.30:0.25).
```

Outer fusion (no cross-encoder — the "reranker" IS this formula):

```
z(x)  = per-query z-score, clipped to [−2, 2] (std 0 → z = 0)
final = 0.50 * z(vec) + 0.40 * z(graph_score) + 0.10 * exact
```

Guardrails (hard post-fusion rules, applied after the weighted sum):

1. **Direct-link floor:** direct=1 and vec in the top half of the
   fused 30 can never fall below rank 6 (swap with the lowest-ranked
   non-qualifying doc in the top-6, so one qualifier never displaces
   another).
2. **Generic-path cap:** a candidate whose best path uses ONLY
   `RELATED_TO` (or only `LOCATED_IN` through a hub) can never
   enter the top-3.
3. **Graph-missing fallback:** no seeds linked, or expansion empty
   → `graph_score = 0` for all, weights renormalise to vec 0.833 /
   exact 0.167, query flagged `graph_missing=true`. Vector-only,
   never a failure.
4. **Per-document cap:** at most 2 chunks from the same
   `document_id` in the final 6.

Ties break toward the stronger graph score, then the stronger
vector score, then doc_id.

## 4. Seeds — gazetteer first (Phase 1)

Deterministic exact match on the demo vocabulary: asset codes
(`O21`), component codes (`motor-M2`, `door-sensor-DS1`, …),
building codes, document ids, fault codes (`F-204` → O21 +
motor-M2, grounded in `FAULTS`), and the unambiguous component
names "door sensor" → DS1, "brake" → BK1. Bare "motor" /
"controller" are ambiguous (M2/M3/M0, C1/C2) and deliberately
NOT mapped. Codes are normalised to the ontology key spelling.
No LLM, no embedding similarity in the seed linker. Seed-linking
miss rate is the single best predictor of graph-signal failure —
it is the Phase 3 gating trigger "gazetteer_miss" (§6).

## 5. Traversal contract (Phases 1–2)

- Depth: **1 hop** over the allowlist `DOCUMENTED_BY /
  HAS_COMPONENT / PART_OF / LOCATED_IN / RELATED_TO`.
- Fan-out: **≤10 neighbours per (seed, relation type)**,
  **≤50 nodes total** per query; both caps deterministic
  (neighbours sorted by key) and truncation is reported
  (`expansion_truncated`), never silent.
- Hub: a node with degree > 50 is a hub; `LOCATED_IN` through a
  hub weighs 0.15, not 0.40; hub transit is penalised per G8 (§3).
- Timeout: 150 ms Neo4j budget; any store error / timeout
  degrades to an empty expansion → guardrail 3 (vector-only).
- Live path: the existing `ontology/knowledge/graph_query.py::
  neighbors` (`expand_via_graph_query`) — no new Cypher.
- PPR runs once per query over the candidate subgraph (≤~200
  nodes by design); seed specificity counts come from the
  grounded linking table in the fake path and MUST be replaced
  with real corpus counts at live wiring time (§9.3).

## 6. Expansion — gated multi-query + decomposition (Phase 3)

Off by default. `ExpandedRetriever` wraps `HybridRetriever`;
when nothing triggers, the base ranking is returned untouched.

**Gating** (deterministic, on the PRE-fusion candidate rows).
Expand iff top-1 vector score < `LOW_TOP1_VEC = 0.35`, OR
top-1/top-2 margin < `SMALL_MARGIN_VEC = 0.15`, OR gazetteer
miss (no exact code match). Thresholds are fake-BoW cosine
values — re-tune live (§9.4). Limit: gating sees the vector arm
only; A01/A04 (confident vector top-1, weak post-fusion rank)
do not expand.

**Multi-query:** ≤3 paraphrases via a pluggable generator
callable. Default live implementation calls the chat model
(`runtime/llm_factory.build_chat_model`, lazy build) with a
constrained few-shot prompt ("paraphrase only; do not add facts
or asset names not in the question"). Verified only with the
deterministic stub generators in tests/eval (no live LLM in
this VM).

**Decomposition:** ONLY on detected multi-part questions.
Detection is a deterministic heuristic: ≥2 distinct asset codes,
≥2 distinct fault codes, a comparison word, or ≥2 wh-words AND
≥2 codes. Documented limits: misses M06 (second part names no
code); over-triggers on multi-hop questions naming asset +
component + fault (M01–M04); "List all …" aggregates are not
decomposed; the stub splitter cannot split "Both X and Y" /
"Compare X with Y" wordings (live LLM generator is the fix).
Sub-questions go through the same pluggable generator interface
and the same merge + fusion. Never decompose single-hop lookups.

**Merge:** per candidate, vec/graph = MEAN across original +
variants, exact = MAX, §4.1 features from the highest-graph
variant row; provenance = variants whose fused top-6 contained
it. Mean (not max) is deliberate — prototyped: max-merge
regressed blurred nDCG@6 0.958 → 0.957, mean-merge held it.
Merged set reranked by the same fusion (§3). Per-variant
retrieval overlap (top-6 Jaccard vs base, candidate Jaccard) is
logged per query.

## 6a. QueryPlan — link + intent + the single gate (Phase 4)

`runtime/retrieval_query_plan.py::build_query_plan` wraps the
Phase 3 machinery in one per-question object. Linking
(gazetteer, §4, now with Worker A's normalisation: diacritics
and separator variants at confidence 0.8) and intent scoring
(`runtime/retrieval_intent.py::score_intents`, six intents,
`final = 0.50·z(vec) + 0.30·graph + 0.20·exact`, deterministic,
no LLM) run ALWAYS; the Phase 3 gate then decides whether the
expensive parts run. A gated-off plan carries the scored intent
+ linked nodes as-is, no variants, `fallback_reason="confident
— no expansion"`. **Difference from Phase 3:** the gate is
single — a confident multi-part question no longer decomposes
(C02/C03/C05, M07 pass through untouched, intent intact),
where Phase 3 decomposed regardless of confidence. When the
gate opens, decomposition takes precedence over paraphrasing,
as in Phase 3. No retriever → gating falls back to
gazetteer-miss-only. No seeds → `fallback_reason="no confident
link"`, no rewrite variants ever.

Rewrite guardrails (`check_rewrite_variants`, pure checks):
(a) preservation floor — variant links to all original seeds,
keeps every fault code (`FAULT_CODE_RE`) and every
number-with-unit (`NUMBER_RE`, both imported, not copied);
(b) drift cap — variant seeds ⊆ original seeds ∪ 1-hop
neighbourhood, and no new fault code / number-with-unit;
(c) no seeds → all variants dropped; (d) cap ≤3. HyDE is
explicitly excluded. The seam is
`HybridRetriever.retrieve(question, query_plan=None)`: with a
plan it uses the plan's seeds and its sub-questions / variants
(merged via the Phase 3 merge path); without a plan it is
byte-identical to Phases 1–2. Golden-set intent labels are
deliberate per-case judgements of the question's ASK (never
copied from `query_type`; unanswerable cases are labelled by
their ask, never `unanswerable`).

## 7. Context assembly (Phase 1, report §5)

`runtime/retrieval_context.py::assemble_context` — 3,000-token
budget with a ~200-token / ≤20-triple graph reserve; best-first
fill by `final` score (never by source); dedup by chunk_id then
near-dup text hash; ≤2 chunks per document; truncate the
lowest-ranked first, never mid-sentence on the top-1 chunk
(chunks are dropped whole); triples verbalised one short sentence
per fact, template per relation type, source doc id in brackets,
NEVER raw JSON; two blocks (`## Relevant documents`,
`## Related facts from the asset graph`). Token counting is
`chars / 4` (`estimate_tokens`) — an approximation, no tokenizer
dependency; re-measure with the real tokenizer live (§9.7).

## 8. Evaluation — what was measured, and the honest verdicts

Golden set `scripts/golden_set_fault_analysis.json` (version
`phase0-v1`, 36 cases: lookup 12, multi-hop 8, comparison 6,
aggregate 4, unanswerable 6; 4 sibling-confusion cases), grounded
in `seeds/demo_data.py`. Harness `runtime/retrieval_harness.py`
(MLflow experiment `agentic-runtime-retrieval`, never raises).
Metrics: Recall@6 / Recall@20, nDCG@6, MRR, Precision@6 (Phase
3). Unanswerable cases are excluded from the means. All numbers
below are FAKE-setup numbers (BoW fake / code-blurred fake,
which drops every digit-bearing token — a harsher blur than any
real embedding model). Two fake setups exist because the BoW
fake saturates and the graph arm cannot beat it.

| Phase | Setup | Baseline → hybrid (nDCG@6) | Headline |
|---|---|---|---|
| 1 | blurred | 0.906 → 0.942; sibling 0.908 → 1.000 | Graph repairs code-blindness; M08 still wrong |
| 2 | blurred | 0.942 → 0.958; multi-hop 0.954 → 1.000 | Hub penalty fixes M08; PPR ablation delta +0.003 |
| 3 | blurred | 0.958 → 0.958 (expansion off → on) | **No measurable gain; expansion value unproven on this corpus** |
| 3 | BoW | 0.958 → 0.965; gated 0.917 → 0.936 | Only movement: aggregate 0.838 → 0.890 |
| 4 | blurred | 0.958 → 0.958 (plan off → on) | **No measurable gain; link recall 0.762, intent accuracy 0.500, faithfulness 100%** |
| 4 | BoW | 0.958 → 0.962; gated 0.917 → 0.928 | Only movement: aggregate 0.838 → 0.870; below Phase 3's 0.965 because seedless A02 no longer expands (linking-failure rule) |

Phase 3 exit numbers (off → on): Precision@6 0.217 → 0.217,
Recall@6 1.000 → 1.000, nDCG@6 0.958 → 0.958 on both setups;
gated subset blurred 0.931 → 0.931, BoW 0.917 → 0.936. Trigger
rate 13/36 (BoW) / 11/36 (blurred) queries — see the note below
on the blurred count; mean top-6 Jaccard vs base 0.88 — variants
DO move rankings, they just do not improve them on 7 docs.
Recall@20 is 1.000 by construction on this corpus (the pool
holds all 7 docs) — never quote it as a gain.

**Phase 4 exit numbers (plan off → on):** Precision@6 0.217 →
0.217, Recall@6 1.000 → 1.000, nDCG@6 blurred 0.958 → 0.958
(gated 0.930 → 0.930), BoW 0.958 → 0.962 (gated 0.917 →
0.928). Entity-link accuracy (entity nodes only: gold nodes
minus `doc-*`): mean recall **0.762**, exact-set rate 0.611
(ceiling below 1.0 — gold nodes include answer nodes the
question never mentions). Intent accuracy vs the deliberate
§4 labels: **18/36 = 0.500** (lookup 0.500, multi-hop 0.500,
comparison 0.667, aggregate 0.500, unanswerable 0.333 — the
unanswerable confusion: only U05 is scored `unanswerable`,
and its gold label is diagnosis; intent ≠ answerability).
Rewrite faithfulness: generated preservation rate 0.857 (BoW)
 / 0.800 (blurred) — the stub's code-stripping variant being
caught — and the kept-variant re-check is 100% on both
setups. Gate opened on 14/36 (BoW) / 12/36 (blurred)
questions. **Note on the Phase 3 trigger counts:** the
blurred count is now 11/36, not 12/36 — Worker A's gazetteer
normalisation lets U04 ("building B2") link, so it is no
longer a gazetteer miss (see `docs/hybrid-retrieval-phase4.md`
§7 for the verified cause); every Phase 3 headline number
still reproduces.

**Grid search (w_graph, vec:exact 5:1, blurred):** 0.30 → 0.957,
0.40 (priors, kept) → 0.958, 0.50 → 0.978. The priors are kept
per the "keep priors if flat" rule — 0.50's whole advantage is 3
ranking swaps on 30 cases. **First to re-check live: w_graph =
0.50.**

**LTR:** every evaluated query logs every candidate's full §4.1
feature row (`ltr_feature_rows.json` in MLflow). Yield: **252
labelled pairs (210 answerable) vs ~3,000 needed for LambdaMART
— LTR training is PENDING DATA.** No training code exists.

## 9. What was deliberately NOT built, and why

- **No cross-encoder reranker** (report §4.6 `bge-reranker-v2-m3`):
  locked owner decision — the "reranker" is the deterministic
  fusion formula. The graph-prefix A/B (§4.4e) is inapplicable
  without a cross-encoder to feed the prefix into, so it is
  skipped, not deferred; run it only if a cross-encoder is ever
  adopted, with the >2-point nDCG@6 bar from §4.7.
- **No BM25 / sparse leg** (Phase 1 without BM25, locked):
  `bm25_rank` is logged as `null`, never invented. Adding it is a
  Phase 1-scope build (Qdrant sparse vectors / payload index),
  not a tuning change.
- **No HyDE, no step-back prompting:** HyDE's hallucinated
  hypothesis is the highest-drift expansion on a 3B model
  (report §2); step-back loses the specificity exact-code
  questions need. Both deferred, gated-only if ever tried.
- **No LLM-as-reranker, no GNN reranker, no Text2Cypher:**
  LLM reranking spends the scarcest resource (3B calls) on a task
  the fusion does deterministically; GNN needs MS MARCO-scale
  training; Text2Cypher lets the 3B write Cypher — the least
  reliable link and a read-safety risk. Fixed Cypher + gazetteer
  first.
- **No LTR training:** 252 pairs is 8% of the threshold; training
  on this golden set would overfit its quirks. Logging only.
- **No live QueryPlan wiring, and no routing consumes intent:**
  the plan is built and evaluated, but `search_documents`, the
  `retrieval_context` hook, the route node and skill selection
  are untouched. Intent accuracy is 0.500 on the fake setup
  (§8) — nothing downstream may treat the scored intent as
  ground truth until the live A/B in §11 passes. The opt-in
  wiring is §12 (`execution.retrieval.query_plan`).
- **No intent-weight tuning on this golden set:** `W_VEC /
  W_GRAPH / W_EXACT` (0.50 / 0.30 / 0.20) are documented
  priors, not tuned; re-tuning them to lift 0.500 on 36 cases
  would be overfitting the fake BoW overlap.
- **No 2-hop traversal as default:** 2 hops is where explosion
  lives; 1 hop + PPR covers the multi-hop cases on this corpus
  (multi-hop nDCG@6 1.000 since Phase 2).
- **No per-variant parallel retrieval in the fake path:** the
  calls are independent and could run concurrently live; the
  fake corpus is too small for it to matter. A live optimisation.

## 10. Locked decisions (all of them, in one place)

Retrieve-then-traverse + ontology-aware reranking (NOT GraphRAG /
LightRAG / HippoRAG wholesale — the curated ontology already
exists). No reranker model; fusion priors `final = 0.50·z(vec) +
0.40·z(graph_score) + 0.10·exact` (w_graph = 0.50 is the
first-to-recheck-live runner-up, priors stay 0.40). Full
graph_score `= 0.45·z(PPR) + 0.30·z(path) + 0.25·coverage` with
Phase 1 G1/G4/G5 adjustments + G8 hub penalty inside the path
term; PPR via log → min-max (NOT z-scored), `restart_prob =
0.5`. 4 hard guardrails (§3). RELATED_TO at 0.25 with
guardrail-2 cap. Traversal 1 hop, fan-out ≤10/type, total ≤50.
Phase 1 WITHOUT BM25. Candidates = Qdrant top-40 dense ∪ Neo4j
1-hop (≤50), RRF k=60 → top-30, final top-6. Context budget
3,000 total, ~200 tokens / ≤20 triples. DOCUMENTED_BY direction
Asset → Document. Pilot mission `fault_analysis`. Multi-query
gated only (low-confidence / no-exact-match, ≤3 variants), HyDE
+ decomposition deferred except the gated decomposition in
Phase 3. Embedding model is the configured EMBEDDING_MODEL with
per-part overrides (already built — do not modify
`app/config.py`, `runtime/llm_factory.py`, or the embedding paths
in `infra/qdrant_client.py`).

## 11. Live-verification checklist (user's machine)

1. **Pull** and confirm the working tree matches the pushed
   Phase 0–3 state (no local `.git` in this VM — push from the
   parent agent; do not push from here).
2. **Seed / backfill:** `make seed` (re-seeds Qdrant docs with
   payload links) or `python -m scripts.backfill_qdrant_links`
   for an existing collection; confirm with a Qdrant scroll that
   all 7 points carry non-empty `node_ids`.
3. **Qdrant + Neo4j live checks:** run the real vector retriever
   through the harness, then the hybrid with
   `use_live_graph=True` against the live Neo4j graph; replace
   seed-specificity counts (currently from the grounded linking
   table) with real corpus counts; confirm live adjacency node
   degrees (a Building with degree >50 is a real hub — the demo's
   buildings have degree 4–8, so the hub penalty was validated on
   small degrees only); confirm PPR latency on the real candidate
   subgraph (≤~200 nodes by design — cap the PPR node set first
   if the live expansion is larger); confirm the 150 ms Neo4j
   timeout and the vector-only fallback on a forced graph error.
4. **Run the golden set live through each phase's eval script:**
   `make eval-retrieval-phase1`, `make eval-retrieval-phase2`,
   `make eval-retrieval-phase3`, `make eval-retrieval-phase4` (pass
   `TRACKING_URI=<mlflow uri>` to log to the live MLflow server;
   confirm the `agentic-runtime-retrieval` experiment shows the
   runs, including the Phase 3 `ltr_feature_rows.json` artifact).
   Replace every fake number in §8 with live ones. Expect the
   live sibling-subset gain to be SMALLER than the blurred fake's
   (real embeddings retain some code signal).
5. **Re-run the w_graph grid live with 0.50 first** (the
   first-to-recheck-live runner-up in Phases 1–2; fake grid:
   0.30 → 0.957, 0.40 → 0.958, 0.50 → 0.978 — three swaps, not a
   live prediction).
6. **Live expansion A/B:** swap the stub generators for the live
   ones (`ExpandedRetriever(base)` with no generator arguments),
   re-run the Phase 3 A/B with the same exit criterion (no
   Precision@6 loss, no Recall@6/nDCG@6 regression, gated subset
   reported separately). Re-tune the gating thresholds live
   (0.35 / 0.15 are fake-BoW cosine values): log the top-1/margin
   distribution on live traffic, target ~10–30% of queries
   expanding. Spot-check that live paraphrases never add asset
   names/codes, and that the live decomposition splits the
   wordings the stub cannot (C02 "Both X and Y", C04/C05
   "Compare X with Y").
7. **Context budget:** re-measure with the real tokenizer
   (`estimate_tokens` is chars/4, an approximation).
8. **Grow the corpus before trusting Recall@20** (saturated by
   construction at 7 docs) — and before LTR (252 pairs vs ~3,000
   needed). Do NOT train a ranker on this golden set.
9. **M08 live analogue:** a query naming an asset whose gold
   document is linked to a *different* asset's component family
   (the O22/C2 → controller-doc shape); confirm the hub penalty
   demotes the building-mediated distractor rather than the gold
   doc.
10. **Live QueryPlan A/B:** swap the stub generators for the
    live ones (`build_query_plan` with no generator arguments)
    and re-run the Phase 4 A/B with the same exit criterion
    (kept-variant re-check 100%, no nDCG@6 regression, gated
    subset reported separately). Re-tune the gating thresholds
    live (0.35 / 0.15 are fake-BoW cosine values): log the
    top-1/margin distribution on live traffic, target ~10–30%
    of queries expanding. Spot-check that live paraphrases never
    add asset names/codes/thresholds (the guardrail drop rate
    on live paraphrases is itself a finding), and that the live
    decomposition splits the wordings the stub cannot (C02
    "Both X and Y", C04/C05 "Compare X with Y").
11. **Grow the QU golden set** before trusting any QueryPlan
    claim or tuning intent weights / thresholds — 36 cases
    with 5 intent labels is enough to measure the plumbing,
    not to tune on.

## 12. Opt-in wiring follow-up (from Phase 1 §6, still NOT done)

Add an opt-in mission-config flag, e.g.
`execution.retrieval.hybrid: true` on `fault_analysis` only,
read by the `retrieval_context` hook (SessionStart, the existing
pre-retrieval hook): when set, the hook calls
`HybridRetriever(use_live_graph=True)` — or `ExpandedRetriever`
over it, once the live A/B in §11.6 passes — with the real
LiteLLM embeddings + Qdrant client + the live Neo4j store
(`expand_via_graph_query`), assembles context with
`runtime/retrieval_context.py::assemble_context`, and logs the
trace (seeds, expansion, per-candidate G1–G9, guardrail flags,
expansion kind/variants/overlap) to MLflow. When unset, the hook
behaves exactly as today. The hook must pass real corpus
doc-counts for seed specificity (§11.3). No change to
`search_documents` / the tool surface is needed; the hybrid
module already accepts injected embeddings/client/store, so the
wiring is a call-site change, not a redesign. The Phase 4
variant of the same flag is `execution.retrieval.query_plan:
true`: build the QueryPlan first (live generators), then call
`HybridRetriever(use_live_graph=True).retrieve(question,
query_plan=plan)` and log the plan (intent, margin, seeds,
gating, fallback reason, kept/dropped variants with reasons)
to MLflow alongside the trace.

Why not wired yet: there is no live Qdrant/Neo4j in this VM to
verify the live path against, and wiring without a live check
would ship an untested retrieval change into the fault mission.
The fake-verified module + the documented call site is the right
stopping point.
