# Hybrid retrieval — Phase 1: link + fused rerank (no reranker model)

Report: `~/workspace/research_notes/hybrid-vector-graph-retrieval-20261114-0430/report.md`
(§4.1, §4.2, §4.7 fusion + guardrails, §5 context assembly, §8 Phase 1).
Phase 0: `docs/hybrid-retrieval-phase0.md`. Pilot mission: `fault_analysis`.

## What was built

| File | What it is |
|---|---|
| `runtime/retrieval_linking.py` | Grounded `DOCUMENT_LINKS` table for the 7 demo docs + `compute_document_links` / `with_document_links` / `linking_coverage`. |
| `runtime/retrieval_graph.py` | Gazetteer seed linking, demo adjacency, 1-hop expansion (caps + hub), G1/G2/G4/G5 + `graph_score`. |
| `runtime/retrieval_fusion.py` | Exact fusion formula, RRF pre-fusion, guardrails 1–4. |
| `runtime/retrieval_blurred.py` | Code-blurred fake embeddings (the exit-criterion setup). |
| `runtime/retrieval_hybrid.py` | `HybridRetriever` (dense + 1-hop graph + fusion) + fake builders. |
| `runtime/retrieval_context.py` | §5 context assembly (budget, dedup, verbalised triples). |
| `infra/qdrant_client.py` | `set_document_payloads` + `backfill_document_links` (payload-only, no re-embedding). |
| `scripts/backfill_qdrant_links.py` | `python -m scripts.backfill_qdrant_links [--fake]`. |
| `scripts/eval_retrieval_phase1.py` | Baseline vs hybrid on both fake setups + w_graph grid search. `make eval-retrieval-phase1`. |
| `tests/test_retrieval_phase1.py` | Main-feature tests only (24 tests). |

Changed, not new: `seeds/demo_data.py` (payload link fields on all
7 `DOCUMENTS`, `validate()` enforces >=80% linking coverage),
`ingest/schemas.py` (`DocumentRow` gains optional link fields),
`ingest/loaders.py` (`load_documents` fills missing links),
`runtime/retrieval_baseline.py` (`FakeQdrantClient` gains
`retrieve` / `scroll` / `set_payload` / `upsert` + a payload store,
so the backfill is testable). Untouched on purpose:
`app/config.py`, `runtime/llm_factory.py`, the embedding paths in
`infra/qdrant_client.py`, `search_documents`, the `retrieval_context`
hook, `services/user_service/`, and every other mission.

## 1. Payload linking (report §3.1)

Every upserted doc carries `node_ids[]` (+ `asset_id` /
`component_id` / `building_id` where known). The mechanism already
existed — `upsert_documents` passes extra payload keys through — so
the seed path just attaches them (`seed_qdrant_demo` passes
`DOCUMENTS` straight through) and the ingest path fills them when a
row omits them (`with_document_links`; an explicit row value wins).
`seeds/demo_data.py::validate()` now fails the seed if coverage
drops below 0.80.

Achieved coverage: **7/7 = 100%** (target >=80%).

| doc | node_ids | grounded in |
|---|---|---|
| `doc-door-sensor` | O21, door-sensor-DS1, building-B1, doc-door-sensor | DOCUMENTED_BY from O21 + F-101 fault |
| `doc-motor-overheat` | O21, motor-M2, building-B1, doc-motor-overheat | DOCUMENTED_BY from O21 + F-204 fault |
| `doc-brake-inspection` | O21, brake-BK1, building-B1, doc-brake-inspection | DOCUMENTED_BY from O21 + F-301 fault |
| `doc-controller-errors` | O21, controller-C1, building-B1 | F-410/F-103 faults; no Document node |
| `doc-leveling` | O21, building-B1 | F-102 fault; no leveling component node |
| `doc-rope-inspection` | O21, building-B1 | One of O21's 6 seeded docs; no rope component node |
| `doc-o23-overview` | O23, building-B2 | Title + text name O23 in B2; no Document node |

Deliberate asymmetry (golden case M05): `doc-motor-overheat`
names F-204, O21's fault, so it links to O21 / motor-M2 **only** —
not to O22 / motor-M3, even though the remedy text also fits O22's
F-206. Linking reflects what the document is about / linked to in
the ontology, never a forced fit to a question. A graph-only
retriever following O22's links misses this doc; the fusion keeps
the vector arm so it still surfaces (verified: M05 stays in the
hybrid top-6 on both fake setups).

### Backfill

`upsert_documents` SKIPS existing points, so docs indexed before
Phase 1 never pick up `node_ids`. `backfill_document_links`
scrolls every point, computes links from `retrieval_linking`, and
calls `set_payload` — payload-only, no re-embedding. Idempotent
(second run: 0 updated, 7 unchanged on the fake). Live:
`python -m scripts.backfill_qdrant_links` (Qdrant up).

## 2. Neo4j 1-hop expansion (report §3.2, §4.1, §4.2)

Seeds: gazetteer exact match first — asset codes (`O21`),
component codes (`motor-M2`, `door-sensor-DS1`, …), building codes,
document ids, fault codes (`F-204` → O21 + motor-M2, grounded in
`FAULTS` rows + fault titles), and the unambiguous component names
"door sensor" → DS1, "brake" → BK1. Bare "motor" / "controller"
are ambiguous (M2/M3/M0, C1/C2) and are deliberately NOT mapped.
Codes are normalised to the ontology key spelling (`o22` → `O22`,
`motor-m2` → `motor-M2`); a lowercase match silently matching
nothing was a real bug found during eval (M05 got zero expansion
before the fix). No LLM, no embedding similarity in Phase 1.

Expansion: 1 hop over the allowlist
`DOCUMENTED_BY / HAS_COMPONENT / PART_OF / LOCATED_IN / RELATED_TO`,
fan-out <=10 per (seed, relation type), total <=50 nodes, hub
handling (a node with degree >50 is a hub; `LOCATED_IN` through a
hub weighs 0.15, not 0.40), Neo4j timeout 150 ms with graceful
vector-only fallback. `expansion_nodes_returned` +
`expansion_truncated` are logged per query in the hybrid trace.
The fake/eval graph is an in-memory adjacency built from
`RELATIONS` + payload doc links (the payload edges are what make
the 4 non-Document docs reachable at all); the live path calls the
existing `ontology/knowledge/graph_query.py::neighbors` via
`expand_via_graph_query` — no new Cypher in Phase 1.

Per candidate: G1 (0 = linked entity IS a seed, 1 = 1-hop neighbour,
None = unreachable), G2 path_score = max over paths of
`prod(w_rel) * 0.6**hops` with §4.2 weights (DOCUMENTED_BY 1.00,
HAS_COMPONENT/PART_OF 0.90, LOCATED_IN 0.40 / 0.15 via hub,
RELATED_TO 0.25), G4 (+1 link to a seed component/asset, -1 link to
a *different* component/asset, else 0), G5 (direct 1 / 0.5 via a
1-hop neighbour / 0 + distinct seeds linked). G6/G7/G8
(coverage/PPR/hub score) are Phase 2; the hub *penalty* is already
here as the 0.15 weight.

## 3. Fusion — the exact formula as built (report §4.7)

```
z(x)   = per-query z-score of x over the fused candidates, clipped to [-2, 2]
         (std 0 -> z = 0 for all)
final  = 0.50 * z(vec) + 0.40 * z(graph_score) + 0.10 * exact
exact  = 1 if a code the question names appears verbatim in the
         candidate's title/content, else 0
```

`graph_score` enters via its per-query z-score, exactly as §4.7's
`graph_score` (itself a z-combination) enters the original
`0.50*z(ce) + 0.30*graph_score + 0.15*z(vec) + 0.05*exact` — with
the cross-encoder arm removed (no model in Phase 1) and its weight
moved to the graph arm (0.30 -> 0.40, locked decision).

`graph_score` itself (from G1/G2/G4/G5):

```
graph_score = clip(path_score            # G2, incl. G3 relation weights
                     and 0.6**hops decay (G1's distance, inside G2)
                   + 0.10 * G4           # type match +1 / mismatch -1
                   + 0.10 * (G5_direct - 0.5)
                   + hop_adj             # G1: 0 -> +0.05, 1 -> 0, None -> -0.05
                   + 0.02 * min(max(n_seeds_linked - 1, 0), 2),
                   -0.3, 1.2)
```

The floor is -0.3, not 0: a G4 mismatch (linked to a *different*
asset/component than the seeds) must score *below* an unlinked doc,
not tie with it — otherwise the wrong-asset doc (the O23 overview
for an O21 question) keeps its vector rank. Found during eval:
with a 0 floor, A01/A04 lost the motor doc out of the top-6.

Pipeline: candidates = Qdrant top-40 dense UNION Neo4j 1-hop
linked docs; RRF (k=60) over vector rank + graph rank -> top-30;
fusion -> guardrails -> top-6. Ties break toward the stronger
graph score, then the stronger vector score, then doc_id.

Guardrails (hard post-fusion rules):
1. **Direct-link floor** — direct=1 and vec in the top half of the
   fused 30 can never fall below rank 6. Implemented as a swap
   with the lowest-ranked *non-qualifying* doc in the top-6, so
   promoting one qualifier never displaces another qualifier
   (a plain promotion did exactly that on A04, pushing the motor
   doc out).
2. **Generic-path cap** — a candidate whose best path uses ONLY
   `RELATED_TO` (or only `LOCATED_IN` through a hub) can never
   enter the top-3.
3. **Graph-missing fallback** — no seeds linked, or expansion
   empty → `graph_score = 0` for all, weights renormalise over
   the surviving arms (vec 0.50/0.60 = 0.833, exact 0.10/0.60 =
   0.167), query flagged `graph_missing=true`. Vector-only, never
   a failure.
4. **Per-document cap** — at most 2 chunks per `document_id` in
   the final 6.

### Grid search (locked: w_graph in {0.30, 0.40, 0.50}, vec:exact 5:1)

On the code-blurred setup (golden set phase0-v1, n=30 answerable):

| w_graph | w_vec | w_exact | nDCG@6 | Recall@6 | sibling nDCG@6 |
|---|---|---|---|---|---|
| 0.30 | 0.583 | 0.117 | 0.943 | 1.000 | 1.000 |
| **0.40 (priors, kept)** | **0.500** | **0.100** | **0.943** | **1.000** | **1.000** |
| 0.50 | 0.417 | 0.083 | 0.958 | 1.000 | 1.000 |

The grid is near-flat (0.30 and 0.40 identical; 0.50 is +0.015
nDCG, about one ranking swap on 30 cases), so the locked priors
(0.40) are kept per the "keep priors if flat" rule. Runner-up:
w_graph=0.50 (nDCG@6 0.958) — the live re-tune on real embeddings
should re-check it first.

## 4. Context assembly (report §5)

`runtime/retrieval_context.py::assemble_context` — best-first fill
in a 3,000-token budget with a ~200-token / <=20-triple graph
reserve; dedup by chunk_id then near-dup text hash; <=2 chunks per
document; truncate the lowest-ranked first, never mid-sentence on
the top-1 chunk (chunks are dropped whole, top-1 always whole);
triples verbalised one short sentence per fact, template per
relation type, source doc id in brackets, NEVER raw JSON; two
blocks (`## Relevant documents`, `## Related facts from the asset
graph`). Token counting is `chars / 4` (`estimate_tokens`) — an
approximation, documented as such; no tokenizer dependency was
added. Live serving must re-measure with the real tokenizer before
trusting the budget edge.

## 5. Numbers — both fake setups (golden set phase0-v1, n=36/30)

Run: `python -m scripts.eval_retrieval_phase1` (exit code 0 = exit
criterion passed). All numbers are fake-setup numbers.

| Setup | Retriever | Recall@6 | nDCG@6 | MRR | sibling Recall@6 | sibling nDCG@6 |
|---|---|---|---|---|---|---|
| BoW (Phase 0 fake) | baseline | 1.000 | 0.942 | 0.922 | 1.000 | 1.000 |
| BoW (Phase 0 fake) | **hybrid** | 1.000 | **0.947** | 0.939 | 1.000 | 1.000 |
| Code-blurred | baseline | 1.000 | 0.906 | 0.872 | 1.000 | 0.908 |
| Code-blurred | **hybrid** | 1.000 | **0.942** | 0.922 | 1.000 | **1.000** |

Per-type, code-blurred (baseline -> hybrid nDCG@6): lookup 1.000 ->
1.000, multi-hop 0.862 -> 0.954, comparison 0.831 -> 0.893,
aggregate 0.822 -> 0.822. The gain is concentrated exactly where
the graph should help: multi-hop and sibling comparison cases.
The remaining blurred misses are the documented traps: M05 (motor
doc is O21-linked only, correctly demoted by the graph but kept in
the top-6 by the vector arm), M06, C06, A01–A03 (multi-gold
aggregate cases where one gold doc sits at rank 2–3).

**Exit criterion: PASS.** On the code-blurred setup, nDCG@6
improves 0.906 -> 0.942 (+0.036) and sibling nDCG@6 improves
0.908 -> 1.000 (+0.092). Recall@6 is 1.000 for both retrievers on
both setups — it cannot improve past the ceiling on a 7-doc corpus
with a top-6 context (every doc is in the top 6 unless actively
demoted), so nDCG@6 + the sibling subset are the discriminating
metrics, as Phase 0 already warned. On the BoW setup the hybrid
also does not regress (0.942 -> 0.947), which the saturated
baseline could not have shown as a win.

Caveat (read before quoting these): the blurred fake drops *every*
digit-bearing token, which is a harsher blur than any real embedding
model (a real model retains some code signal). The numbers prove
the graph arm *can* repair code-blindness on this corpus; they do
not predict the live gain. Live re-measurement is required (below).

## 6. Integration decision (documented, NOT wired live)

The hybrid retriever is a **new module** (`runtime/retrieval_hybrid.py`)
evaluated through the Phase 0 harness. `search_documents` and the
`retrieval_context` hook are untouched, so no mission's behaviour
changes in this phase.

Follow-up wiring step (after Phase 1 exit passes — it has — and
after live verification below): add an opt-in mission-config flag,
e.g. `execution.retrieval.hybrid: true` on `fault_analysis` only,
read by the `retrieval_context` hook (SessionStart, the existing
pre-retrieval hook): when set, the hook calls
`HybridRetriever(use_live_graph=True)` with the real LiteLLM
embeddings + Qdrant client + the live Neo4j store
(`expand_via_graph_query`), assembles context with
`runtime/retrieval_context.py::assemble_context`, and logs the
trace (seeds, expansion, per-candidate G1–G5, guardrail flags) to
MLflow. When unset, the hook behaves exactly as today. No change
to `search_documents` / the tool surface is needed for this step;
the hybrid module already accepts injected embeddings/client/store,
so the wiring is a call-site change, not a redesign.

Why not wire it now: there is no live Qdrant/Neo4j in this VM to
verify the live path against, and wiring without a live check
would ship an untested retrieval change into the fault mission.
The fake-verified module + the documented call site is the right
stopping point for Phase 1.

## 7. Live verification (user's machine)

1. `make seed` (re-seeds Qdrant docs — now with payload links) or
   `python -m scripts.backfill_qdrant_links` for an existing
   collection; confirm with a Qdrant scroll that all 7 points carry
   non-empty `node_ids`.
2. Run the real vector retriever through the same harness, then
   the hybrid with `use_live_graph=True` against the live Neo4j
   graph; replace the fake numbers above with live ones. Expect
   the live sibling-subset gain to be *smaller* than the blurred
   fake's (real embeddings retain some code signal) — re-run the
   w_graph grid search live and re-check 0.50 vs the 0.40 priors.
3. Point the eval at the live MLflow server
   (`python -m scripts.eval_retrieval_phase1 --tracking-uri <uri>`)
   and confirm the `agentic-runtime-retrieval` experiment shows the
   runs.
4. Grow the corpus before trusting Recall@20: with 7 docs it is
   saturated by construction.
5. Re-measure the context budget with the real tokenizer
   (`estimate_tokens` is chars/4, an approximation).
