# Hybrid retrieval — Phase 2: PPR + coverage + hub (no reranker model)

> **REMOVED IN PART 2026-11-14 (user option B — delete the
> fakes entirely).** This phase's eval script
> (`scripts/eval_retrieval_phase2.py`,
> `make eval-retrieval-phase2`) is DELETED. The PPR /
> coverage / hub code survives. The numbers in §5 are
> HISTORICAL fake-setup measurements and are NOT
> reproducible; the only retrieval eval left is
> `make eval-retrieval-live`.

Report: `~/workspace/research_notes/hybrid-vector-graph-retrieval-20261114-0430/report.md`
(§4.1 G6/G7/G8, §4.4 PPR + damping warning, §4.7 fusion + guardrails,
§8 Phase 2). Phase 1: `docs/hybrid-retrieval-phase1.md`. Phase 0:
`docs/hybrid-retrieval-phase0.md`. Pilot mission: `fault_analysis`.

## What was built

| File | What it is |
|---|---|
| `runtime/retrieval_ppr.py` (new) | Pure-Python Personalized PageRank (G7): power iteration, `restart_prob = 0.5`, seed weights `confidence × specificity`. No new dependency. |
| `runtime/retrieval_graph.py` | G6 coverage, G7 PPR passthrough, G8 hub transit in `graph_features`; `phase2_graph_scores` — the locked full formula. `graph_score` (Phase 1) unchanged. |
| `runtime/retrieval_hybrid.py` | `HybridRetriever(phase2=True, w_ppr=0.45)` — PPR once per query, Phase 2 scoring, `w_ppr=0` ablation flag; `phase2=False` reproduces Phase 1. |
| `scripts/eval_retrieval_phase2.py` (new) | ~~Baseline vs Phase 1 hybrid vs Phase 2 hybrid on both fake setups + w_graph grid search + PPR ablation. `make eval-retrieval-phase2`.~~ **DELETED 2026-11-14** (fake removal). Use `make eval-retrieval-live`. |
| `scripts/eval_retrieval_phase1.py` | ~~Pins `phase2=False` everywhere, so it keeps measuring Phase 1 exactly as documented.~~ **DELETED 2026-11-14** with the other phase evals. |
| `tests/test_retrieval_phase2.py` (new) | Main-feature tests only (13 tests). |
| `Makefile` | ~~New `eval-retrieval-phase2` target.~~ Target removed 2026-11-14 (fake removal). |

Changed nothing else. Untouched on purpose: `app/config.py`,
`runtime/llm_factory.py`, the embedding paths in
`infra/qdrant_client.py`, `search_documents`, the
`retrieval_context` hook, `services/user_service/`, and every
other mission. No new dependency was added (see §1).

## 1. PPR (G7) — pure Python, restart convention pinned

Checked first in `.venv`: `igraph` NOT installed, `networkx` NOT
installed, `scipy` IS installed. Decision: **pure-Python power
iteration** (~30 lines in `runtime/retrieval_ppr.py`), no new
dependency. Rationale: the candidate subgraph is ≤~200 nodes by
design (report §4.4c), power iteration over it is milliseconds
(mean latency 0.0013 s/query on the harness, vs 0.0008 s for
Phase 1 — the PPR cost is ~0.5 ms/query on the fake corpus);
pulling in `scipy.sparse` for that would add a heavy dependency
to a hot path, and `igraph` would be a brand-new install. No GDS
dependency either — the GDS PageRank docs page returned 403
during research (report §4.4).

**Damping-convention trap (report §4.4, mandatory).** The
parameter is named `restart_prob = 0.5` everywhere — restart to
the seeds with probability 0.5, follow an edge with probability
0.5 — which is HippoRAG's `damping = 0.5` and GDS
`dampingFactor: 0.5` in the follow-edge convention. It is never
called bare "damping". Note 0.5 is self-dual (restart == follow),
so the convention is pinned by the *tests* varying
`restart_prob` (0.9 keeps mass at the seeds, 0.1 diffuses it to
the 3-hop node), not by the default value alone.

- Personalization: each seed weighs `confidence × specificity`,
  `spec_s = 1 / max(1, docs containing the seed)` (HippoRAG
  node-specificity). A seed in no document gets spec 1.0
  (maximally specific); O21, in 6 documents, seeds at 1/6; a
  component seed in 1 document seeds at 1.0. Counts come from
  the grounded linking table (`DOCUMENT_LINKS`); live wiring
  must replace them with real corpus counts (see §7).
- Edge weights: the §4.2 relation weights, row-normalised per
  node (weighted PPR, §4.4c). Dangling nodes teleport to the
  personalization vector. Iteration stops at L1 < 1e-9 or 200
  iterations; scores sum to 1.0.
- Toy-graph assertion (report §4.4, in the tests): two seeds,
  one 1-hop node, one 3-hop node — the 1-hop node outranks the
  3-hop node, seeds outrank both, scores sum to 1.0.
- A document's PPR is its **document node's** score. The
  alternative — max over the doc's linked entities, as the
  report's G7 row suggests — was prototyped and **rejected**:
  it lets a doc inherit a seed-adjacent entity's mass even when
  the doc itself is far, and it dropped blurred aggregate
  nDCG@6 0.838 → 0.822.

## 2. Coverage (G6) and hub (G8)

- **G6 coverage** = `|{seeds within 2 hops of d}| / |S|`,
  BFS from the document node over the undirected adjacency.
  A candidate sitting between two seeds scores higher than one
  near a single seed — the conjunction `d_min` misses.
- **G8 hub** = `log(1 + degree)` of the highest-degree node on
  the best path — but only when that node is a **transit**
  node of type Building or generic Document. A direct seed link
  and a doc that IS a 1-hop neighbour have no transit node (the
  doc is the endpoint) and are never hub-penalised; only the
  "linked entity is a 1-hop neighbour" branch has a transit
  node (that entity). A Component/Asset transit is structural,
  not a hub, and is not penalised. `graph_features` returns the
  transit key (`g8_transit_key`) in the trace so any hub
  penalty is auditable.

## 3. Fusion — the exact formula as built (report §4.7)

```
path_term   = graph_score(row)        # Phase 1 G1/G2/G4/G5
              - HUB_LAMBDA * g8_hub   # G8, HUB_LAMBDA = 1.0
ppr_norm    = log1p(PPR) then min-max to [0, 1] per query   # G7
graph_score = 0.45 * ppr_norm + 0.30 * z(path_term)
              + 0.25 * coverage                            # G6
z(x)        = per-query z-score of x, clipped to [-2, 2]
```

The outer fusion is unchanged from Phase 1:
`final = 0.50 * z(vec) + 0.40 * z(graph_score) + 0.10 * exact`,
guardrails 1–4 unchanged in shape, RRF (k=60) → top-30 →
top-6 unchanged.

Three prototype findings shaped the formula (all measured on the
golden set, blurred setup — see §5 for the numbers):

1. **PPR is log→min-max, NOT z-scored.** Report §4.7 sanctions
   this explicitly ("PPR via log then min-max"). It is also
   empirically required: on this corpus usually exactly one
   candidate has nonzero PPR, and a z-score would hand it
   z = +2.0 however small its mass — prototyped, z-scored PPR
   regressed blurred multi-hop nDCG@6 0.954 → 0.908.
2. **The hub penalty lives INSIDE the path term**, not
   subtracted from the final graph_score. Both placements fix
   M08 at λ = 1.0 (overall 0.957 vs 0.958 — a wash), but
   inside-path is strictly safer live: a hub can have degree
   200+ (log1p ≈ 5.3), and subtracting that from the final
   score would let one hub dominate the fusion; inside the
   z-scored path term the penalty saturates at z = −2.
   λ sweep on the blurred setup: 0.6–1.25 give identical
   numbers (the z-clip saturates the effect), so λ = 1.0 — the
   full log-degree value — is used, not a tuned knife-edge.
   Down-weighting Building-incident PPR edges as well (×0.25)
   was prototyped and changed nothing (0.957), so it was left
   out: one hub mechanism, in one place.
3. **Coverage stays raw [0, 1], not z-scored**, per the locked
   formula. It is a small term (max +0.25) and deliberately so:
   on M08 it rewards the hub-connected distractor (0.5 vs 0
   for the gold doc) — the hub penalty has to outvote it, and
   does (see §4).

`w_ppr=0` is the PPR ablation: it zeroes the PPR sub-weight
with **no renormalising** of the surviving sub-weights, so the
ablation measures PPR alone (path:coverage stays 0.30:0.25).

## 4. Why M08 is fixed — the one case Phase 1 got wrong

M08: *"Elevator O22's controller C2 has CAN bus errors (F-207).
Which document explains CAN bus diagnosis?"* Gold:
`doc-controller-errors`. Phase 1 hybrid (blurred) ranked it
2nd behind `doc-o23-overview` (nDCG 0.631) — the only
multi-hop miss. Per-candidate features under seeds
O22 + controller-C2 (both specificity 1.0):

| candidate | Phase 1 graph | G6 cov | G7 PPR | G8 hub (transit) | Phase 2 graph |
|---|---|---|---|---|---|
| `doc-o23-overview` | 0.54 | 0.50 | 0.0162 | 1.6094 (building-B2) | **−0.025** |
| `doc-controller-errors` (gold) | −0.20 | 0.00 | 0.0000 | 0.0 | 0.09 |
| `doc-leveling` | −0.10 | 0.00 | 0.0000 | 0.0 | 0.184 |

The distractor's whole Phase 1 advantage is one path:
O22 —PART_OF→ building-B2, and the doc links building-B2.
PPR and coverage both *reward* it (it is the only candidate
with nonzero PPR — mass O22 → B2 → doc — and the only one
with coverage > 0), because the gold doc sits in the
disconnected B1 component and no diffusion can reach it. The
hub penalty is the only signal that can fix this case: the
path term drops 0.54 − 1.61 = −1.07 (z ≈ −2, clipped), which
outvotes the PPR + coverage boost. The gold doc then wins on
the vector arm (vec 0.4234, the highest blurred vector score
in the query) and ranks 1st (final 0.9364); the distractor
drops out of the top-6 entirely. This is the G8 failure mode
the report describes — a Building making every asset in it
"1 hop" from every other — reproduced and fixed on the demo
corpus.

## 5. Numbers — both fake setups (golden set phase0-v1, n=36/30) — HISTORICAL, not reproducible

~~Run: `python -m scripts.eval_retrieval_phase2` (exit code 0 =
exit criterion passed).~~ The eval script is deleted. All numbers are fake-setup numbers.

| Setup | Retriever | Recall@6 | nDCG@6 | MRR | multi-hop nDCG@6 | sibling nDCG@6 |
|---|---|---|---|---|---|---|
| BoW | baseline | 1.000 | 0.942 | 0.922 | 0.954 | 1.000 |
| BoW | hybrid Phase 1 | 1.000 | 0.947 | 0.939 | 0.954 | 1.000 |
| BoW | **hybrid Phase 2** | 1.000 | **0.958** | 0.940 | **1.000** | 1.000 |
| Code-blurred | baseline | 1.000 | 0.906 | 0.872 | 0.862 | 0.908 |
| Code-blurred | hybrid Phase 1 | 1.000 | 0.942 | 0.922 | 0.954 | 1.000 |
| Code-blurred | **hybrid Phase 2** | 1.000 | **0.958** | 0.940 | **1.000** | 1.000 |

Per-type, code-blurred (Phase 1 → Phase 2 nDCG@6): lookup
1.000 → 1.000, multi-hop 0.954 → **1.000**, comparison
0.893 → 0.898, aggregate 0.822 → 0.838. BoW per-type is
identical to blurred for Phase 2 (lookup 1.000, multi-hop
1.000, comparison 0.898, aggregate 0.838) — the graph arm,
not the vector arm, decides these cases on this corpus.

**Exit criterion: PASS.** Blurred multi-hop nDCG@6 improves
0.954 → 1.000 (+0.046, the M08 fix), overall 0.942 → 0.958
with no regression anywhere (BoW 0.942 → 0.958 vs baseline),
and the PPR ablation shows a measurable delta (below).

### Grid search (locked: w_graph in {0.30, 0.40, 0.50}, vec:exact 5:1)

On the code-blurred setup (Phase 2):

| w_graph | w_vec | w_exact | nDCG@6 | multi-hop nDCG@6 | sibling nDCG@6 |
|---|---|---|---|---|---|
| 0.30 | 0.583 | 0.117 | 0.957 | 1.000 | 1.000 |
| **0.40 (priors, kept)** | **0.500** | **0.100** | **0.958** | **1.000** | **1.000** |
| 0.50 | 0.417 | 0.083 | 0.978 | 1.000 | 1.000 |

The priors (0.40) are kept, per the "keep priors if flat"
rule and the Phase 1 precedent — with a caveat. w_graph=0.50
is the grid-best (+0.020 nDCG), but the whole advantage comes
from 3 ranking swaps on 30 cases (C06 +0.113, A01 +0.218,
A04 +0.267 — all comparison/aggregate cases, where nDCG is
ordering-among-golds), and 0.30/0.40 are identical (0.957 vs
0.958). Moving a locked prior on three fake-setup swaps would
be overfitting the golden set. Runner-up / first to re-check
live: **w_graph=0.50** (same runner-up as Phase 1, where it
was +0.015).

### PPR ablation (code-blurred, w_ppr=0 vs full 0.45 — real flag)

| | overall nDCG@6 | multi-hop nDCG@6 |
|---|---|---|
| w_ppr=0 | 0.955 | 1.000 |
| full (0.45) | 0.958 | 1.000 |
| **delta** | **+0.003** | **+0.000** |

The delta is small but real and attributable: the flag changes
the top-6 ranking in 24 of 36 cases (it is a live code path,
not a dead parameter), and exactly two cases' nDCG@6 move —
C06 +0.031 and A04 +0.068, both cases where PPR separates the
component-linked gold doc from other same-asset docs. The
multi-hop delta is 0.000 and that is expected, not a failure:
all 8 multi-hop cases sit at 1.000 with the hub fix in place
(the hub penalty, not PPR, fixes M08), so PPR has nothing left
to move there; its contribution on this corpus is in
comparison/aggregate ordering. At w_graph=0.50 the ablation
delta is larger (+0.008 overall), same multi-hop 0.000.

Caveat (read before quoting these): the blurred fake drops
*every* digit-bearing token, which is a harsher blur than any
real embedding model. The numbers prove the graph arm *can*
repair code-blindness on this corpus; they do not predict the
live gain. Live re-measurement is required (see §7). The
n=30 answerable cases also mean a ±0.003–0.02 move is one to
three ranking swaps — report the direction, not the decimals,
as a live prediction.

## 6. Skipped experiment — the graph-prefix A/B (§4.4e)

**Explicitly NOT built.** Report §4.4e prepends a short
structured prefix (`[Asset: … | Path: …]`) to the passage
*before a cross-encoder* and A/Bs it against the numeric
fusion. There is no cross-encoder in this system — locked
decision: no reranker model, the "reranker" is the
deterministic fusion formula. With no cross-encoder there is
nothing for the prefix to feed into: prepending graph text to
a passage that is then scored by a BoW/cosine vector arm
would change the vector itself, which is a different
experiment (and a worse one — it would leak graph signal into
the vector arm and destroy the arm separation the fusion
depends on). The experiment is inapplicable as specified, so
it is skipped, not deferred: if a cross-encoder is ever
adopted (a Phase 3 decision, not taken), the A/B should be
run then, against the numeric fusion, with the prefix kept
short (≤30 tokens) and the >2-point nDCG@6 bar from §4.7.

## 7. Live verification (user's machine)

Same standing items as Phase 1, plus Phase 2 specifics:

1. `make seed` or `python -m scripts.backfill_qdrant_links`;
   confirm all 7 points carry non-empty `node_ids`.
2. Run the golden set live via `make eval-retrieval-live`
   (live baseline + live hybrid through the same harness);
   replace the fake numbers above with live ones. Expect the
   live sibling-subset gain to be *smaller* than the blurred
   fake's (real embeddings retain some code signal) — re-run
   the w_graph grid search live and re-check 0.50 vs the 0.40
   priors first. (The per-phase eval script is deleted, so
   there is no `make eval-retrieval-phase2` live run anymore.)
3. **PPR live checks:** confirm the live adjacency's node
   degrees (a Building with degree >50 is a real hub — the
   demo's buildings have degree 4–8, so the hub penalty was
   validated on small degrees only); confirm PPR latency on
   the real candidate subgraph (≤~200 nodes by design — if the
   live expansion is larger, cap the PPR node set before
   iterating); confirm seed-specificity counts come from the
   real corpus (Phase 5 wired this: `count_documents_per_node`
   over Qdrant payloads — the old fake path counted the demo
   `DOCUMENT_LINKS` table, which is gone).
4. **M08 live analogue:** run a query naming an asset whose
   gold document is linked to a *different* asset's component
   family (the O22/C2 → controller-doc shape) and confirm the
   hub penalty demotes the building-mediated distractor rather
   than the gold doc.
5. Point the live eval at the live MLflow server
   (`python -m scripts.eval_retrieval_live --tracking-uri <uri>`)
   and confirm the `agentic-runtime-retrieval` experiment shows
   the runs. (The old `python -m scripts.eval_retrieval_phase2`
   command is deleted with the fake eval.)
6. Grow the corpus before trusting Recall@20: with 7 docs it
   is saturated by construction.
7. Re-measure the context budget with the real tokenizer
   (`estimate_tokens` is chars/4, an approximation).

## 8. Integration decision (documented, NOT wired live)

Unchanged from Phase 1: the hybrid retriever is a new module
evaluated through the Phase 0 harness. `search_documents` and
the `retrieval_context` hook are untouched, so no mission's
behaviour changes in this phase. The follow-up wiring step
(opt-in mission-config flag, e.g. `execution.retrieval.hybrid:
true` on `fault_analysis` only, read by the `retrieval_context`
hook) is the same as Phase 1's, with two Phase 2 additions:
the hook must pass real corpus doc-counts for seed
specificity (see §7.3), and the trace it logs to MLflow
should include the per-candidate G6/G7/G8 values (already in
the retriever trace: `g6_coverage`, `g7_ppr`, `g8_hub`,
`g8_transit`). No change to `search_documents` / the tool
surface is needed; the hybrid module already accepts injected
embeddings/client/store.
