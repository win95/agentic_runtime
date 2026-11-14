# Hybrid retrieval — Phase 5: Live wiring (real Qdrant + Neo4j, no new scoring)

Report: `~/workspace/research_notes/hybrid-vector-graph-retrieval-20261114-0430/report.md`.
Phase 0: `docs/hybrid-retrieval-phase0.md`. Phase 1:
`docs/hybrid-retrieval-phase1.md`. Phase 2:
`docs/hybrid-retrieval-phase2.md`. Phase 3:
`docs/hybrid-retrieval-phase3.md`. Phase 4:
`docs/hybrid-retrieval-phase4.md`. Consolidated as-built design:
`docs/hybrid-retrieval.md`. Pilot mission: `fault_analysis`.

**No live numbers exist.** There is no Qdrant/Neo4j in this VM
and `EMBEDDING_MODEL` is set by the user in his own `.env` on
his machine, so every metric in this doc is either a fake-setup
number carried over from Phases 0–4 (labelled as such) or
unverified-live. The only live-mode behaviour verified in this
VM is the fail-fast preflight (§5, §7).

## What was built

| File | What it is |
|---|---|
| `runtime/retrieval_hybrid.py` | `build_live_hybrid_retriever` (new factory, §1) + the three live-source kwargs on `HybridRetriever` (§2). No fusion, weight, guardrail or gating change. |
| `infra/qdrant_client.py` | `count_documents_per_node`, `get_document_links`, `get_document_payloads` (new, §2) over one shared `_scroll_payloads` helper. |
| `runtime/retrieval_graph.py` | `build_live_adjacency` + `build_expansion_adjacency` (§3), `refresh_gazetteer_codes` + `clear_gazetteer_codes` and the dynamic-gazetteer live mode (§4). |
| `ontology/knowledge/graph_query.py` | `list_entity_keys` (new, §4) — the read the dynamic gazetteer uses. No new Cypher anywhere in Phase 5. |
| `scripts/eval_retrieval_live.py` (new) | Live golden-set eval: preflight → inventory → live baseline → live hybrid (§5). `make eval-retrieval-live`. |
| `tests/test_retrieval_live.py` (new) | 29 tests: factory, Qdrant helpers, live-source replacement, live adjacency. |
| `tests/test_retrieval_live_eval.py` (new) | 10 tests: preflight ordering/fail-fast, baseline closure, inventory. |
| `tests/test_retrieval_gazetteer_live.py` (new) | 15 tests: dynamic gazetteer modes, refresh/clear, fault-code limit. |
| `Makefile` | New `eval-retrieval-live` target (+ `.PHONY`). |

Changed nothing else. Untouched on purpose: `search_documents`
(the vector arm itself), the `retrieval_context` hook, the route
node, skill selection, `app/config.py`, `runtime/llm_factory.py`,
the embedding paths in `infra/qdrant_client.py`,
`services/user_service/`, `.env.example` (see §1 — all three env
vars were already documented there), and every other mission.
No new dependency was added. **No execution-path wiring was
done** — the opt-in wiring follow-up (`execution.retrieval.hybrid`
/ `query_plan`) is still NOT done; see
`docs/hybrid-retrieval.md` §12.

## 1. Live factory — `build_live_hybrid_retriever`

`runtime/retrieval_hybrid.py::build_live_hybrid_retriever(*,
client=None, graph_store=None, embeddings=None, phase2: bool =
True, w_ppr: float = W_PPR, **kwargs)` — `W_PPR` is 0.45
(`runtime/retrieval_graph.py`), the locked Phase 2 prior, not a
new value. It returns a `HybridRetriever` with real sources
injected (§2) and `use_live_graph=True`; the retriever itself
is unchanged — same fusion, same guardrails, same gating.

What it injects, in order:

* **Query embeddings only** — `build_embeddings(model=
  settings.resolved_embedding_model_query)`, i.e.
  `EMBEDDING_MODEL_QUERY` falling back to `EMBEDDING_MODEL`,
  read from settings. Only the query model is built here: the
  retriever embeds questions only. The INDEX model governs
  seeding via the existing `embed_texts` default, and the two
  models must produce compatible vectors (same dimension /
  vector space — changing the index model requires
  re-indexing Qdrant).
* **Real Qdrant client** — `infra.qdrant_client.get_client()`.
* **Live graph, lazily** — `graph_store=None` is passed
  through; `expand_via_graph_query` resolves the default from
  `ontology/projection/provider.py::get_graph_store` on first
  use.
* **Dynamic gazetteer, best-effort** — calls
  `refresh_gazetteer_codes(store=graph_store)` (§4); on any
  failure a warning is logged and the factory continues with
  whatever gazetteer state results. A failed refresh leaves the
  gazetteer in live mode with an empty cache (it links nothing
  rather than falling back to demo codes).

**No connections at construction.** `get_client()` only builds
the client object; the graph store stays `None` unless passed;
the Qdrant payload callables (§2) resolve lazily on first
query. Constructing the retriever is safe with the stores
down — the first query (or the preflight, §5) is what fails.

**Fail fast** (at construction, before any retrieval):

* Empty `QDRANT_URL` → `RuntimeError` naming Qdrant.
* Missing embedding config → `ValueError` from
  `require_embedding_config`.
* `get_client()` returning `None` despite a set URL →
  `RuntimeError`.

**Error contract** (the same contract the live eval and any
future wiring rely on):

* Graph-store failure → `graph_missing` → the pure-vector
  fallback (guardrail 3, the existing behaviour in
  `expand_via_graph_query`). Never a crash, never a partial
  graph silently treated as complete.
* Qdrant failure → `RuntimeError` naming Qdrant, propagated.
  Never swallowed into empty results.

**Env vars:** no new ones. `QDRANT_URL`, `NEO4J_URI` and
`EMBEDDING_MODEL` already exist and are already documented in
`.env.example` (Qdrant lines 65–67, Neo4j lines 71–74,
embedding lines 34–50, including the
`EMBEDDING_MODEL_INDEX` / `EMBEDDING_MODEL_QUERY` split) —
verified, so `.env.example` was NOT touched.

## 2. Real-source replacement in `HybridRetriever`

`HybridRetriever.__init__` gains three optional kwargs —
`doc_counts`, `doc_texts`, `doc_links` — each a precomputed
dict or a callable, resolved **lazily (once, cached, on the
first query that needs them)** and never at construction. The
fake path (all three `None`) keeps the demo-table behaviour
byte-identically and never touches Qdrant; the live factory
supplies callables bound to the real client.

| Kwarg | Replaces (demo) | Live source |
|---|---|---|
| `doc_counts` | `DOCUMENT_LINKS`-derived counts in `seed_specificity` | `count_documents_per_node(client=live_client)` |
| `doc_links` | `DOCUMENT_LINKS` in the graph arm | `get_document_links(client=live_client)` |
| `doc_texts` | `seeds/demo_data.py::DOCUMENTS` for graph-only candidates | `get_document_payloads(doc_ids, client=live_client)` (callable taking the doc ids to fetch) |

**Why a third leg (`doc_links`) was added, beyond the brief's
two:** the graph arm built its `graph_doc_ids` directly from
the demo `DOCUMENT_LINKS` table, so with only counts and texts
replaced the live graph arm could never find a non-demo doc.
`doc_links` is the live replacement for that table: a doc is a
graph candidate when its linked nodes touch a seed or a 1-hop
neighbour.

**The Qdrant helpers** (`infra/qdrant_client.py`), all three
raising `RuntimeError` naming Qdrant on failure:

* `count_documents_per_node(client=None, collection_name=None)
  -> dict[str, int]` — scrolls every point and counts its
  payload `node_ids`. `seed_specificity` treats a missing /
  empty count as maximally specific, so an empty store
  degrades to `{}` here, never a crash.
* `get_document_links(client=None, collection_name=None) ->
  dict[str, list[str]]` — doc id → payload `node_ids`. Docs
  with no payload `node_ids` are omitted (an unlinked doc
  scores `graph_score = 0` and the vector arm carries it). An
  empty store returns `{}`.
* `get_document_payloads(doc_ids, client=None,
  collection_name=None) -> dict[str, dict]` — payloads
  (title / content / node_ids / …) for the given doc ids, via
  one batched `client.retrieve`. Point ids use the same uuid5
  scheme as `upsert_documents`
  (`uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{doc_id}")`). Unknown
  ids are simply absent from the result. This is the fetch for
  graph-only candidates (docs the graph arm found that the
  vector arm did not), so the exact feature and the
  title/content in the candidate row come from real payloads,
  not the demo corpus. An empty `doc_ids` list never calls
  the client.

The two scroll-based helpers share `_scroll_payloads`: one
scroll pattern, `limit=100`, `with_payload=True`; a scroll
failure surfaces as `RuntimeError("Qdrant scroll failed: …")`,
never a raw client error and never a silent empty result.

**Degradation, stated plainly:** an empty store yields counts
`{}`, links `{}`, and no texts — a doc with no text contributes
no text feature (exact 0), never a crash. The retriever still
ranks by the vector arm; the graph arm simply finds nothing.

## 3. Live adjacency — PPR / coverage / hub over real edges

Before Phase 5, the live path had a hole: the 1-hop expansion
came from the real Neo4j store, but PPR (G7), coverage (G6)
and hub (G8) ran over `self.adjacency` — the demo adjacency.
A live doc's PPR computed from demo edges would be
meaningless. Worker E closed it with two builders in
`runtime/retrieval_graph.py`, both the exact shape of
`build_demo_adjacency()` (node key → `{"type": str,
"neighbors": [(rel, to_key, to_type)]}`):

* `build_live_adjacency(seeds, expansion, *, store=None,
  timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
  max_nodes: int = MAX_LIVE_ADJACENCY_NODES) -> dict` —
  `DEFAULT_TIMEOUT_SECONDS` is 0.15, `MAX_LIVE_ADJACENCY_NODES`
  is 200. Contents: the seeds + the 1-hop expansion nodes +
  the 1-hop neighbours of each expansion node, via
  `ontology/knowledge/graph_query.py::neighbors` depth 1 — no
  new Cypher. Only `ALLOWED_RELATIONS` edges. Neighbours are
  processed sorted by key, nodes visited in BFS order (seeds
  in given order, then expansion nodes sorted by key, then
  second-hop nodes in discovery order), so the same store
  contents always yield the same adjacency.
* `build_expansion_adjacency(seeds, expansion) -> dict` — the
  degradation substrate: seeds + expansion nodes + their
  seed→node edges only. Real store data (the 1-hop expansion
  rows), partial — no second-hop edges, so PPR / coverage /
  hub see only the seed fringe. Same shape as the demo
  adjacency.

**The two caps, and why they are shaped this way:**

* **Total nodes ≤ 200** (counting seeds + expansion +
  second-hop nodes). Seeds + expansion nodes are always
  present (they are the query's own fringe); the cap binds on
  second-hop additions only — an edge to a not-yet-known node
  past the cap is dropped, edges between known nodes are
  still added, and the truncation is reported with a warning
  log.
* **One shared wall-clock budget of 0.15 s**, checked before
  every `neighbors()` call, NOT per call. A per-call budget
  over up to ~51 calls (seeds + expansion) could stretch to
  ~7.6 s and blow the retrieval latency budget. On timeout
  the build stops and returns the partial adjacency built so
  far (seeds + expansion nodes are always in it, so it is
  never empty for a non-empty seed set).

**A store error RAISES out of the builder** — it is not
swallowed there. The caller (`HybridRetriever._candidate_rows`)
treats a failed live build as a degradation (below). A partial
graph after an error would be indistinguishable from a
complete small graph, so the builder never returns one.
`{}` is returned only when there are no seeds (the caller
never calls it then — `graph_missing` applies instead).

**Wiring** (`HybridRetriever._candidate_rows`, live path only):
when `use_live_graph=True` and the expansion is non-empty,
PPR / G6 / G8 run over the live adjacency; the trace gains
`graph_adjacency_source` (`"live"` / `"expansion-only"`) and
`graph_adjacency_nodes`, on the live path only. The fake path
keeps `self.adjacency` (byte-identical behaviour).

**Degradation:** if the live build raises or comes back empty
while the 1-hop expansion itself succeeded, the retriever
falls back to the expansion-only adjacency — real data,
partial — **never the demo adjacency on the live path** (a
demo edge would be a fabricated edge about the live
ontology) and never a crash. If the expansion itself is
empty, the existing `graph_missing` path applies unchanged
(no live build is attempted).

## 4. Dynamic gazetteer — keys from the live ontology

`ontology/knowledge/graph_query.py::list_entity_keys(
entity_type, *, limit: int = 200, store=None) -> list[str]` —
`MATCH (n:Label) RETURN DISTINCT n.id`, reusing the `_label`
validation (an arbitrary caller string can never inject
Cypher) and the 1..200 cap; rows with a missing / non-string
`id` are skipped; the result is sorted plain strings, so a
refresh is deterministic for the same store contents.

`runtime/retrieval_graph.py::refresh_gazetteer_codes(store=None)
-> dict` fetches Asset / Component / Building / Document keys
via `list_entity_keys`, compiles the match patterns once, and
commits only on full success, returning e.g.
`{"assets": n, "components": n, "buildings": n, "documents": n,
"source": "ontology"}`. `clear_gazetteer_codes() -> None`
drops the cache and returns to demo mode (a test helper, and
the manual way back from live mode).

**Cache:** in-memory only, no TTL, no beat job, no timer —
refresh is called explicitly by the live factory (and by
hand). Patterns are compiled once at refresh; a second
refresh replaces the cache wholesale, so no stale keys
survive it.

**Two modes, on an explicit flag** (`_DYNAMIC_MODE`, set by
the first refresh):

* **Demo mode (default, before any refresh):**
  `gazetteer_seeds` is byte-identical to Phases 1–4.
* **Live mode (refresh called ≥ once): the demo patterns are
  DISABLED — only cached dynamic keys link.** An unreachable
  ontology → an empty cache → links nothing. There is NO
  fallback to demo codes in live mode.

**The degradation rule, and why (decision 1 of §6):** a
failed refresh **clears any earlier good cache** and returns
`{"assets": 0, "components": 0, "buildings": 0, "documents": 0,
"source": "unreachable"}`, keeps live mode ON, and never
raises. A stale cache could link codes the ontology no longer
has, and **a wrong link is worse than no link**: a live
question naming O21 when the live ontology has no O21 must
not link. Linking nothing degrades to vector-only
(guardrail 3); linking wrongly would feed the graph arm a
false seed and poison the fusion.

**Fault codes are NOT fetched.** There is no Fault entity
type, and `FAULT_CODE_LINKS` maps fault codes to
asset+component from the demo table — so the dynamic list
covers Asset / Component / Building / Document keys only.
Fault-code expansion stays demo-grounded in demo mode and
**does not run at all in live mode: a fault code links
nothing** until Fault becomes an ontology entity type. A
question that names only a fault code (e.g. "F-204") is a
gazetteer miss live, and expands / falls back exactly as a
seedless question does.

**Matching reuses the Phase 4 normalisation:** verbatim hit
= 1.0, separator / diacritics variant = 0.8, a 1.0 twin wins
the dedup (same rule as the demo pass). Both patterns are
whole-token bounded (`\b` both ends), so a key embedded in a
longer token never matches. Seeds come out in first-mention
order.

## 5. Live eval mode — `scripts/eval_retrieval_live.py`

`make eval-retrieval-live` (flags `--tracking-uri` /
`--output` / `--golden-set`, mirroring the Phase 1 eval).
Flow:

1. **Preflight** — `preflight_live_stores(*, client=None,
   graph_store=None) -> dict` (importable, so tests call it
   with mocked stores). Ordered, fail-fast, exit code 2 via
   `LivePreflightError`:
   * (i) **Embedding config** — `require_embedding_config`;
     the message is reworded to name the fix: set
     `EMBEDDING_MODEL` in `.env` to the model name your
     LiteLLM proxy serves.
   * (ii) **Qdrant reachable AND the collection exists AND
     holds ≥1 point** — a missing / empty collection names
     Qdrant and the seed step (`make seed` for demo docs with
     payload links, `make ingest` for real technical
     documents).
   * (iii) **Neo4j** — `is_healthy()` when the store has it,
     else a one-row `run_query` probe, plus a best-effort
     node count (a failed count never flips reachability).
     This one **DEGRADES to a vector-only run with a warning,
     never hard-fails** (decision 3 of §6): a missing graph
     is a legitimate live state — guardrail 3.
2. **Store-inventory preamble** — Qdrant URL / collection /
   dim / point count, resolved embedding model names
   (index + query), Neo4j reachability + node count.
3. **Live plain-vector baseline** —
   `build_live_baseline_retriever(limit=20)`, a `retrieve`
   closure over the real `search_documents` + real query
   embeddings, no graph arm.
4. **Live hybrid** — `build_live_hybrid_retriever()`.
5. Both go through the Phase 1 harness (`run_retrieval_eval`
   + `format_results_table`): the same aggregate +
   sibling-confusion tables.

**No grid search** on the live run — live grid re-runs are the
user's call (`docs/hybrid-retrieval.md` §11.6).

**The verdict is INFORMATIONAL.** It uses the Phase 1 exit
criterion (hybrid nDCG@6 > baseline, Recall@6 ≥ baseline,
sibling nDCG@6 > baseline) but the script exits 0 once the
eval completes, **even on FAIL** — the corpus is small and
real embeddings retain code signal, so a live FAIL is
printed, not treated as a code failure. Exit code 2 means the
preflight failed, before any retrieval.

**Observed in this VM (the only live-mode behaviour verified
here):**

* Default env → `PREFLIGHT FAILED: Embeddings are not
  configured: missing EMBEDDING_API_KEY (or LLM_API_KEY). …
  Set EMBEDDING_MODEL in .env to the model name your LiteLLM
  proxy serves.` — exit 2.
* With a dummy embedding config → `PREFLIGHT FAILED: Qdrant
  is unreachable at http://localhost:6333: … Connection
  refused. Start Qdrant (docker compose up) and check
  QDRANT_URL in .env.` — exit 2.

Both failures name the store and the fix, in order, exactly
as designed. No retrieval ran in either case.

## 6. The three documented decisions

1. **Gazetteer degradation: a wrong link is worse than no
   link** (§4). A failed refresh clears any earlier good
   cache, keeps live mode on, and links nothing — rather
   than falling back to demo codes or keeping stale keys.
   The cost is deliberate: a live question that names a code
   the ontology no longer has degrades to vector-only
   (guardrail 3), which is safe; a false seed would poison
   the fusion with a real-looking graph signal.
2. **Error contract: graph degrades, Qdrant raises** (§1).
   A graph-store failure is a degradation (`graph_missing`,
   pure-vector fallback) because the graph is an advisory
   arm — the vector arm alone still answers. A Qdrant
   failure raises `RuntimeError` naming Qdrant and is never
   swallowed into empty results, because an empty vector arm
   is indistinguishable from "no documents match" and would
   silently return a wrong answer. The asymmetry is
   deliberate: the arm that can answer alone may fail
   quietly; the arm that cannot may not.
3. **Preflight: Neo4j-down degrades, Qdrant-down fails**
   (§5). Step (iii) never hard-fails — a missing graph is a
   legitimate live state (guardrail 3) and the run still
   produces the live vector baseline + a vector-only hybrid.
   Steps (i) and (ii) fail fast with exit 2, because without
   embeddings or a non-empty collection there is nothing to
   evaluate at all.

**Unchanged, stated once more:** no scoring / fusion formula,
weight, guardrail, or gating threshold changed anywhere in
Phase 5. `final = 0.50·z(vec) + 0.40·z(graph_score) +
0.10·exact`, `graph_score = 0.45·minmax_log(PPR) +
0.30·z(path) + 0.25·coverage`, the four guardrails, and the
0.35 / 0.15 gating thresholds are exactly as documented in
`docs/hybrid-retrieval.md` §3/§6. Phase 5 replaces the data
sources under the formula, not the formula.

## 7. What was verified — fakes / mocks vs the user's machine

**Verified here (54 new tests, main-feature only; full suite
670 passed, ruff clean):**

* Factory: injects the real stores, builds query-model
  embeddings only, fails fast on empty `QDRANT_URL` and on
  missing embedding config, tolerates refresh failure /
  missing refresh / a refresh finding nothing.
* Qdrant helpers: counts / links / payloads from payload
  `node_ids`, scroll past 100 points, empty store degrades,
  no client raises, empty doc-id list never calls the
  client, search failure surfaces as `RuntimeError`.
* Live-source replacement: the live retriever reads counts
  and texts from Qdrant, an empty store degrades, precomputed
  dict sources are used as given.
* Live adjacency: real second-hop edges, no-seeds empty,
  cap deterministic (sorted-first neighbours kept), timeout
  returns partial, the live retriever uses the live adjacency
  for PPR / coverage, a failed live build falls back to
  expansion-only.
* Dynamic gazetteer: live mode disables demo patterns,
  refresh / clear round-trip, failed refresh clears the
  cache, fault codes link nothing in live mode.
* Live eval: preflight ordering and fail-fast, the baseline
  closure, the inventory preamble.
* Preflight observed live in this VM, twice, exit 2 (§5) —
  the only live-mode behaviour verified here.

**NOT verified here — needs the user's machine
(`docs/hybrid-retrieval.md` §11):**

* Every live number. There are no live nDCG@6 / Recall@6 /
  sibling numbers anywhere in Phase 5 — the golden set has
  never run against real embeddings + Qdrant + Neo4j. The
  fake-setup numbers in `docs/hybrid-retrieval.md` §8
  (Phase 1 blurred 0.906 → 0.942, Phase 2 0.942 → 0.958,
  Phase 3 / 4 no measurable gain) are unchanged by this
  phase and remain fake-setup numbers.
* The live factory against a real Qdrant collection (real
  payload `node_ids`, real point ids, real dims) and a real
  Neo4j graph (real degrees — the demo's buildings have
  degree 4–8, so the hub penalty was validated on small
  degrees only; a Building with degree >50 is a real hub).
* The dynamic gazetteer against the real ontology's key
  vocabulary (including which codes the live ontology
  actually has — and therefore which live questions link).
* The live adjacency under the 0.15 s budget on a real graph
  (how often the cap / timeout binds, and whether the
  expansion-only fallback is the common case).
* The live eval end-to-end (`make eval-retrieval-live` past
  the preflight), the informational verdict on real data,
  and the w_graph grid re-run live (0.50 first).

## 8. Live verification (user's machine)

The full consolidated checklist is in
`docs/hybrid-retrieval.md` §11; the Phase 5 items are:

1. Set `EMBEDDING_MODEL` in `.env` to the model name your
   LiteLLM proxy serves (he sets it himself — never ask for
   its value), then seed Qdrant (`make seed` for demo docs
   with payload links, or `make ingest` for real technical
   documents) and confirm with a Qdrant scroll that all
   points carry non-empty `node_ids`; backfill an existing
   collection with `python -m
   scripts.backfill_qdrant_links`.
2. Drain the ontology projection into Neo4j (the L2 queue
   drain) and confirm the graph the gazetteer and the live
   adjacency will read is the live one.
3. Run `make eval-retrieval-live` (pass
   `TRACKING_URI=<mlflow uri>` to log to the live MLflow
   server) and replace every fake number in
   `docs/hybrid-retrieval.md` §8 with live ones. Read the
   informational verdict as what it is: on a 7-doc corpus
   with real embeddings, a FAIL is a finding about the
   corpus / embeddings, not a code failure. Expect the live
   sibling-subset gain to be SMALLER than the blurred fake's
   (real embeddings retain some code signal).
4. Check the live trace: `graph_adjacency_source` should
   read `"live"` on queries with a non-empty expansion
   (`"expansion-only"` means the live build failed or timed
   out — check the warning log), and `graph_adjacency_nodes`
   should stay ≤200. If `"expansion-only"` is the common
   case, the 0.15 s budget or the store latency needs
   looking at, not the retriever.
5. Check the dynamic gazetteer: the refresh summary should
   show the real key counts (`{"source": "ontology", …}`);
   `"unreachable"` with zero counts means the gazetteer
   linked nothing on the whole run — fix Neo4j / the store
   wiring, do not re-enable demo codes.
6. Re-run the w_graph grid live with 0.50 first (the
   first-to-recheck-live runner-up; fake grid 0.30 → 0.957,
   0.40 → 0.958, 0.50 → 0.978 — three swaps, not a live
   prediction).
7. Live expansion / QueryPlan A/Bs with the live generators
   (the Phase 3 / Phase 4 items in §11.7 / §11.11) — the
   gating thresholds (0.35 / 0.15) are fake-BoW cosine
   values; log the top-1 / margin distribution on live
   traffic and re-tune to ~10–30% of queries expanding.
8. Opt-in wiring follow-up (NOT done here): add
   `execution.retrieval.hybrid: true` /
   `execution.retrieval.query_plan: true` on
   `fault_analysis` only, read by the `retrieval_context`
   hook — when set, the hook calls
   `build_live_hybrid_retriever()` (and builds a QueryPlan
   with live generators for the `query_plan` flag), assembles
   context with `assemble_context`, and logs the trace to
   MLflow. When unset, the hook behaves exactly as today.
