# Ontology runtime — canonical (PostgreSQL) + projection (Neo4j)

The knowledge graph has two layers. **L1 canonical** is the only
source of truth and lives in PostgreSQL. **L2 projection** copies
canonical rows into Neo4j, where the Read API queries them. Nothing
writes Cypher `MERGE` directly any more: ingest, seeds and the
backfill script all go through L1, and L2 follows.

> History: the graph used to be FalkorDB, written to directly. It was
> replaced by this canonical + Neo4j design. The only FalkorDB remnant
> is the backfill script and its driver pin (see "Deferred" below).

## L1 — canonical (PostgreSQL)

### Tables (migration `0003`, indexes `0004`)

| Table | Contents |
| --- | --- |
| `ontology_entities` | One row per entity: `entity_type`, `entity_key`, `properties` (JSON), `status`, `version`, `state_hash`, `trace_id`, `source_event_id` |
| `ontology_relationships` | One row per relationship: `relationship_type`, source/target type+key, `properties`, `status`, `version`, `state_hash`, `trace_id`, `source_event_id` |
| `ontology_versions` | Append-only version history: one row per write (`subject_kind`, `subject_ref`, `version`, `change_type`, `snapshot`, `trace_id`, `source_event_id`, `transaction_time`). Never updated, never deleted. |
| `ontology_projection_state` | The L2 outbox: one row per subject (`subject_kind`, `subject_ref`, `canonical_version`, `projected_version`, `status`, `change_type`, `attempts`, `last_error`, `trace_id`). |

### Identity

- Entity types (v1 allowlist, `ontology/canonical/schema.py`):
  `Asset`, `Building`, `Component`, `Document`.
- Relationship types (v1 allowlist): `PART_OF`, `HAS_COMPONENT`,
  `LOCATED_IN`, `RELATED_TO`, `DOCUMENTED_BY`.
- Entity identity is `(entity_type, entity_key)`; keys are stripped,
  max 200 chars, and must not contain `:` or control characters —
  keys are embedded in `subject_ref` values
  (`entity:Asset:O21`, `rel:PART_OF:Asset:O21:Building:B1`) whose
  parsers split on `:`, so a colon in a key corrupts the ref and
  poisons the projection queue.
- Relationship identity is
  `(relationship_type, source_type, source_key, target_type, target_key)`.
- Statuses: `active`, `archived`, `deleted`.

### Writing — `OntologyWriteService`

All canonical writes go through `OntologyWriteService`
(`ontology/canonical/write_service.py`):

- `write_entity(...)` / `write_relationship(...)` — insert or update
  a row, append a version row, and enqueue the subject for projection
  (upsert into `ontology_projection_state` with `status="pending"`).
- `ensure_entity(...)` — create the entity only if it does not exist;
  an existing endpoint is left untouched (a noop, not a replay), so a
  relation merge that knows only endpoint identity never nulls out
  properties an earlier write stored.
- Every write carries `WriteProvenance(trace_id, source_event_id)`:
  `trace_id` is required; `source_event_id` identifies the source
  event (e.g. an ingest file row) and drives idempotency.
- `state_hash` is a SHA-256 of `{properties, status}`. A write whose
  hash and status already match the stored row is a **noop** (no new
  version, no re-projection). A write with the same `source_event_id`
  as the stored row is a **replay** (same: no new version). Otherwise
  the version increments (`_next_version` takes the max of the row
  version and the latest version row, so concurrent writers cannot
  collide).
- Concurrent duplicate inserts are retried once from a `SAVEPOINT`
  (`IntegrityError` → re-read the row and apply as an update).
- `force_update=True` bypasses the noop/replay short-circuits. Callers
  should not set it unless they mean "write a new version even if
  nothing changed".

### Reading canonical rows

`ontology/canonical/entity_read.py::get_entity` is the API-level read:
it returns `None` for missing rows **and** for `deleted`/`archived`
rows (the API answers 404). The filter lives there deliberately —
**not** in `repository.get_entity`, because the write service and the
projection engine rely on the repository returning non-active rows
(write upsert + delete-from-graph decisions).

## L2 — projection (outbox → Neo4j)

### Outbox + drain

- Every canonical write enqueues its subject in
  `ontology_projection_state` (`status="pending"`). The drain claims
  pending rows oldest-first (`claim_pending_projections`, limit per
  batch), projects each one, and marks it `synced` — conditionally:
  the mark only succeeds if `canonical_version` still equals the
  version that was read, so a write landing mid-projection is not
  silently dropped (the row stays `pending` for the next drain).
- Projected nodes/relationships carry their canonical properties
  **plus** `status`, `trace_id` and `canonical_version`, so a Neo4j
  node can always be traced back to its canonical row.
- Projection is a full property replace (`SET n = $props`,
  `SET r = $props`): stale properties are removed, never left behind.
- Relationship projection `MATCH`es its endpoints first and never
  creates stub nodes; a relationship whose endpoints are not yet
  projected fails that drain row and retries later.
- A `deleted`/`archived` canonical row deletes the node/relationship
  from Neo4j on projection (archived is **not** kept in the graph —
  see "Deferred").
- A `status="synced"` row is never re-projected unless a new canonical
  write re-enqueues it.

### How the drain runs

| Trigger | How |
| --- | --- |
| Celery beat | Task `ontology.projection_drain`, every `PROJECTION_DRAIN_INTERVAL_SECONDS` (default 60 s), batch `PROJECTION_DRAIN_BATCH_SIZE` (default 32). No catch-up: a failed beat just leaves rows `pending` for the next beat. |
| CLI | `python -m scripts.project_ontology_queue [--limit N]` (default limit = `PROJECTION_DRAIN_BATCH_SIZE`). Exits 1 when the drain reports errors. |
| HTTP | `POST /ontology/internal/project-drain` (same batch size as the beat task). See "Read API" — "internal" means the same `X-API-Key`, no separate token. |

The drain builds the Neo4j store lazily and checks health first. When
Neo4j is unhealthy the drain returns a uniform shape
`{"processed": 0, "errors": 1, "dead": 0, "claimed": 0, "detail": "neo4j_unhealthy"}`
and projects nothing.

### Poison / dead policy

A projection row that fails `projection_max_attempts` times (default 5, setting `PROJECTION_MAX_ATTEMPTS`) is marked `status = "dead"` and is never re-claimed: the claim query only selects rows with `status = "pending"`. The row keeps its `last_error` so an operator can inspect why it died. A dead row is terminal — it is not retried automatically. To re-queue it, re-enqueue the subject via `upsert_projection_pending` (which resets status to `"pending"`); if it fails again it goes dead again. Each drain result reports a `"dead"` count (number of rows that went dead in that drain). Note that `attempts` is reset to 0 when a **dead** row is re-enqueued (`upsert_projection_pending` in `ontology/canonical/repository.py`), so a re-queued dead row gets a fresh `projection_max_attempts` failures before going dead again and can actually recover; re-enqueueing a `pending` / `projecting` row leaves its attempt count untouched.

### Neo4j constraints

`Neo4jGraphStore` runs `ensure_constraints()` best-effort at init:
one uniqueness constraint per v1 entity type,
`CREATE CONSTRAINT IF NOT EXISTS FOR (n:Label) REQUIRE n.id IS UNIQUE`.
A failure is logged as a warning and does not crash the store
(the constraint may already exist, or Neo4j may be briefly down).

### Backend switch

`GRAPH_DATABASE_BACKEND` is a hard switch in the graph-store factory
(`ontology/graph/factory.py`): only `"neo4j"` is allowed, anything else
raises `ValueError` at store construction. There is no FalkorDB backend
any more.

## Read API

Base path `/ontology`, router `app/routers/ontology.py`. All endpoints
require the platform API key (`X-API-Key` header) unless `API_KEY` is
empty (dev only, auth disabled). `entity_type` must be in
`ENTITY_TYPES_V1` and `entity_key` must be valid (Asset keys match
`^[A-Za-z0-9_-]{1,64}$`); otherwise **400**. A limit/depth outside its
range is **422**. Neo4j (or the canonical store) unreachable is **503**.

| Endpoint | Description |
| --- | --- |
| `GET /ontology/entities/{type}/{key}` | Canonical entity (PostgreSQL). **404** when missing, `deleted` or `archived`. **503** `"store unavailable"` when the canonical read fails. |
| `GET /ontology/entities/{type}/{key}/relationships` | Neighbours from Neo4j (depth 1). `?limit=` (default 50, 1–200). **503** `"graph unavailable"` when Neo4j is down. |
| `GET /ontology/entities/{type}/{key}/knowledge` | Documents linked via `RELATED_TO` / `DOCUMENTED_BY`. `?limit=` (default 50, 1–200). **503** when Neo4j is down. |
| `GET /ontology/graph?root_type=&root_key=&depth=&limit=` | Subgraph from Neo4j. `depth` default 2 (1–3), `limit` default 100 (1–500). **503** when Neo4j is down. |
| `POST /ontology/internal/project-drain` | Trigger one projection drain (batch = `PROJECTION_DRAIN_BATCH_SIZE`). Returns `{"processed", "errors", "dead", "claimed", "detail"}`. "Internal" means it uses the same `X-API-Key` as every other endpoint — there is no separate token. |

## Environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `GRAPH_DATABASE_BACKEND` | `neo4j` | Hard switch; only `neo4j` is accepted. |
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt URI of the Neo4j server. |
| `NEO4J_USERNAME` | `neo4j` | |
| `NEO4J_PASSWORD` | `changeme` | Change in production. |
| `NEO4J_DATABASE` | `neo4j` | |
| `GRAPH_FEATURES_ENABLED` | `true` | `false` makes graph reads return empty instead of raising (mission runtime only). |
| `PROJECTION_DRAIN_INTERVAL_SECONDS` | `60` | Beat interval for `ontology.projection_drain`. |
| `PROJECTION_DRAIN_BATCH_SIZE` | `32` | Rows per drain (beat, CLI default, internal endpoint). |
| `PROJECTION_MAX_ATTEMPTS` | `5` | Failures before a projection row goes `dead`. |

No `FALKOR_*` variables exist in the app any more. `FALKOR_HOST` /
`FALKOR_PORT` are accepted **only** as env fallbacks by the one-off
backfill script below, never by `app/config.py`.

## How to run Neo4j + run the drain

```bash
# Start the stack (Neo4j is a compose service with a healthcheck):
docker compose up -d neo4j
# or start everything: docker compose up -d

# Apply migrations (creates the ontology_* tables):
make migrate

# Seed the demo ontology (O21 + 5 relations) and project it:
make seed
# or ontology only: python -m seeds.seed_ontology_demo

# Drain the queue by hand (beat does this every 60 s anyway):
python -m scripts.project_ontology_queue
# or via the API:
curl -X POST http://localhost:8000/ontology/internal/project-drain \
  -H "X-API-Key: $API_KEY"
```

## Backfill — FalkorDB → canonical

One-off script: `scripts/backfill_falkor_to_canonical.py`. It reads
**all** nodes and relationships from a FalkorDB graph and writes them
through `OntologyWriteService` into the canonical tables; the normal
L2 projection then carries them into Neo4j.

**Run it BEFORE deleting the FalkorDB service.** If there is no
production graph data to migrate, skip it entirely and re-seed /
re-ingest instead (`make seed`, `make ingest`).

```bash
# Dry run first — counts what would be imported, writes nothing:
.venv/bin/python -m scripts.backfill_falkor_to_canonical \
  --falkor-host localhost --falkor-port 6379 \
  --falkor-graph knowledge_graph --dry-run

# Real run (same flags without --dry-run). Env fallbacks for this
# one-off run only: FALKOR_HOST / FALKOR_PORT.
.venv/bin/python -m scripts.backfill_falkor_to_canonical \
  --falkor-host localhost --falkor-port 6379 \
  --falkor-graph knowledge_graph
```

Rules and guarantees:

- Needs the driver: `pip install falkordb==1.2.0` (backfill-only pin
  in `requirements.txt`). If it is missing, the script exits non-zero
  with that instruction. If FalkorDB is unreachable, it exits
  non-zero with a clear message and writes nothing.
- Only entity types in `ENTITY_TYPES_V1` are written; anything else
  is skipped with a warning. Relationship types must be in
  `RELATIONSHIP_TYPES_V1`. A node needs an `id` property (its key);
  nodes without one are skipped with a warning. The node's `id`
  property itself is not stored as a property (it is the key).
- One session, entities committed first, then relationships; on any
  error the session rolls back and the script exits 1.
- Idempotent: `trace_id="falkor-backfill"`,
  `source_event_id="falkor:{type}:{key}"` per node and
  `source_event_id="falkor:rel:{from_key}:{rel_type}:{to_key}"` per
  relationship, so a re-run replays instead of duplicating.
- Relationship endpoints that have no node row in FalkorDB are created
  bare (`ensure_entity`, no properties) so the relationship still has
  endpoints.
- After the backfill, run the drain (`python -m
  scripts.project_ontology_queue`) and check the Neo4j graph before
  deleting the FalkorDB service.

## Deferred

- **FalkorDB pin removal**: `FalkorDB==1.2.0` stays in
  `requirements.txt` (commented "backfill-only") until the migration
  is done and the backfill script is deleted; remove both together.
- **Archived stays deleted-from-graph**: `archived` canonical rows are
  projected as deletions from Neo4j, same as `deleted`. Keeping
  archived entities in the graph was considered and deferred.
- **No separate internal token**: `POST /ontology/internal/project-drain`
  uses the same `X-API-Key` as the rest of the API. A separate token
  was considered and deferred.
- **Document type kept**: `Document` / `DOCUMENTED_BY` stay in the v1
  schema even though documents also live in Qdrant; the graph link is
  what the `/knowledge` endpoint reads.
