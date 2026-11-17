# Ontology runtime — L1 state-only (PostgreSQL) + Neo4j holds the objects

> Rewritten 2026-11-17 for the redesign
> (`~/workspace/ontology-redesign-20261117.md`): L1 keeps **state only**,
> Neo4j is where objects live. The old design (L1 as source of truth,
> Neo4j as a projection copy, `ontology_versions` snapshots) is gone —
> do not follow the pre-2026-11-17 version of this doc.

The knowledge graph has two layers with different jobs:

| Layer | Holds | Role |
| --- | --- | --- |
| **L1 canonical** (PostgreSQL) | State only: identity, `status`, `version`, `trace_id`, `source_event_id`, projection state | Idempotency, audit of *that* something changed, drain queue |
| **Neo4j** | Objects: nodes + edges + properties + ontology metadata + derived edges | The real graph — every object read and every object write goes here |

Nothing writes Cypher `MERGE` directly any more except the write
service and the backfill script: ingest, seeds and the API all go
through `OntologyWriteService`, which dual-writes Neo4j + L1.

## L1 — state-only (PostgreSQL)

### Tables (migration `0015_ontology_state_only`, after `0014`)

| Table | Columns (kept) | Dropped in 0015 |
| --- | --- | --- |
| `ontology_entities` | `id`, `entity_type`, `entity_key`, `version`, `status`, `trace_id`, `source_event_id`, `created_at`, `updated_at` | `properties` (JSON), `state_hash` |
| `ontology_relationships` | `id`, `relationship_type`, source/target type+key, `version`, `status`, `trace_id`, `source_event_id`, `created_at`, `updated_at` | `properties` (JSON), `state_hash` |
| `ontology_projection_state` | `subject_kind`, `subject_ref`, `canonical_version`, `projected_version`, `status`, `change_type`, `attempts`, `last_error`, `trace_id` | — (kept, new role: see "Drain") |
| `ontology_versions` | — | **Whole table dropped** |

Why those columns went:

- `properties` — object data lives on Neo4j now; L1 has nothing to store.
- `state_hash` — it hashed `(properties + status)`; with no properties
  it has no meaning. Change detection now compares against the Neo4j
  node before writing (see "Writing").
- `ontology_versions` — full-object snapshots are gone by tran's
  decision 2 (2026-11-17). Consequence, accepted by design: **audit of
  WHAT-changed is lost** — L1 still knows a change happened (`version`,
  `change_type`, time, `trace_id`/`source_event_id`), but not what the
  object changed from/to. If object audit is ever needed, build a
  separate event log; do not put snapshots back into L1.

### Identity

- Entity types are classes defined in Neo4j metadata (see "Metadata"),
  with the v1 set as fallback when no metadata was seeded:
  `Asset`, `Building`, `Component`, `Document`
  (`ontology/canonical/schema.py`: `ENTITY_TYPES_V1`).
- Relationship types are object properties in metadata, v1 fallback:
  `PART_OF`, `HAS_COMPONENT`, `LOCATED_IN`, `RELATED_TO`,
  `DOCUMENTED_BY` (`RELATIONSHIP_TYPES_V1`).
- Entity identity is `(entity_type, entity_key)`; keys are stripped,
  max 200 chars, and must not contain `:` or control characters —
  keys are embedded in `subject_ref` values
  (`entity:Asset:O21`, `rel:PART_OF:Asset:O21:Building:B1`) whose
  parsers split on `:`, so a colon in a key corrupts the ref and
  poisons the projection queue.
- Relationship identity is
  `(relationship_type, source_type, source_key, target_type, target_key)`.
- Statuses: `active`, `archived`, `deleted`.

## Neo4j — objects + metadata + derived edges

### Instances

- **Node**: one label = the instance's direct class, plus properties
  `id`, `class` (= class id), `status`, `version`, and the user
  properties. Ancestor labels are **not** materialized on the node —
  subclass is expanded at query/validation time (R1).
- **Relationship**: type = object property id; properties on the edge
  as before, plus `status` / `version`.
- **Delete** is soft: L1 status → `archived`/`deleted`, Neo4j node
  `DETACH DELETE`d in the same write (derived edges go with it).
- Uniqueness: `Neo4jGraphStore.ensure_constraints()` creates one
  constraint per v1 label,
  `CREATE CONSTRAINT IF NOT EXISTS FOR (n:Label) REQUIRE n.id IS UNIQUE`
  (best-effort at store init; a failure is logged, not fatal).

### Metadata — Ontology Management (group 1)

Class, subclass, object property and data property definitions live
**as nodes on Neo4j**, not as constants in code
(`ontology/metadata/store.py`):

- `(:OntologyClass {id, name, description})` — 4 v1 classes
  (`Asset`, `Building`, `Component`, `Document`) are seeded as nodes.
  Adding a class = adding a node, not editing code.
- Subclass: `(:OntologyClass)-[:SUBCLASS_OF]->(:OntologyClass)`.
  Adding one checks for cycles (a class may not become a subclass of
  itself or of its own descendant).
- `(:ObjectProperty {id, domain, range, inverse_of})` — `domain` /
  `range` are class ids, or `"any"` for an unconstrained endpoint
  (`RELATED_TO`). 5 v1 relations are seeded, with `inverse_of`:
  `PART_OF` ↔ `HAS_COMPONENT`.
- `(:DataProperty {id, domain, datatype, required})` — one node per
  (property key, domain class) pair; the same key may be typed
  differently for different classes. Datatypes: `str` / `int` /
  `float` / `bool` (and the long names `string` / `integer` /
  `number` / `boolean`).
- Metadata is cached in-process for 60 s (`get_metadata`); every
  mutation invalidates the cache.
- `ontology/canonical/schema.py` now keeps only the Enums
  (`EntityStatus`, `ChangeType`, `WriteOutcome`, `ProjectionStatus`),
  the v1 allowlists (fallback only), and the identifier regex.

**Validation at write time** (`ontology/metadata/validate.py`),
replacing the old code allowlist:

- The class must be a defined `OntologyClass`; the relation type must
  be a defined `ObjectProperty`.
- Domain/range is checked against the endpoint's class (the node's
  `class` property, else its label) — **including subclasses** of the
  domain/range class (R1) — but only for endpoints that exist on
  Neo4j; a missing endpoint's side is skipped, not assumed valid.
- Data properties: a present value must match the declared datatype
  (`bool` is never accepted as `int`/`float`; an `int` is accepted
  for `float`); a `required` property that is missing is rejected; an
  unknown key is rejected — but only when the class already has at
  least one data property defined, so legacy data is not rejected for
  keys nobody defined yet.
- **v1 fallback**: when the store holds *no* `OntologyClass` node at
  all (metadata never seeded — bootstrap, and the older fake-store
  tests), validation falls back to `ENTITY_TYPES_V1` /
  `RELATIONSHIP_TYPES_V1` and skips datatype/domain checks. The moment
  at least one class node exists, metadata is authoritative.

System property keys (never treated as user data properties, never
datatype-checked): `id`, `status`, `version`, `class`, `base_ref`,
`derived`, `trace_id`.

### Reasoning — R1–R3 (group 6)

`ontology/reasoning/` — rules catalog (`rules.py`), materialization
(`materialize.py`). Only three rules in v1, enough to prove the
mechanism:

| Rule | Content |
| --- | --- |
| R1 Subclass | An instance of class C is also an instance of every ancestor of C. Query-time only (`subclass_ancestors` walks up, `subclass_descendants` walks down `SUBCLASS_OF`, both BFS with a `visited` set so cycles terminate). Class labels are never materialized on instance nodes. |
| R2 Inverse | `PART_OF` ↔ `HAS_COMPONENT`: writing one direction derives the other as a derived edge. |
| R3 Transitive | `PART_OF` is transitive: A → B and B → C derives A → C (both directions expanded at write time). |

**Mechanism: materialize at write time** (plain Cypher, no APOC/GDS/n10s
— `NEO4J_PLUGINS='[]'`):

- When a base edge is written, derived edges are `MERGE`d with
  `derived=true`, `rule="inverse:PART_OF"` (or `"transitive:PART_OF"`),
  `base_ref=<canonical ref of the base edge>` — the `MERGE` is keyed
  on `(source, type, target, base_ref)`, so re-running the same write
  is a no-op instead of duplicating the edge.
- Editing/deleting a base edge deletes its derived edges by `base_ref`
  (`remove_derived_for_base`).
- Adding a rule or editing metadata does not rewrite history by
  itself — run **recompute** (`POST /ontology/reasoning/recompute`,
  one entity or the whole graph): it deletes the derived edges in
  scope, then re-derives from the base edges.
- Derived edges live **only on Neo4j; L1 does not know them**. Neo4j
  is allowed to hold facts L1 lacks, as long as they are marked
  `derived` and traceable to a base edge via `base_ref`.

## Writing — dual write, Neo4j-first, fail-fast

All writes go through `OntologyWriteService`
(`ontology/canonical/write_service.py`): `write_entity(...)` /
`write_relationship(...)` / `ensure_entity(...)` (create-only-if-absent,
used for relation endpoints). Every write carries
`WriteProvenance(trace_id, source_event_id)`; `source_event_id` must be
**stable for the same logical write** — retries without one cannot
self-heal.

Order (design §5): **Neo4j first, L1 second**.

1. **Validate** against ontology metadata (§ above).
2. **Idempotency / self-heal** — look up L1 by `source_event_id`:
   - Same `source_event_id` as the stored row → **replay**: return the
     current object from Neo4j, write nothing.
   - L1 has no state but Neo4j already has the object (a previous
     write died after step 3) → **write the missing L1 state** and
     return. This is the self-heal for partial failure.
   - New properties identical to the Neo4j node → **noop**, version
     does not increase (this replaces the old `state_hash` check).
3. **Write Neo4j**: `MERGE` node/edge + `SET` all properties +
   `status` / `version` (mirrored with L1). Derive/remove derived
   edges per § above. Neo4j error → **fail immediately**: nothing is
   written to L1, nothing is queued (tran's decision 3).
4. **Write L1** (one transaction): upsert the state row (status,
   version+1, `trace_id`, `source_event_id`) + upsert the outbox row
   (state-sync intent). L1 error → the caller gets an error; the
   object is already on Neo4j, and a retry with the same
   `source_event_id` self-heals at step 2.
5. Return `(ref, version, change_type)` as before.

Version increases in both places and stays in sync; noop/replay do
not increase it. Concurrent duplicate inserts are retried once from a
`SAVEPOINT`. `force_update=True` bypasses the noop/replay
short-circuits — callers should not set it unless they mean "write a
new version even if nothing changed".

**Accepted consequences** (design §10): dual write has no
cross-system transaction, so a partial failure is always possible —
mitigated by idempotent `source_event_id` + self-heal + the drain
reconciler + alerts, never eliminated. Neo4j is a SPOF for object
reads/writes: when it is down, object reads/writes fail (state reads
from L1 still work, but there is no object to return). **Neo4j backup
is mandatory** — L1 can no longer rebuild Neo4j.

## Drain — from "projection" to state reconciler

Every write enqueues its subject in `ontology_projection_state`
(`status="pending"`). The drain no longer carries object payloads —
each row is a **state-sync intent**: "object (type, key) is at version
V, status S — make sure the Neo4j object matches that state".

- The drain claims pending rows oldest-first, converges each one, and
  marks it `synced` — conditionally (the mark only succeeds if
  `canonical_version` still equals the version read, so a write
  landing mid-drain is not silently dropped).
- L1 says active, Neo4j node exists, `status`/`version` differ →
  `SET` them to match.
- L1 says archived/deleted, Neo4j node still exists → delete the node
  (`DETACH DELETE`).
- L1 says active, Neo4j node is **gone** → it cannot be recreated
  (L1 has no object data). The row records `last_error`, goes dead,
  and **alerts** — this drift needs a human (restore from Neo4j
  backup). It is not retried forever.
- A row failing `PROJECTION_MAX_ATTEMPTS` times (default 5) is marked
  `dead` and never re-claimed. Re-enqueueing via
  `upsert_projection_pending` resets it to `pending` with `attempts=0`.

| Trigger | How |
| --- | --- |
| Celery beat | Task `ontology.projection_drain`, every `PROJECTION_DRAIN_INTERVAL_SECONDS` (default 60 s), batch `PROJECTION_DRAIN_BATCH_SIZE` (default 32). No catch-up. |
| CLI | `python -m scripts.project_ontology_queue [--limit N]` (default = batch size). Exits 1 when the drain reports errors. |
| HTTP | `POST /ontology/internal/project-drain` (same batch size). "Internal" means the same `X-API-Key`, no separate token. |

The drain builds the Neo4j store lazily and checks health first. When
Neo4j is unhealthy it returns
`{"processed": 0, "errors": 1, "dead": 0, "claimed": 0, "detail": "neo4j_unhealthy"}`
and projects nothing. Each drain result also reports `pending` /
`dead_total` (queue depth after the batch).

## Reading

- **Objects: from Neo4j.** `ontology/canonical/entity_read.py::get_entity`
  reads the node (`get_node`), returns `None` for a missing node and
  for an inactive `status` (the API answers 404). L1 is consulted
  only for state/idempotency/audit, never for object content.
- **State/version: from L1** — idempotency checks, state audit,
  version history (version + change_type + trace only).
- **Retrieval (hybrid, gazetteer, expansion, PPR)**: unchanged — it
  already read Neo4j + Qdrant. The gazetteer still reads live keys
  through `graph_query.list_entity_keys`.

## Read API

Base path `/ontology`, router `app/routers/ontology.py`. All endpoints
require the platform API key (`X-API-Key` header) unless `API_KEY` is
empty (dev only, auth disabled). `entity_type` must be a defined class
(v1 or metadata); otherwise **400**. A limit/depth outside its range
is **422**. Neo4j (or the canonical store) unreachable is **503**.
Restricting metadata writes to admin/service-key is still an open
design question (§12) — any API-key holder can currently edit metadata.

### Canonical reads + knowledge graph

| Endpoint | Description |
| --- | --- |
| `GET /ontology/entities/{type}/{key}` | Entity from Neo4j. **404** when missing or inactive. **503** `"store unavailable"` when the read fails. |
| `GET /ontology/entities/{type}/{key}/relationships` | Neighbours from Neo4j (depth 1). `?limit=` (default 50, 1–200). **503** `"graph unavailable"` when Neo4j is down. |
| `GET /ontology/entities/{type}/{key}/knowledge` | Documents linked via `RELATED_TO` / `DOCUMENTED_BY`. `?limit=` (default 50, 1–200). |
| `GET /ontology/graph?root_type=&root_key=&depth=&limit=` | Subgraph from Neo4j. `depth` default 2 (1–3), `limit` default 100 (1–500). |
| `POST /ontology/internal/project-drain` | Trigger one drain (batch = `PROJECTION_DRAIN_BATCH_SIZE`). Returns `{"processed", "errors", "dead", "claimed", "detail", "pending", "dead_total"}`. |

### Ontology Management (group 1) — metadata CRUD

`GET/POST /ontology/classes`, `GET/PATCH/DELETE /ontology/classes/{id}`,
`POST /ontology/classes/{id}/subclasses`,
`DELETE /ontology/classes/{id}/subclasses/{parent_id}`,
`GET/POST /ontology/object-properties`,
`GET/PATCH/DELETE /ontology/object-properties/{id}`,
`GET/POST /ontology/data-properties`,
`GET/PATCH/DELETE /ontology/data-properties/{id}`.
Create returns **201**. Delete of a class is refused while a subclass
points at it or an instance still carries the label; delete of an
object property is refused while any edge uses the type. Adding a
subclass that would close a cycle is **400**.

### Instance Management (group 3) — CRUD through the write service

| Endpoint | Description |
| --- | --- |
| `POST /ontology/entities` | Create an entity (dual write). Returns `{"ref", "version", "change_type", "applied", "changed", "replayed"}`. |
| `PATCH /ontology/entities/{type}/{key}` | Replace the entity's properties wholesale (an absent key is removed; `{}` clears all user properties). **404** when absent. |
| `DELETE /ontology/entities/{type}/{key}` | Soft-delete: L1 status → `deleted`, Neo4j node `DETACH DELETE`d. **404** when absent. |
| `POST /ontology/relationships` | Create a relationship (dual write + derived edges). |
| `DELETE /ontology/relationships?type=&source_type=&source_key=&target_type=&target_key=` | Soft-delete a relationship. **404** when the edge does not exist on Neo4j. |
| `POST /ontology/entities/{type}/{key}/validate` | Dry-run validation: runs the metadata validator for the would-be write, writes nothing. **400** on violation. |

### Graph Data Management (group 2)

| Endpoint | Description |
| --- | --- |
| `GET /ontology/schema` | Classes + object/data properties from Neo4j metadata; uniqueness constraints from `SHOW CONSTRAINTS`. Falls back to the v1 allowlists when no metadata was seeded, and to the default UNIQUE `(label, id)` constraint per v1 label when the constraints query fails or returns nothing. |
| `GET /ontology/graph/stats` | Node counts per label, edge counts per type, derived-edge count, totals. Labels/types come from metadata with the v1 fallback. |

```bash
curl http://localhost:8000/ontology/schema -H "X-API-Key: $API_KEY"
curl http://localhost:8000/ontology/graph/stats -H "X-API-Key: $API_KEY"
```

### Query and Retrieval (group 4)

| Endpoint | Description |
| --- | --- |
| `POST /ontology/query` | Arbitrary **read-only** Cypher (see below). |
| `GET /ontology/search?q=&limit=` | Entities whose `id` or `name` contains `q` (case-insensitive `CONTAINS`). `limit` default 50 (1–500). |
| `GET /ontology/entities?class=&include_subclasses=&limit=` | Instances of a class, **including subclasses (R1)** by default. `limit` default 100 (1–500). No conflict with `GET /ontology/entities/{type}/{key}` — the list path has no extra segments. |
| `GET /ontology/traverse?root_type=&root_key=&depth=&limit=&rel_types=` | Multi-hop traversal; `rel_types` may repeat (`?rel_types=A&rel_types=B`) or be comma-separated (`?rel_types=A,B`). The filter applies to **every hop** of the path. |

```bash
curl -X POST http://localhost:8000/ontology/query \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"cypher": "MATCH (n:Asset) RETURN n.id AS id, n.name AS name LIMIT 10"}'
curl "http://localhost:8000/ontology/search?q=lift" -H "X-API-Key: $API_KEY"
curl "http://localhost:8000/ontology/entities?class=Asset&include_subclasses=true" \
  -H "X-API-Key: $API_KEY"
curl "http://localhost:8000/ontology/traverse?root_type=Asset&root_key=O21&depth=2&rel_types=PART_OF" \
  -H "X-API-Key: $API_KEY"
```

**`POST /ontology/query` — read-only guard.** Body:
`{"cypher": str, "params": {}, "limit": 100, "timeout_seconds": null}`.
The guard (`graph_query._assert_read_only_cypher`) requires (a) the
first meaningful keyword to be one of `MATCH`, `WITH`, `RETURN`,
`CALL`, `UNWIND`, `OPTIONAL`, and (b) no banned keyword as a
standalone word anywhere in the query (after stripping `--` and
`/* */` comments): `MERGE`, `CREATE`, `DELETE`, `DETACH`, `SET`,
`REMOVE`, `DROP`, `LOAD` — plus no `CALL apoc.*`. Violation → **400**,
before the query reaches Neo4j. `limit` is clamped to 1..500 and
applied to the rows returned (it is *not* appended to the Cypher —
appending would break queries that already end with `LIMIT` or `;`).
Response: `{"columns": [...], "rows": [...]}` — `columns` are the
keys of the first row, or `[]` when there are no rows.

**Timeout.** `timeout_seconds` (per query, default 30 s, clamped
1..120 s) is passed to the Neo4j 5 driver as
`session.run(..., timeout=timeout)` — the server terminates a
transaction that runs longer. `Neo4jGraphStore.run_query` accepts a
`timeout=` keyword; if the driver in use does not accept it, the
query is retried once without it (and a warning is logged) — the
timeout is a guard, never a reason to fail a read. Note this was
verified only against the driver's documented API, not against a
live Neo4j in this workspace (the `neo4j` package is not installed
in the dev venv). For a server-side default that applies to every
transaction, set `db.transaction.timeout` in `neo4j.conf` instead.

**Search — no full-text index (deliberate, design §12 open
question).** `search_entities` is a label scan with
`toLower(...) CONTAINS toLower($query)` over `id` and `name`,
excluding ontology metadata nodes. That is fine at demo scale; if
the graph grows, revisit with a Neo4j full-text index (or gazetteer
prefix search) — the decision is still open, not silently settled.

### Reasoning (group 6)

| Endpoint | Description |
| --- | --- |
| `GET /ontology/reasoning/rules` | The v1 rule catalog: R1 subclass, R2 inverse, R3 transitive. |
| `GET /ontology/reasoning/derived?entity_type=&entity_key=&limit=` | Derived edges (`derived=true`). Whole graph when no entity is given; when `entity_key` is given, only edges touching that entity (`entity_type`, when also given, must be its label). Each row carries `rule` and `base_ref`. |
| `POST /ontology/reasoning/recompute` | Re-run materialization: `{"entity_type", "entity_key"}` for one entity, or `{"all": true}` for the whole graph. Exactly one scope, else **400**. Returns `{"removed", "created", "scope"}`. Run after adding a rule or editing metadata. |

```bash
curl http://localhost:8000/ontology/reasoning/rules -H "X-API-Key: $API_KEY"
curl "http://localhost:8000/ontology/reasoning/derived?entity_key=O21" \
  -H "X-API-Key: $API_KEY"
curl -X POST http://localhost:8000/ontology/reasoning/recompute \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"all": true}'
```

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
FalkorDB backfill script below, never by `app/config.py`.

## How to run Neo4j + seed + drain

```bash
# Start the stack (Neo4j is a compose service with a healthcheck):
docker compose up -d neo4j
# or start everything: docker compose up -d

# Apply migrations (creates the ontology_* tables, head 0015):
make migrate

# Seed the demo ontology into Neo4j + L1 via the write service
# (metadata nodes first — 4 classes + 5 object properties + data
# properties — then 17 entities + 15 relations: O21/O22/O23/O24,
# O20 archived, 2 buildings, 6 components, 3 Document nodes):
make seed
# or ontology only: python -m seeds.seed_ontology_demo

# Drain the queue by hand (beat does this every 60 s anyway):
python -m scripts.project_ontology_queue
# or via the API:
curl -X POST http://localhost:8000/ontology/internal/project-drain \
  -H "X-API-Key: $API_KEY"

# Opt-in edge-case seed (NOT in `make seed`): plant one poisoned
# projection row; drain 5 times and it goes dead:
make seed-edge
python -m scripts.project_ontology_queue  # repeat 5x, or wait for 5 beats
```

## Migration data — L1 → Neo4j backfill + migration 0015

This already ran once for the redesign cutover (migration `0015`
is the current head). The sequence below is the record of that
cutover, and the procedure to repeat if the data ever has to be
rebuilt from an old L1 backup. **Do not run it against live data
without a backup.**

1. **Backup first**: `pg_dump` the Postgres database, and back up
   Neo4j (mandatory now — L1 can no longer rebuild it).
2. **Seed**: `make seed` — writes metadata nodes first, then
   instances, through the dual-write service.
3. **Backfill** (only when L1 still holds old `properties` data):
   `scripts/backfill_l1_to_neo4j.py` reads every `properties` value
   from `ontology_entities` / `ontology_relationships` on L1 and
   `MERGE`s it onto the matching Neo4j node/edge (`SET` properties +
   `status` + `version`).
   ```bash
   # Dry run first — counts what would be written, writes nothing:
   .venv/bin/python -m scripts.backfill_l1_to_neo4j --dry-run
   # Real run:
   .venv/bin/python -m scripts.backfill_l1_to_neo4j
   ```
4. **Verify**: `--verify` compares Neo4j against L1 without writing
   — counts active nodes/edges vs active L1 rows, and spot-checks
   the properties of a few sample entities (default 3).
   ```bash
   .venv/bin/python -m scripts.backfill_l1_to_neo4j --verify
   ```
   Spot-check a couple of entities by hand (O21, O22, …) — ids,
   properties and `status`/`version` must match L1.
5. **Migration 0015** — only after backfill + verify:
   `alembic upgrade head` (drops `properties` + `state_hash` on both
   tables, drops `ontology_versions`). Destructive, no automatic
   rollback — `downgrade()` restores the schema only, the dropped
   object data is **not** recovered.
6. **Drain**: `python -m scripts.project_ontology_queue` (or the
   internal endpoint) until `pending` is 0 and `dead_total` is 0.
   A dead row after the cutover means L1 says active but Neo4j has
   no node — restore that object from the Neo4j backup, do not
   re-enqueue blindly.

### Backfill — FalkorDB → canonical (legacy, one-off)

Script: `scripts/backfill_falkor_to_canonical.py`. It reads **all**
nodes and relationships from a FalkorDB graph and writes them through
`OntologyWriteService` into the canonical tables; the dual write
carries them into Neo4j.

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

### Backfill fixture — test the script without production data

`seeds/backfill_fixture.json` is a small legacy FalkorDB graph that is
NOT in canonical: asset O19 (freight elevator) + building B1 +
component motor-M9 + a Document node, with PART_OF / HAS_COMPONENT /
DOCUMENTED_BY edges. Load it into a FalkorDB instance, then run the
backfill against that graph and verify O19 appears in canonical +
Neo4j.

No compose service is shipped for FalkorDB (it exists only for this
one-off migration); run it externally:

```bash
docker run -d -p 6379:6379 falkordb/falkordb:latest

# Load the fixture (idempotent MERGE; env FALKOR_HOST/FALKOR_PORT):
python -m scripts.load_backfill_fixture
# or dry-run (validate only, writes nothing):
python -m scripts.load_backfill_fixture --dry-run

# Backfill the fixture graph (not knowledge_graph):
.venv/bin/python -m scripts.backfill_falkor_to_canonical \
  --falkor-host localhost --falkor-port 6379 \
  --falkor-graph backfill_fixture --dry-run
.venv/bin/python -m scripts.backfill_falkor_to_canonical \
  --falkor-host localhost --falkor-port 6379 \
  --falkor-graph backfill_fixture
```

## Verify live on tran's machine (Neo4j + Postgres real)

Run after pulling, with the real stack up (`docker compose up -d`,
`alembic upgrade head` → expect `0015_ontology_state_only`,
`make seed`, worker restarted):

```bash
# 1. Head + both stores reachable:
.venv/bin/python -m pytest tests/test_alembic_migrations.py -q
curl http://localhost:8000/ontology/graph/stats -H "X-API-Key: $API_KEY"
#    expect nodes_by_label Asset 4+ / Building 2 / Component 6 /
#    Document 3 (demo seed), derived_edges >= 1

# 2. Schema reads metadata, not the fallback:
curl http://localhost:8000/ontology/schema -H "X-API-Key: $API_KEY"
#    classes must list Asset/Building/Component/Document with
#    descriptions, object_properties must carry domain/range
#    (PART_OF: Asset -> Building); constraints must list the real
#    Neo4j constraints, not only the unique_*_id defaults

# 3. Query API against real Neo4j:
curl -X POST http://localhost:8000/ontology/query \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"cypher": "MATCH (n:Asset) RETURN n.id AS id LIMIT 5"}'
curl -X POST http://localhost:8000/ontology/query \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"cypher": "MATCH (n) SET n.x = 1 RETURN n"}'   # expect 400
curl "http://localhost:8000/ontology/search?q=lift" -H "X-API-Key: $API_KEY"
curl "http://localhost:8000/ontology/entities?class=Component" \
  -H "X-API-Key: $API_KEY"
curl "http://localhost:8000/ontology/traverse?root_type=Asset&root_key=O21&depth=2" \
  -H "X-API-Key: $API_KEY"

# 4. Reasoning against real Neo4j:
curl http://localhost:8000/ontology/reasoning/rules -H "X-API-Key: $API_KEY"
curl "http://localhost:8000/ontology/reasoning/derived?entity_key=O21" \
  -H "X-API-Key: $API_KEY"   # expect >= 1 inverse edge
curl -X POST http://localhost:8000/ontology/reasoning/recompute \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"entity_type": "Asset", "entity_key": "O21"}'
#    then re-check /reasoning/derived — same edges, no duplicates

# 5. Drain converged, nothing dead:
curl -X POST http://localhost:8000/ontology/internal/project-drain \
  -H "X-API-Key: $API_KEY"   # expect pending 0, dead_total 0

# 6. Subclass query (R1) live: create a subclass via the API, seed an
#    instance of it, and confirm GET /ontology/entities?class=<parent>
#    returns it while include_subclasses=false does not.
```

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
- **Search full-text index**: open question (design §12) — see
  "Query and Retrieval" above. Not decided, not built.
- **Metadata write permissions**: open question (design §12) — whether
  editing classes/properties should require admin/service-key. Any
  API-key holder can currently edit metadata.
