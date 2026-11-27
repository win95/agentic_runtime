# Adding an agent (declarative harness)

Each mission declares runtime policy in a **`harness:`** block (merged with
`runtime/harness/defaults.yaml` and validated at seed time).

Configure **`HARNESS_SOURCE`** (`db` | `seed`, default `db` in production).

| Policy | Source of truth |
|--------|-----------------|
| RBAC (`user_runnable`) — admin path only | The global harness index (`_get_index`, TTL 30s) — **`HARNESS_SOURCE=db`:** `missions.execution_config.harness` for rows in Postgres; per-row fallback to that mission's seed YAML only when the `harness` key is missing; invalid DB harness → default fallback (see "Failure semantics" below — NOT uniformly closed: `builtin_tools.deny` stays false, so built-ins stay allowed). **Empty table → no missions in index** (no RBAC grants). **`HARNESS_SOURCE=seed`:** all `seeds/missions/*.yaml`. The index is NO LONGER on the execution hot path (see "Load path" below). |
| Guest `/guest/chat` routing | A light scan (`_scan_guest_routing`): reads only the raw `guest_channel.exposes_public_chat` key per mission — db mode one `SELECT (id, execution_config)`, seed mode the seed YAMLs; a mission whose raw flags say anything is validated with one lenient parse, so a semantically invalid harness cannot route. Exactly one chat mission or `GuestChatMisconfiguredError`. |
| Per-execution deny / flow / budget / sandbox flags during a run | `load_mission_harness(mission_id)` — one load path, parsed once, cached per mission (see "Load path" below). The run's snapshot (`execution_config.harness`, same as DB after seed) is parsed instead when the execution carries it. |
| Guest tool registration (`reply_to_guest`, `confirm_booking_create`) | **One function per `HARNESS_SOURCE`** (`guest_output_capabilities()` in `runtime/harness/registry.py`): db mode reads the raw `guest_output` keys from DB rows; seed mode reads the seed YAMLs. No more two hardcoded paths. |
| No AgentVersion system prompt | The agent class's own fallback: `BaseAgent.fallback_system_prompt` reads the seed YAML; `AdminAgent` returns its own text (no `system_prompt_fallback` Literal anymore) |

## Load path (Plan B)

One load path: `load_mission_harness(mission_id)` in
`runtime/harness/registry.py`. `effective_harness(mission_id,
execution_config)` is a thin accessor over it.

- **Per mission, cached.** db mode reads the mission's DB row (one
  indexed SELECT); seed mode reads that mission's seed file only. The
  cache key is `(updated_at,)` (db — `onupdate` bumps it on any row
  change, so a re-seed is visible as soon as the row changes) or
  `(seed file mtime_ns, harness version)` (seed), plus the short TTL
  (`harness_index_ttl_seconds`, 30s) as a safety bound.
- **Parse once per run.** The mission graph resolves the harness once
  (in the deep-agent node) and passes the model down — `context["harness"]`,
  `HookMiddleware`/`HookContext`, and the `harness=` kwarg on
  `harness_requires_output_files` / `mission_denies_builtin_tools` /
  `builtin_deny_message` / `use_flow_subgraph_harness` /
  `effective_history_tokens` / `builtin_tools_exclusion` /
  `allowed_builtin_tool_names` / `use_flow_subgraph`. Call sites
  without the model (post-run notify in `app/worker/tasks.py`,
  sandbox service, artifact sink) still call `effective_harness`,
  which is cheap: the mission load is cached and snapshot parses are
  memoized by content hash, so the same snapshot dict parses once per
  process.
- **Snapshot semantics unchanged.** When the execution carries its
  own `execution_config.harness`, that dict is what is parsed
  (memoized); an invalid snapshot falls back to the mission load, not
  straight to default.
- **Guest routing / caps are NOT the index.** They come from the light
  raw scan (see the table above). The global index only serves admin
  lookups (`user_runnable_mission_ids`) and `GET /health`
  (`harness_index_degraded`).
- **No allowlist.** There is no `HARNESS_MISSION_ALLOWLIST` anymore:
  a split worker loads only its own mission through
  `load_mission_harness`, so no index filter is needed (see
  `docs/worker-per-agent.md`).

## Harness version & compatibility

`MissionHarness.version` is the schema version this code understands
(current: **1**). It travels YAML → DB through `model_dump` at seed
time, so a stored row records which schema wrote it. Rows / YAML
without `version` read as 1 (backward compatible — no migration).

| Where | Policy |
|-------|--------|
| Seed / load time (`make seed`, `load_mission_file`, `agent_config_check`) | **Strict**: unknown field or unknown version fails fast with an explicit error (`Unsupported harness version 2 ...`). A typo or a schema the code does not know must be caught before it reaches the DB. |
| Runtime read (`effective_harness`, per-mission load) | **Lenient**: unknown fields are dropped (warned about once per process per dotted path, e.g. `notifications.future_flag`); unknown versions are warned about once, then the known version-1 fields are read. A newer row must never kill a run on older code. Genuine semantic errors in known fields (deny without its message template, negative budget, bad `sink` value) still fail → fallback, below. |

When adding a harness field: bump nothing unless the meaning of a
known field changes — a new optional field keeps `version: 1`. Bump
`version` only when old code cannot read the row correctly anymore,
and update this section with what changed.

## Failure semantics (runtime)

Every runtime fallback to the default goes through
`fallback_harness(mission_id, reason)` — it logs the reason + mission
at warning level and returns the default harness. There is no silent
default anymore. (The one exception is the `execution_config` path,
which falls back to the mission load instead — see the table.)

| Path | Behaviour | Log |
|------|-----------|-----|
| Seed file invalid (`iter_seed_mission_harnesses`) | File skipped from the index / seed-mode load (mission loses its harness); other files unaffected | warning, naming the file + reason |
| DB index row invalid (`_load_index_from_db`) | Lenient parse; semantic error → default fallback for that row only | warning via `fallback_harness` |
| DB row has no `harness` key and no seed fallback | Default fallback for that row | warning via `fallback_harness` |
| `execution_config.harness` invalid during a run | Warn, then fall back to the mission load for that mission | warning (falls back to mission load, not straight to default) |
| Mission missing (no DB row / no seed file) / no mission id | Default fallback | warning via `fallback_harness` |
| Guest scan entry with raw guest flags but invalid harness | Entry ignored (no routing, no caps) — the old index build would have fallen back to the closed default | warning via `_guest_flags_validated` |

Default fallback value = `MissionHarness()` unchanged, and it is
deliberately NOT uniformly closed — do not assume "deny everything":

- closed: `security.user_runnable` false (JWT users cannot run the
  mission), `guest_channel.exposes_public_chat` false (no guest
  routing), `flow_subgraph.enabled` false (structured flow off),
  `guest_output.register_reply_tools` false, both `notifications`
  flags false, `output_files.required` false.
- OPEN: `builtin_tools.deny` false — deepagents built-ins stay
  allowed under the fallback. Changing any fallback flag needs an
  explicit decision, not a code tweak here.

`GET /health` reports `harness_index_degraded` when the DB index refresh failed
and a stale snapshot is served.

**`security.user_runnable` defaults to false.**

Docker images copy `seeds/` and `skills/` at build time — rebuild the image
(or mount the repo) after adding scaffolded files, then run `make seed`.

## Harness fields (mission YAML)

| Block | Purpose |
|-------|---------|
| `version` | Harness schema version this code understands (default `1`; stamped into DB rows at seed; see "Harness version & compatibility") |
| `security.user_runnable` | JWT `user` role may `POST /executions` for this mission |
| `guest_channel.exposes_public_chat` | Route `POST /guest/chat` here (exactly one mission) |
| `builtin_tools.deny` | Hide/deny deepagents built-ins (hotel) |
| `builtin_tools.deny_message_template` | Hook message (`{tool_name}` placeholder only) when deny is true |
| `flow_subgraph.enabled` | Use the structured flow router when skills are mapped |
| `guest_output.register_reply_tools` | Register `reply_to_guest` / booking confirmation tools (seed-time only) |
| `guest_output.capability` | Registry capability for those synthetic tools |
| `notifications.notify_admin_actions` | SUCCEEDED runs of this mission notify Slack when output has a non-empty `actions_taken` (admin actions; failures notify for every mission) |
| `context_budget.reclaim_retrieval_for_history` | Opt-in: move unused retrieval slice into history trim budget |
| `output_files.required` | Opt-in: `stop_completion` nudges the agent to write top-level report files under `/output` in the live sandbox before stopping (same `list_directory` + run start mtime logic as file export) |
| `output_artifacts.sink` | `none` (default) or `minio`: after the run, the worker uploads `output.files[]` to MinIO when the mission `output_schema` declares `files` (`minio` requires that field). `none` leaves inline content in the result row |
| `output_artifacts.strip_inline` | When `sink: minio`, drop each file's `content` after a successful upload (default true) |
| `output_artifacts.inline_fallback` | When upload fails, keep inline `content` and set `storage: inline` on that entry (default true); run status is unchanged |
| `sandbox.command_timeout_seconds` | Default per-command timeout when the model omits one (default `90`) |
| `sandbox.command_timeout_max_seconds` | Ceiling for model-requested timeouts (default `600`); must be ≥ `command_timeout_seconds` |
| `tool_calls.sequential` | Tool names that must not run in parallel within one model turn; later parallel calls get a deferral `ToolMessage` (enforced in middleware, not via model bind) |
| `tool_budget.tools` | Tool names subject to the shared execution result budget |
| `tool_budget.max_result_tokens` | Max estimated tokens of tool results charged per execution (main + subagents) |
| `tool_budget.max_calls` | Per-tool call caps for the execution (e.g. `tavily_search: 4`) |

Agent fallback prompt (no AgentVersion): the agent class declares it
— `BaseAgent.fallback_system_prompt` reads the agent's seed YAML
(`system_prompt`); `AdminAgent` returns its own text without a file.
There is no `system_prompt_fallback` Literal anymore.

## Add a new agent (after)

1. `make new-agent NAME=my_agent` — agent + mission + `skills/.../SKILL.md`
2. Edit the new YAML: `allowed_tools`, objective, harness overrides if needed
3. `make seed-agent AGENT=my_agent MISSION=my_agent_mission`
4. Optional: map MCP tools in `.env` `MCP_TOOL_CAPABILITIES`
5. `python scripts/manual/agent_config_check.py --static-only --mission …`
   (harness + seed/DB drift, no live LLM)

**Five steps**, one Python package touched only when you add genuinely new
runtime behavior.

## Before this refactor (rough estimate)

To add an agent you had to discover and edit many hardcoded touchpoints (~12–15
edits across Python, YAML, env, and docs).

## Still code-keyed (not in `harness:` yet)

| Concern | Location |
|---------|----------|
| Capability subgraph (`fault_analysis` only) | `runtime/mission_graph.py` |
| Admin Slack notifications | `runtime/notifications/slack.py` |
| Tool → capability registration (fault/admin/MCP) | `runtime/tools/*`, `MCP_TOOL_CAPABILITIES` |
| Manual / eval script mission constants | `scripts/manual/*.py`, etc. |