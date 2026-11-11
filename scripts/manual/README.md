# Manual API test scripts

Small scripts used to drive the running stack through the real API (not
mock LLM, not the eval harness). Each `POST /executions` is a **new
execution / new conversation** — the agents keep no memory between calls,
so name exact ids in the request text.

## Prerequisites

- Docker backing services up (`postgres`, `redis`, `qdrant`, `neo4j`, `minio`, `mlflow`).
- API on `http://localhost:8000` and a Celery worker consuming the queue.
- Only **one** worker should consume the `celery` queue. A leftover
  docker-compose `worker` container runs the image's old code (it fails
  today with `KeyError: app.worker.tasks.run_due_schedules` and old
  FalkorDB settings) and steals executions from the local worker. Stop it
  with `docker compose stop worker` before manual runs.
- Run from the repo root with the project venv: `.venv/bin/python ...`
- `API_KEY` and `API_BASE_URL` are read from the repo-root `.env`
  (existing environment variables win; `API_BASE_URL` defaults to
  `http://localhost:8000`).

## Scripts

| Script | What it does |
|---|---|
| `admin_bot.py` | Sends requests to the `system_admin` mission (admin_bot): health, list, create/update/delete mission, dispatch/cancel. `--suite` runs the canned read/write/refusal/injection cases against the throwaway mission `demo_admin_test`. `--dispatch-then-cancel` dispatches a long child execution and has the bot cancel it. |
| `fault_investigator.py` | Runs the seeded `fault_analysis` mission (agent `fault_investigator`, hybrid): O21 full month, first week, Layer 1 validation failures (expect HTTP 422), unknown asset. |
| `fault_autonomous.py` | Creates a throwaway mission with `mode=autonomous` so the Deep Agent/LLM path actually runs, then deletes its executions + mission rows from Postgres (`--keep` to leave them). |
| `skill_map_check.py` | Confirms the skill-map feature through the real LLM. `map` case: throwaway autonomous mission with `skills=["data-access"]`; validates `plan_trace["data-access"]` against `skills/data-access/SKILL.md` (visited/skipped partition the nodes, tools belong to their node, deviations are outside the map, every successful action is traced). `mapless` case: `skills=["fault-investigation"]` must produce **no** `plan_trace`. `--strict` also requires the agent to visit every tool-bearing node in map order with no deviations. Cleans up unless `--keep`. |
| `agent_turns.py` | Proves the agent loop runs **multiple model/tool turns inside one execution** (`system_admin`). Reads `llm_usage` + `audit_logs` from Postgres to print a timeline (model turn → its tools → next model turn) and checks: SUCCEEDED, `--min-turns` model calls (default 2), `--min-tools` tool calls (default 2), every audit row terminal. `--strict` also requires tools in ≥ 2 different model turns. `--request` overrides the admin request; `--cleanup` deletes the run's execution/result/usage/audit rows. |

`_api.py` is the shared helper (`.env` loading, JSON calls, poll-until-terminal).

## Examples

```bash
# Admin bot: one request
.venv/bin/python scripts/manual/admin_bot.py "Kiểm tra sức khỏe hệ thống"

# Admin bot: full read/write/refusal suite
.venv/bin/python scripts/manual/admin_bot.py --suite

# Admin bot: real dispatch + cancel
.venv/bin/python scripts/manual/admin_bot.py --dispatch-then-cancel

# Fault investigator: all hybrid cases (happy, week, invalid, unknown)
.venv/bin/python scripts/manual/fault_investigator.py --case all

# Fault investigator: only the validation rejects
.venv/bin/python scripts/manual/fault_investigator.py --case invalid

# Real LLM path (temporary autonomous mission, auto-cleanup)
.venv/bin/python scripts/manual/fault_autonomous.py

# Skill maps: plan_trace contract + a mapless skill (real LLM, auto-cleanup)
.venv/bin/python scripts/manual/skill_map_check.py

# Same, but exit 1 unless the agent follows the plan (all tool nodes, in order)
.venv/bin/python scripts/manual/skill_map_check.py --case map --strict

# Multi-turn: prove the agent loop (2+ model calls, 2+ tools) in one execution
.venv/bin/python scripts/manual/agent_turns.py

# Multi-turn: stricter thresholds, custom request, delete the run afterwards
.venv/bin/python scripts/manual/agent_turns.py --min-turns 3 --min-tools 4 --cleanup
.venv/bin/python scripts/manual/agent_turns.py --request "Kiểm tra sức khỏe hệ thống"
```

## Notes from the manual test session (2026-11-10)

- Hybrid `fault_analysis` never calls the LLM: it stops in the
  deterministic capability graph. Use `fault_autonomous.py` (or
  `make eval-real` for the eval harness) to exercise the agent loop.
- `scripts/eval_mission.py` sets `mission["execution_mode"]`, but
  `run_mission` reads `mission["execution"]["mode"]` — so all classic
  eval cases currently run hybrid, including the one named `autonomous`.
- Autonomous `fault_analysis` now ends SUCCEEDED: `deep_agent_node`
  fills the declared output-schema keys from the agent outcome (Option A,
  2026-11-11). Earlier it failed with
  `Result missing output-schema keys: ['root_causes', 'recommendations']`.
- Capability-only executions keep `agent_version = NULL` (Option C
  version resolution happens in `deep_agent_node`), so the API cannot
  say which agent config ran for a hybrid run that never escalated.
- `delete_mission` refuses when the mission has executions or schedules
  (`deleted: false, refused: true`) — the API returns 409 in that case.
- Cancelling an already-finished execution keeps its status
  (`cancelled: false`); small models may still claim "successfully
  cancelled" — check `actions_taken` and the DB status, not the prose.

## Notes from the skill-map check (2026-11-11)

- `skill_map_check.py` separates two questions. **Structural** checks
  (always enforced) prove the trace is a faithful function of the run's
  tool calls: `visited`/`skipped` partition the map nodes, `tools_by_node`
  only names each node's declared tools, deviations are tools outside the
  map, and every successful `actions_taken` entry appears in the trace.
  **Conformance** (reported, enforced only with `--strict`) asks whether
  the agent actually followed the map.
- `identify` and `synthesize` declare no tools, so they can never appear
  in `visited` — the trace is computed from tool calls. They are
  reasoning-only steps and always show under `skipped`; that is by design,
  not evidence that the agent skipped them.
- With the local `qwen2.5-3b` model the pilot run called all five
  tool-bearing `data-access` nodes (`search_fault_history`,
  `get_maintenance_records`, `get_sensor_data`,
  `query_asset_relationships`, `search_technical_documents`) but started
  at maintenance instead of faults and also called
  `calculate_failure_rate` (a deviation). So the structural contract
  passes while `--strict` fails on purpose — this is a model-quality
  signal, not a feature bug.
- The mapless case must SUCCEED with no `plan_trace` key: the map-less
  `fault-investigation` skill behaves exactly as before the feature.
- Hybrid `fault_analysis` never reaches the Deep Agent with the seed data
  (the capability graph already yields enough evidence), so both cases
  here create `mode=autonomous` missions; the map is exercised only on
  the autonomous path.

## Notes from the multi-turn check (2026-11-11)

- There is **no conversation memory across executions** (README §8), and
  no chat/thread API: a new `POST /executions` is a new conversation. So
  `agent_turns.py` measures the agent loop inside one execution, using
  the middleware rows as evidence.
- `llm_usage.created_at` is written at `on_llm_end` (model call finished)
  and `audit_logs.created_at` when the tool starts, so sorting both gives
  `model turn 1 -> its tools -> model turn 2 -> ...`. A tool row belongs
  to the latest model call at or before it.
- With the local `qwen2.5-3b` model, a request needing four admin tools
  reliably produces **2 model calls** and **4 tool calls**: the model
  emits all four tool calls in one assistant message, then answers from
  their results. That passes the default checks, but `--strict` fails
  with `only 1 tool-bearing turn` because the tools were not split over
  two turns. This is a model-behaviour signal, not a runtime bug — do not
  treat `--strict` as the default gate.
- The final answer can still hallucinate details the tools never returned
  (e.g. inventing an `execution_id`), so check `actions_taken` and the
  audit rows, not the prose. A request that explicitly says "call one
  tool at a time" does not change qwen's batching.
- `--cleanup` removes `ExecutionResult`, `LLMUsage`, `AuditLog`, then the
  `Execution`; without it the run stays in the admin history like any
  other `system_admin` execution.
