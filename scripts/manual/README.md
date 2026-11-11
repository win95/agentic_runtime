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
