# Manual API test scripts

Small stdlib scripts used to drive the running stack through the real
API (not mock LLM, not the eval harness). Each `POST /executions` is a
**new execution / new conversation** — the agents keep no memory
between calls, so name exact ids in the request text.

## Prerequisites

- Docker backing services up (`postgres`, `redis`, `qdrant`, `falkordb`, `minio`, `mlflow`).
- API on `http://localhost:8000` and a Celery worker consuming the queue.
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
```

## Notes from the manual test session (2026-11-10)

- Hybrid `fault_analysis` never calls the LLM: it stops in the
  deterministic capability graph. Use `fault_autonomous.py` (or
  `make eval-real` for the eval harness) to exercise the agent loop.
- `scripts/eval_mission.py` sets `mission["execution_mode"]`, but
  `run_mission` reads `mission["execution"]["mode"]` — so all classic
  eval cases currently run hybrid, including the one named `autonomous`.
- Autonomous `fault_analysis` ends `FAILED` with
  `Result missing output-schema keys: ['root_causes', 'recommendations']`:
  the Deep Agent outcome fills answer/summary/actions_taken/evidence
  only. Product decision pending.
- Capability-only executions keep `agent_version = NULL` (Option C
  version resolution happens in `deep_agent_node`), so the API cannot
  say which agent config ran for a hybrid run that never escalated.
- `delete_mission` refuses when the mission has executions or schedules
  (`deleted: false, refused: true`) — the API returns 409 in that case.
- Cancelling an already-finished execution keeps its status
  (`cancelled: false`); small models may still claim "successfully
  cancelled" — check `actions_taken` and the DB status, not the prose.
