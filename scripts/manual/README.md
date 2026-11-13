# Manual API test scripts

Small scripts used to drive the running stack through the real API (not
mock LLM, not the eval harness). By default each `POST /executions` is a
**new execution / new conversation** — agents keep no memory between
calls, so name exact ids in the request text. Passing a shared
`session_id` is the exception: those executions form one conversation
and later runs see the prior turns (`docs/sessions.md`).

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
| `session_conversation.py` | Drives a real **conversation across executions** using Option B sessions (`docs/sessions.md`): 3 turns share one `session_id`; the bot must recall a random code it was given in turn 1, then carry it across 2 prior turns; `GET /sessions/{id}` must list the executions oldest first. `--control` asks the same question in a fresh session (must not know the code). `--cleanup` deletes the session's rows. |
| `agent_config_check.py` | Asserts that runs use the config pinned in the active `AgentVersion`. Static: seed YAML ↔ `agents` row ↔ version snapshot, tool-manifest drift, `allowed_tools` in the registry, mission skills/hooks resolve, session limits parse. Live (2 turns, one fresh session): both runs record the version + no drift, `llm_usage` model matches, and the MLflow trace shows the system prompt **starts with** the pinned text, the pinned model/temperature, all `allowed_tools` at the model call (no registry tool leaks), the selected skills, and turn 2's session-history header with the agent's limits. `--no-trace` skips the MLflow assertions; `--cleanup` deletes the runs. |
| `hotel_reception.py` | Drives the `hotel_receptionist` agent end to end through the real API → worker → LLM → MCP client → mock PMS server. **Preflight** checks the pinned 14-tool contract, the `MCP_TOOL_CAPABILITIES` map, the server's tool list, and the capability gate. Scenarios: `availability`, `soldout`, `policy`, `task`, `booking` (confirmation gate: best-effort with the 3B model, direct `get_booking` cross-check, booking-code recall turn), `memory` (3-turn recall + `GET /sessions/{id}`), `journey` (full booking lifecycle incl. change dates + re-check), `task_full` (housekeeping task + status), `cancel` (free + one-night-fee cancellation), `multi_request` (availability → policy → booking in one session), `soldout_switch` (sold-out recovery to Deluxe). Graders are outcome-based (architecture Phase 1): state verifiers + semantic checks in `verifiers.py`; trajectory findings are warnings only. `--scenario` is repeatable; `--session-id`, `--timeout`, `--cleanup` as usual. |
| `verifiers.py` | Pure outcome graders for the hotel suite (no live stack needed): state verifiers (no booking created for a sold-out window, task row correct), semantic checks (unavailability + policy facts via small paraphrase sets — interim until the LLM judge model is chosen), scenario verdicts (state AND semantic), trajectory diagnostics. Unit tests in `tests/test_manual_verifiers.py`, including the do-nothing (must fail every verdict) and gold (must pass all) negative controls. |
| `mlflow_report.py` | Read-only analysis of the runs `hotel_reception.py` left in MLflow: discovers the experiment, joins runs↔traces on `mlflow.sourceRun`, classifies each trace (scenario, turn index, status/version/drift, latency, tokens, contract vs built-in tools, hook failures, derailed prose), then writes a Markdown report + one-row-per-trace CSV. No API/worker/MCP needed; re-run after any test run or DB cleanup. Defaults: experiment `agentic-runtime`, session prefix `sess-hotel-`, outputs in `reports/`. |

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

# Conversation across executions (Option B session, 3 turns + isolation control)
.venv/bin/python scripts/manual/session_conversation.py --control --cleanup

# Config check: seeds → agents row → version → the prompt/model/tools the LLM saw
.venv/bin/python scripts/manual/agent_config_check.py
.venv/bin/python scripts/manual/agent_config_check.py --cleanup
.venv/bin/python scripts/manual/agent_config_check.py --no-trace   # no MLflow

# Hotel receptionist: full preflight + all 6 scenarios (needs the mock MCP server)
.venv/bin/python scripts/manual/hotel_reception.py

# Hotel receptionist: one scenario, then clean its rows
.venv/bin/python scripts/manual/hotel_reception.py --scenario booking --cleanup

# Hotel receptionist: pick scenarios (repeatable), custom session base, longer timeout
.venv/bin/python scripts/manual/hotel_reception.py \
    --scenario availability --scenario policy --timeout 600
.venv/bin/python scripts/manual/hotel_reception.py --session-id sess-hotel-demo

# MLflow report: regenerate from everything the hotel runs left behind
.venv/bin/python scripts/manual/mlflow_report.py
make mlflow-report

# MLflow report: one day only, or a different experiment / session prefix
.venv/bin/python scripts/manual/mlflow_report.py --since 2026-11-12
.venv/bin/python scripts/manual/mlflow_report.py \
    --experiment agentic-runtime --session-prefix sess-hotel-

# MLflow report: print to stdout, no files
.venv/bin/python scripts/manual/mlflow_report.py --out "" --csv ""
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

- `agent_turns.py` measures the agent loop **inside one execution**
  using the middleware rows as evidence. Conversation memory **across**
  executions is now Option B sessions (`docs/sessions.md`): executions
  sharing a `session_id` are one conversation and later runs are shown
  the prior turns; without a `session_id` a run is still its own
  one-turn session, exactly as before. Use `session_conversation.py`
  to exercise that.
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

## Notes from the session conversation check (2026-11-11)

- `session_conversation.py` proves the session memory end to end: turn 2
  answers with the random code from turn 1, turn 3 carries it across two
  prior turns, and `GET /sessions/{id}` lists the three executions oldest
  first with `agent_id=admin_bot`, `mission_id=system_admin`.
- The `--control` run asks the same recall question in a fresh session;
  it must not know the code. That is the isolation check — without
  `session_id` there is no history to read.
- Each turn names a tool on purpose. `system_admin` declares the
  `stop_completion` hook: if the agent tries to finish with zero tool
  calls, the hook forces another model turn with a "gather evidence"
  nudge. `qwen2.5-3b` then calls `system_health` (the only no-arg tool)
  and its long output can drown out the recall. Naming a tool keeps the
  turn deterministic while still requiring the model to read history.
- Session history is the prior **input** + a result summary (answer,
  root causes/recommendations) + an evidence **count**, not the full
  evidence. The rerun therefore proves the stored turns are enough for
  continuity without re-reading the live stores.
- A session is bound to one agent + mission: joining it with a different
  pair is rejected with 422 (`docs/sessions.md`).
- `--cleanup` deletes the demo session's `ExecutionResult`/`LLMUsage`/
  `AuditLog`/`Execution` rows (and the control session when `--control`
  is set). Without it the conversation stays in the DB for inspection.

## Notes from the agent config check (2026-11-11)

- `agent_config_check.py` closes the "config vs. what ran" gap: a run can
  succeed while the prompt/model/tools the model saw differ from the
  active `AgentVersion` (edit the YAML without re-seeding, edit the
  `agents` row without creating a version, or point the runtime at a
  different model). The script asserts all four layers instead of only
  the execution status.
- **Prompt prefix** is read from the MLflow trace's `CHAT_MODEL` span,
  not reconstructed locally: the system prompt the model received must
  `startswith` the pinned `prompt_text`. The skills index and the
  session-history block are appended after it by design, so the check is
  a prefix, not equality.
- **Model/temperature** come from the same span (`mlflow.llm.model`,
  `invocation_params.temperature`), and `llm_usage.model` in Postgres is
  checked as the runtime's own record.
- **`allowed_tools`** must all be offered at the model call, and no
  *registry* tool outside the list may leak through the gate. The 8
  deepagents built-ins (`ls`, `read_file`, `write_file`, `edit_file`,
  `glob`, `grep`, `task`, `delete`) are not in the registry gate; the
  script reports them as a warning, not a failure (known behaviour, see
  `docs/agent-versioning.md`).
- **Session limits** are asserted on turn 2 of a fresh session: the
  header must name the agent's own caps (`limited to N turn(s) / M
  tokens for agent X`) and exactly one `<data source="session_history">`
  block must be present. This proves the agent/version limits are the
  ones applied, not the globals.
- Traces are found by `execution_id`: search the experiment's runs for
  `tags.execution_id`, then `search_traces(run_id=...)` (the autolog
  trace links to the run via `metadata["mlflow.sourceRun"]`). The script
  polls up to `--trace-wait` seconds because the tracking server can lag
  the execution's SUCCEEDED state.
- `--no-trace` runs the static + DB checks without MLflow (useful on a
  stack without tracing); the prompt/model/tools checks are then
  reported as skipped, not passed.

## Notes from the hotel reception check (2026-11-12)

- The agent is config-only; every action goes through the 14 MCP tools in
  `seeds/agents/hotel_receptionist.yaml` (`docs/hotel-reception.md`).
  `hotel_reception.py` needs the mock server too:

  ```bash
  export NO_PROXY=127.0.0.1,localhost
  export no_proxy=127.0.0.1,localhost
  .venv/bin/python -m scripts.mock_hotel_mcp_server   # 127.0.0.1:8081/mcp
  ```

  The API and worker read `.env` at startup, and pydantic does not export
  `NO_PROXY`, so export it in the shell that launches them and restart
  after changing any `MCP_*` value.
- `MCP_SERVER_URL`, `MCP_TRANSPORT=streamable_http`, and
  `MCP_TOOL_CAPABILITIES` (all 14 tool names → `hotel_reception`) live in
  the repo-root `.env`. A tool with no mapping is denied by default; the
  preflight asserts the mapping, the server's tool list, and that a
  non-hotel mission gets none of the tools.
- MCP tools are environment-dependent, so they are **not** part of the
  pinned built-in tool manifest (`compute_tool_manifest` skips
  `permission == "mcp"`). If they leaked in, every long-lived worker
  would report false `agent_version_drift` after the server connected.
- The mock keeps bookings/tasks in SQLite (`mock_hotel.db`, gitignored),
  so data survives a restart — a clean state needs
  `python -m scripts.mock_hotel_mcp_server --reseed`, restart alone no
  longer resets it. The booking scenario uses a random 2027–2029 window
  per run instead of a fixed pair of dates. Booking ids grow from
  `BK-1004`; the cross-check scans contiguously until the first missing
  id. Verify the store any time with
  `python -m scripts.mock_hotel_mcp_server --dump`.
- With the local `qwen2.5-3b` the final prose is unreliable: it sometimes
  reads a hallucinated `/large_tool_results/*.json`, calls the
  deepagents `task` tool with an unknown subagent, or echoes plan-node
  noise. Tool **arguments**, `audit_logs`, the direct MCP calls, and the
  rendered session history are the source of truth. Prose checks become
  WARN only when the answer is such noise; a normal answer that is wrong
  still fails. So a run can legitimately end
  `PASS — N passed, 0 failed, few warnings`.
- The booking gate is intentionally two turns: turn 1 asks only for a
  price and must not call `create_booking`/`cancel_booking`/
  `change_booking_dates`; turn 2 confirms and must call `create_booking`
  with the exact guest, room, dates, and guest count. A direct
  `get_booking` MCP call then proves the booking really exists and is
  `CONFIRMED` — a hallucinated `BK-*` code fails the check. It uses
  Deluxe (2,000,000 VND/night) because the 3B model reliably misroutes
  Standard price questions to policy/staff tools. Booking is
  best-effort with the 3B model: a model that never completes the
  booking, or omits the code from its prose, only WARNs — but booking
  before confirmation and a `create_booking` error still FAIL, and a
  booking that exists is always cross-checked strictly. Turn 3 then
  asks "Cho tôi xin mã booking vừa tạo": when the code reached the
  prior-turn history the agent must recall that exact code; when it
  never reached the conversation the turn only WARNs, since the
  session-history summary is the prose answer and the agent has no
  other way to know the code. Turn 3 must never book again.
- The small model sometimes books during the quote turn anyway, or ends
  it in hook feedback. The quote is therefore retried up to 4 times in
  fresh `<session>-qN` conversations with a new date window; the gate
  needs one attempt that neither writes nor derails. Confirmation also
  allows two natural follow-ups in the same session before the tool/
  audit/MCP checks are evaluated.
- `--reseed` the mock before a clean full run: preflight asserts exactly
  5 Deluxe rooms on 2026-12-20, and earlier ad-hoc bookings can reduce
  that count (restart alone no longer resets the SQLite store).
- `memory` proves recall across executions, not inside one: turn 2 must
  repeat the random `HOTEL-*` code from turn 1, and the script also
  renders the exact prior-turn block the runtime gave turn 2 and turn 3.
  That platform-side check keeps the proof even when the model's final
  answer is hook feedback.
- `--cleanup` deletes the run's `ExecutionResult`, `LLMUsage`,
  `AuditLog`, then `Execution` rows per session; MLflow traces are kept,
  and the mock's SQLite bookings/tasks persist — use `--reseed` on the
  mock for a clean state.

## Notes from the long-scenario additions (2026-11-13)

- Five longer scenarios share four helpers (`_confirm_booking`,
  `_book_confirmed_deluxe`, `_ask_booking_code`, `_cancel_booking_flow`,
  `_change_dates_flow`) so the old scenarios stay untouched: `journey`
  (quote → confirm → create → code recall → change dates → re-check),
  `task_full` (housekeeping, room 305, incl. the `task-status` node),
  `cancel` (free + one-night-fee, each only after confirmation),
  `multi_request` (availability → policy → booking in one session,
  proving the subgraph re-routes and the resume rule), `soldout_switch`
  (Suite sold out → Deluxe on other dates).
- Every write gate (create / cancel / change) is two turns: an ask turn
  that must NOT call the write tool (FAIL if it does — same severity as
  booking before confirmation), then a confirm turn that must call it
  (one follow-up, then best-effort WARN when the 3B model never
  completes it; a tool error still FAILs). The store is the gate in
  every case (`get_booking` / `get_task` cross-check).
- `cancel` computes its late window from today at run time (check-in
  tomorrow → fee = one night, 2,000,000 VND); the free part uses a
  random future window like the booking scenario. The fee mention in
  the answer is checked only on coherent prose — the fee lives in the
  tool result the model may or may not quote.
- Known map gap the cancel/change ask turns surface: the
  `booking-change` node only has the write tools, so the agent cannot
  look the fee up itself before cancelling — it can only ask for
  confirmation. A turn A that answers the fee anyway did it from the
  policy doc via a re-route.
- The `multi_request` and `soldout_switch` quotes are single strict
  attempts (no `-qN` retries): the booking scenario already covers the
  flaky-quote retry path, and a retry in a fresh session would lose the
  conversation the scenario is testing.


## Notes from the MLflow report pipeline (2026-11-12)

`mlflow_report.py` turns the post-run analysis into a repeatable command.
It only reads MLflow, so it can run long after the API, worker and mock MCP
server are gone — and after `hotel_reception.py --cleanup` deleted the
Postgres rows (traces are the source of truth; cleanup never touches them).

Stages, one function each:

1. **discover** — resolve the experiment by name or id, list its runs and
   every trace (`--max-traces` caps both searches).
2. **join** — keep runs whose `tags.session_id` starts with
   `--session-prefix`, link each trace to its run via
   `mlflow.sourceRun`, and de-duplicate on `(run_id, execution_id)`.
3. **classify** — per trace: scenario (session suffix after the random
   id, with `-qN` retry suffixes removed), turn index inside the session,
   status / `agent_version` / drift, latency, tokens, span names split
   into contract vs built-in tools, `PostToolUseFailure` hook events,
   nested tool spans, model tool fan-out, derailed prose, explicit vs
   natural prompt, and the planted `HOTEL-*` memory code.
4. **aggregate** — per-scenario table, latency, tokens, tool usage, file
   tool noise, fan-out distribution, booking quote/confirm gate, memory
   recall, slowest/biggest outliers.
5. **render** — `reports/hotel-mlflow-report.md` plus
   `reports/hotel-mlflow-traces.csv` (one row per trace, all raw columns,
   including excluded scenarios).

Scope and definitions worth remembering:

- **Stat scope is the suite.** `adhoc` (exploratory probes) and `probe`
  (one-off wiring checks) are excluded from every statistic via
  `--exclude-scenarios` (default `adhoc,probe`); they still appear in the
  CSV and in the "toàn bộ" rows of section 1. Use
  `--exclude-scenarios ""` to analyse everything.
- **derail** is prose-only: the answer contains `tool_error_feedback`,
  `does not exist`, `deviation:`, `subagent`, or `plan nodes visited:`.
  It is a model-quality signal, not an execution failure.
- **hook feedback** counts traces with a `PostToolUseFailure` event whose
  decisions include `feedback`, i.e. the failure was fed back to the model.
- **nhiễu file** counts traces that called deepagents built-ins
  (`read_file`/`write_file`/…), which are outside the MCP capability gate.
- **booking** splits the first turn of each session (quote: clean if it
  only calls `check_room_availability`/`get_room_rates`, early write if it
  calls a write tool) from the remaining turns (confirm: did it call
  `create_booking`). Sessions are retried as `-qN`, so each attempt is its
  own session and is counted.
- **memory** plants the code in the first turn of a session and checks
  recall in the later turns' prose, in the rendered
  `<data source="session_history">` block, and per session.
- **nested tool spans** are instrumentation (`ToolNode` span → registered
  wrapper → inner tool): one MCP call yields two nested spans with the same
  `tool_call_id` and therefore two `audit_logs` rows. Section 12 reports
  them as a reading note, not as double execution.
- Numbers are data-derived, so a re-run after new tests changes them;
  the report prints the generation time, tracking URI, experiment and
  filters at the top so two reports can be compared honestly.
