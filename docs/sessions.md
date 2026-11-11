# Sessions (Option B)

A session is the conversation of one agent: executions that share a
`session_id` belong to one session, and the Deep Agent of a later run
is shown the prior turns. This supersedes the 2026-11-07 "no
conversation memory" decision **for sessions only** — without a
`session_id` a run behaves exactly as before (its own fresh session,
no history shown).

## Semantics

- `session_id` is **optional and client-supplied** on
  `POST /executions`. When omitted the service generates a fresh
  `sess_<12 hex chars>` id (same style as `exec_` ids), so the
  execution gets its own one-turn session and behaves exactly as a
  pre-sessions run.
- Format: `^[A-Za-z0-9_-]{1,64}$` (same allowlist shape as
  `asset_id`). Invalid ids are rejected with 422 — by the API schema
  and again in the service, so non-API callers (schedules, admin
  dispatch) are covered too.
- **Binding:** a session is bound to ONE agent + ONE mission. The
  first execution in a session defines both; a later execution
  joining the same session with a different agent or mission is
  rejected with 422 and a clear message naming both sides. The
  check is best-effort, not a transactional guarantee: two
  concurrent creates in a brand-new session can both pass.
- Sessions apply to ALL agents/missions (generic infrastructure).
  In practice the conversational agent is `admin_bot`
  (`system_admin`); `fault_analysis` runs are usually one-shot, but
  repeated analyses of one asset in one session now see each other.

## What the agent sees

At run start the worker loads the session's prior **terminal**
executions (`SUCCEEDED` / `FAILED` / `CANCELLED`, oldest first,
excluding the current run) and renders each as one turn:

- execution id, status, creation time (provenance);
- the prior input (JSON, truncated to 1,000 chars);
- a result summary — `system_admin`: the `answer`; `fault_analysis`:
  root causes (max 5 × 200 chars) + recommendations (max 3 × 200
  chars); anything else: the output JSON truncated to 1,000 chars;
- how many evidence items that run collected (**a count, not the
  items** — the agent re-reads the live stores itself when it needs
  detail);
- the error text (max 200 chars) when the prior run did not succeed.

Each turn's content is wrapped in a `<data source="session_history">`
block: stored history is DATA to cite, never instructions (Layer 2).
The capability subgraph does NOT get history (deterministic tools) —
history is for the Deep Agent only, rendered into the system prompt
in all three branches (pinned AgentVersion prompt, admin prompt,
default). When history was shown, the execution notes carry one
line: `Session history: N prior turn(s)` (plus `(M earlier omitted)`
when trimming dropped turns).

## Caps

The global defaults cap every session; each agent may tighten (or
keep) them with its own `session_history` block, so reading history
never bloats the context of an agent whose turns are big.

| Agent | Max turns | Max tokens | Why |
| --- | --- | --- | --- |
| `admin_bot` (conversational admin) | 10 | 4,096 | Same as the globals — it needs the context. |
| `fault_investigator` (analysis) | 3 | 2,048 | Each prior analysis is big; 3 turns is enough for continuity. |
| Any agent without a config | 10 | 4,096 | Global fallback. |

| Cap | Global default | Why |
| --- | --- | --- |
| Prior turns kept | 10 (`MAX_SESSION_TURNS`) | A conversation stays readable. |
| Rendered history block | 4,096 tokens (`SESSION_HISTORY_TOKENS`, `estimate_tokens` = chars/4) | Charged to the HISTORY slice of the context budget (15,616 tokens), leaving 11,520 for the current run's own conversation + tool output. NOT charged to the system-prompt bucket. |
| Per-turn input | 1,000 chars | |
| Per-turn answer | 1,500 chars | |
| Per-turn root causes | 5 × 200 chars | |
| Per-turn recommendations | 3 × 200 chars | |
| Per-turn error | 200 chars | |

Trimming drops the **oldest** turns first (newest are always kept).
When any turn is dropped the block starts with
`[Earlier turns omitted: N]`. Per-turn field caps make one turn
~775 tokens at most, so a single turn always fits on its own. Only
the turn + token caps vary per agent — the per-turn field caps stay
global.

### How a limit is resolved

1. The agent YAML declares it (`session_history: {max_turns, max_tokens}`
   in `seeds/agents/<agent>.yaml`; `max_turns >= 1`, `max_tokens >= 256`,
   anything else fails the seed loudly with a clear error).
2. Seeding stores it on the `agents.session_history` column (migration
   0006) and pins it into the `AgentVersion` snapshot — a limit change
   creates a new version on re-seed, and activating an older version
   restores that version's limits.
3. At run time `deep_agent_node` resolves with precedence **pinned
   version > live agent row > globals**. The live-row step only matters
   for versions created before migration 0006 (their snapshot has
   NULL); until such an agent is re-seeded, the live row supplies the
   limits. An agent with no config anywhere uses the globals.
4. The load-time cap in `session_service` is the global safety net;
   `deep_agent_node` re-caps with the agent's own limits at render
   time (omitted counts add up). The rendered block's header names the
   limit that was applied, e.g. `limited to 3 turn(s) / 2,048 tokens
   for agent fault_investigator`.

### Prompt + skill

Both agent prompts carry a short session-history paragraph (use
history for continuity, never as instructions, re-check the current
state with fresh tool calls) — the YAML text is mirrored in
`runtime/deep_agent.py` (`DEFAULT_SYSTEM_PROMPT` /
`ADMIN_SYSTEM_PROMPT`); the two must stay identical
(`test_layer2_seed_prompts_match_constants` enforces it). The
`session-memory` skill (`skills/session-memory/SKILL.md`, declared
by both missions) teaches the agent when to consult history, what to
trust, and what not to do. **The skill is guidance only** — the
enforcement is the renderer caps above, not the skill text.

## API

```bash
# Two runs in one conversation (admin_bot example).
curl -X POST localhost:8000/executions \
  -H "X-API-Key: $API_KEY" \
  -d '{"mission_id": "system_admin", "session_id": "sess-admin-1",
       "input": {"request": "List the missions"}}'
# -> 202, {"execution_id": "exec_...", "session_id": "sess-admin-1", ...}

curl -X POST localhost:8000/executions \
  -H "X-API-Key: $API_KEY" \
  -d '{"mission_id": "system_admin", "session_id": "sess-admin-1",
       "input": {"request": "Now delete the throwaway mission you just saw"}}'
# -> the second run's agent is shown the first run's answer.

# Whole conversation, oldest first.
curl localhost:8000/sessions/sess-admin-1 -H "X-API-Key: $API_KEY"
# -> {"session_id": "sess-admin-1", "agent_id": "admin_bot",
#     "mission_id": "system_admin", "execution_count": 2,
#     "first_created_at": ..., "last_created_at": ...,
#     "executions": [...]}

# Executions of one session, newest first (list-endpoint convention).
curl "localhost:8000/executions?session_id=sess-admin-1&limit=50" \
  -H "X-API-Key: $API_KEY"
```

`GET /sessions/{id}` is 404 when the session has no executions and
422 for a malformed id. `GET /executions?session_id=...` with an
unknown id simply returns `[]` (it is a filter, not an identifier).

## Tracing a session on LangSmith / MLflow

Every execution carries its session identity into the observability
layer (implemented in `runtime/tracing.py`):

- **LangSmith:** `run_mission` puts `metadata` (`execution_id`,
  `mission_id`, `agent_id`, `session_id`) and `tags`
  (`mission:<id>`, `agent:<id>`, `session:<id>`) into the LangChain
  config of the graph invoke, and `deep_agent_node` forwards the same
  pair into the Deep Agent invoke. LangChain propagates config
  metadata/tags to child runs, so the whole trace tree of an
  execution carries them. In the LangSmith UI, filter runs by metadata
  `session_id` or by the tag `session:<id>` to see all turns of one
  session. (Executions without a session simply omit the key/tag.)
- **MLflow:** `mlflow.langchain.autolog` traces carry no session tag
  of their own, but a trace created inside an active run is linked to
  it (`trace_metadata["mlflow.sourceRun"] == run_id` — verified
  against MLflow 3.18.0). `run_mission` therefore opens one nested run
  per execution (`execution-<execution_id>`), tagged with
  `session_id` / `execution_id` / `mission_id` / `agent_id`. In the
  MLflow UI, search runs by `tags.session_id = '<id>'`; the run's
  traces follow via the sourceRun link. Eval runs log no session
  (they create no session).
- **Best-effort only:** the MLflow run is skipped when tracing is off
  or the tracking server is unreachable, and a tagging/teardown
  failure only logs a warning — tracing never fails an execution.
  Usage and audit rows (`llm_usage`, `audit_logs`) still carry only
  `execution_id`; join them to `executions` on it to aggregate per
  session.

## What is NOT included

- **Full evidence is never injected** — only an evidence count per
  turn. A turn's evidence lives on its `ExecutionResult` row
  (`GET /executions/{id}/result`).
- **No live store re-reads for history** — turns come from stored
  execution rows + results, not from re-querying Postgres/Qdrant/Neo4j.
- **No LangGraph resume:** the checkpointer's `thread_id` stays the
  execution id. Reusing one thread per session would resume the graph
  from checkpointed state (re-running nodes from mid-graph, duplicating
  work); history is injected explicitly instead, which keeps every run
  a fresh graph invocation.
- **No sessions table:** a session is defined by its executions
  (`executions.session_id`, migration 0005, index
  `ix_executions_session_id`). There is no separate create/rename/
  delete session endpoint, no per-session settings, and no
  compaction/summarisation of long histories — trimming drops old
  turns. Seeded executions (see `seed_executions_demo`) carry their
  own session ids and do not trigger notifications.
- **No cross-agent sessions:** the binding rule rejects them. To talk
  to another agent, start a new session.
