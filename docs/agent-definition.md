# Agent Definition — the MDA shape on top of missions

This doc covers the four Group A features of the MDA standardization
(see `~/workspace/your_files/ke-hoach-agent-ke-thua/ke-hoach-agent-ke-thua.md`
§9): `interrupt_on` (docs/interrupt.md), `response_format` /
`context_schema`, the Threads/Runs API, and SSE streaming. All four
are **declarations on the mission YAML / API surface** — the agent
classes themselves stay YAML-driven; nothing here moves a decision
from a skill map into code.

## response_format — structured output (A.3)

`execution.response_format` on a mission resolves to a JSON Schema
handed to `create_deep_agent` as its `response_format`. The model
must return that shape; `run_deep_agent` folds
`result["structured_response"]` into the outcome (a summary/answer
field beats the prose summary), and `deep_agent_node` merges the
structured keys into the result — structured keys win over
prose-derived mappings for the mission's required output keys.

Values (see `runtime/agent_context.py::resolve_response_format`):

- omitted / `false` — off (the default). The 3B model is unreliable
  at whole-JSON output; analytis file-output and hotel guest-text
  don't want it.
- `true` — use the mission's own `output_schema` (must have
  non-empty `properties`, else the mission fails fast at build time).
- a JSON Schema dict — used as-is (must be an object schema).

Rules:

- `flow_subgraph.enabled` + `response_format` fails fast — the flow
  compiler has its own result assembly and does not merge structured
  output yet (v2).
- The format is resolved next to `interrupt_on` in `deep_agent_node`
  and passed to `build_deep_agent` only when configured (test
  doubles for `build_deep_agent` have fixed signatures — new build
  kwargs are only added to `_agent_kw` when configured).

## context_schema — typed agent context (A.3)

`runtime/agent_context.py::AgentContext` is the TypedDict for what a
run carries about itself: `mission_id`, `execution_id`, `tenant`,
`capabilities`, `input`, `execution_config`, `session_id`, `user_id`.
`validate_agent_context` type-checks it (`execution_id` must be a
string — the node builds it with `state.get("execution_id") or ""`)
and the node fails fast at build time when it is malformed.

`context_schema` is deliberately **not** passed into
`create_deep_agent`: `invoke` carries no context object today, so
the graph has nothing to type-check. `AgentContext` is enforced at
the node boundary instead. Passing a schema the graph never sees
would be a dead parameter — when a context object exists at invoke
time, wire it then.

## Threads / Runs API (A.4)

The MDA Agent Server shape over the existing session/execution
model. A **thread is a session** (`thread_id` is the `session_id`)
and a **run is an execution** (`run_id` is the `execution_id`).
These routes are thin aliases in `app/routers/threads.py` — no new
tables, no new agent code; `GET /threads/{id}` and
`GET /sessions/{id}` always agree, as do `GET /runs/{id}` and
`GET /executions/{id}`.

| Route | Behaviour |
| --- | --- |
| `POST /threads` (201) | Mint a thread id (`sess_…`). No row until the first run — like sessions, threads are created implicitly, so GET on a fresh id is 404. |
| `GET /threads/{thread_id}` | Thread summary: agent/mission, execution count, first/last created, runs oldest first. |
| `GET /threads/{thread_id}/runs` | The thread's runs (oldest first). |
| `POST /threads/{thread_id}/runs` (202) | Run a mission in the thread — the MDA-shaped `POST /executions` (same mission-role rule, same input validation, same dispatch). |
| `GET /runs/{run_id}` | One run. |
| `GET /runs/{run_id}/result` | The run's result — same read as `GET /executions/{id}/result` (guest projection included). |

Ownership mirrors the aliased routes exactly: a JWT user only
touches their own threads/runs (another user's is 404); service-key
calls stay unrestricted. Creating a run needs no ownership check —
like `POST /executions`, the run is owned by whoever creates it.

Schemas: `ThreadRead`, `RunRead`, `ThreadRunCreate` in
`app/schemas.py`.

## Streaming — SSE lifecycle events (A.2)

`GET /executions/{execution_id}/stream` streams the run's lifecycle
as Server-Sent Events (`text/event-stream`, `event: status` +
`data: {json}`). The worker publishes one small JSON event per
status transition to Redis pub/sub (`runtime/events.py`, channel
`exec_events:{execution_id}`): `RUNNING` after the claim, then the
final status — or `INTERRUPTED` with the pending gated calls, so a
listening client learns at once that a human decision waits
(docs/interrupt.md). `CANCELLED` is published from
`execution_service.cancel_execution`.

The event payload carries **no output** — the client fetches the
full result from `GET /executions/{id}/result` when the stream ends.
Token-by-token model output is deliberately out of scope for v1
(it needs callback-handler plumbing through the deep agent).

Reliability rules:

- Publishing never raises and never blocks a run. A missing or
  unreachable Redis means the event is lost — the endpoint also
  re-reads the executions row every second, so a lost publish still
  surfaces within 1s. The broker URL is not always Redis (tests use
  `memory://`), in which case the publisher is a no-op and the
  endpoint runs on DB polling alone.
- A status change is committed **before** its publish, and a row
  already terminal at subscribe time is emitted from the DB — so no
  event is missed at the subscribe boundary.
- The stream closes on the terminal event or after
  `MAX_STREAM_SECONDS` (600s); a `: ping` comment goes out every
  15s so proxies don't close an idle stream. The client reconnects
  (or polls) from there.
- Same ownership rule as GET: another user's run is 404.

Guest chat CLI: `scripts/manual/guest_chat.py --stream` waits on the
SSE stream instead of polling, prints status transitions as they
arrive, then fetches the result once. A stream failure (or a result
row not written yet when the stream ends) falls back to the old
polling loop. An `INTERRUPTED` terminal prints the pending gated
calls instead of an answer.
