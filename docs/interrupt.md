# Human-in-the-loop interrupts (`interrupt_on`)

MDA's `interrupt_on`: a mission declares the tools that must pause for
a human decision before they run. The run stops, the worker persists
status `INTERRUPTED` with the pending calls in the result, and a human
resumes it with approve/reject. Unlike the permission-request flow
(`docs/prompt-injection-defense.md`, Option C), which re-dispatches a
*fresh* execution after approval, an interrupt resumes the *same*
execution in place — history, evidence and context survive.

## Enabling it

In the mission YAML, under `execution`:

```yaml
execution:
  mode: autonomous
  interrupt_on:
    delete_mission: true
    cancel_execution:
      allowed_decisions: [approve, reject]
      description: "Cancelling an execution is irreversible."
```

Values are the `create_deep_agent` `interrupt_on` mapping: a tool
name to `true` (interrupt before the call, decisions approve/reject)
or to a config with `allowed_decisions` and `description` only
(`when` is a callable and cannot come from YAML). A malformed
declaration fails the run fast with a clear error — a typo must never
silently leave a tool ungated. No mission enables it by default;
opt in per mission.

## Resume flow

1. The agent calls a gated tool → the deep agent pauses; the mission
   graph pauses inside `deep_agent_node`; the worker stores
   `status: INTERRUPTED` and `output.interrupt`:
   `{"requests": [{"tool": ..., "args": {...}}], "allowed_decisions": ["approve", "reject"]}`.
   `finished_at` stays unset — the run is waiting, not finished.
2. `POST /executions/{id}/resume` with `{"decision": "approve"}`
   (or `"reject"`). Same ownership rule as GET: another user's run is
   404; a non-INTERRUPTED run is 409; any other decision is 422.
3. The worker resumes the same execution: the mission graph continues
   after its `interrupt()`, the deep agent continues with the
   decision, and the run finishes normally. Approve runs every pending
   call; reject skips them with a reviewer message. A run that hits
   another gated call later pauses again (up to 10 pauses per run).

## Checkpointer requirement

A pause only survives a worker restart when the checkpointer is
Postgres: set `CHECKPOINT_ENABLED=true`, install the
`postgres-checkpoint` extra (`pip install -e ".[postgres-checkpoint]"`),
restart the worker. Without it, a mission that declares
`interrupt_on` refuses to run (fail-closed — the gated tool must not
run unapproved). In dev/test the fallback is a process-local memory
checkpointer: pause/resume works, but only in the same process.

## What it does not do (v1)

- **Flow mode**: `flow_subgraph.enabled` missions cannot declare
  `interrupt_on` yet — the run fails fast with a clear error. The
  flow compiler needs its own checkpoint/resume plumbing (flow
  subgraph thread + per-flow agent threads). Free-form agents
  (admin_bot, fault escalation) work now.
- **Decisions**: approve/reject only. `edit` (change the tool args)
  and `respond` (answer instead of calling) are not wired yet.
- **The permission flow stays**: `delete_mission` /
  `cancel_execution` for admin_bot are still gated by the
  `request_permission` hook + `GET/POST /permissions` (Option C).
  A mission may use either mechanism; they do not replace each other.
- **Only resume via `POST /resume`**: re-dispatching an INTERRUPTED
  execution any other way fails it with a clear error — the pause is
  consumed only by a decision that reaches the worker.
