# Skill maps — plans for the Deep Agent

Option C (tran 2026-11-11): a skill's guidance can be a **map** — a plan
the Deep Agent follows — instead of a linear playbook. The map keeps
the agent's autonomy: it may deviate when the evidence justifies it,
but it must say why, and the run is audited deterministically from its
tool calls. The context stays clean: only the plan ("how to do it") +
tools are injected; data is fetched through tools when a node needs it,
never stuffed in upfront.

> **2026-11-13 — the map is now enforced by code, not just prose.**
> For the pilot mission the map is compiled into a LangGraph subgraph;
> for missions still on the free-form path a plan gate enforces the
> order at the tool-call boundary. See [Enforcement](#enforcement-subgraph--plan-gate)
> below. Option C's "deviate with reason" now applies only to missions
> not yet on the subgraph.
>
> **2026-11-15 — Direction C: the pilot moved from one agent per map
> node to one agent per flow** (see the flow bullet below). The node
> compiler stays as the fallback; the v4b live run showed its failure
> mode is structural (a `NEXT:` line inside the guest's answer, one
> routing decision per node, a silent walk across flow boundaries on a
> bad parse), not a parse bug to patch.

## Enforcement: subgraph + plan gate

One run uses exactly one enforcement mechanism, chosen in
`deep_agent_node` (`runtime/mission_graph.py`):

- **Flow subgraph** (`runtime/subgraph/flow.py`, pilot:
  `hotel_reception`, Direction C, 2026-11-15). `build_flow_subgraph`
  compiles a `SkillMap` into a LangGraph of **flow agents + one
  router**: flows are derived from the map (each branch target of the
  start node starts a flow; the flow walks the map in list order until
  a terminal node or a resume node, which starts its own flow; the
  start node itself is the router, never a flow member). Each flow
  runs ONE Deep Agent with the **union of its flow's tools** and its
  steps as a checklist — no `NEXT:` inside a flow. The router is a
  plain model call that declares `FLOW: <flow-id>` (tolerant parse:
  bold markers, indentation, trailing text after the token are all
  accepted) once per flow boundary; after a flow ends it runs again so
  a multi-request turn can enter a second flow, and a flow already
  visited this turn is never re-entered. An invalid or missing
  declaration ENDS the turn with an error recorded — there is no
  "next in order" fallback, because in a flat list of independent
  flows there is no meaningful next. The answer channel is separate
  from the control channel: each flow writes its reply text to a
  dedicated `answer` state key, and the outcome summary comes from
  that key, never from the message list — so no control line can leak
  into the guest's answer. Resume nodes (e.g. `create-booking`) are
  flow boundaries: the booking-create flow is enterable only when the
  session already has prior turns, and its tools are only
  `create_booking` + the report step — the booking-quote flow does not
  HAVE `create_booking`, so the agent cannot call it before the guest
  confirms. Step cap: 12 node executions per run (route calls + flow
  executions). A second, finer cap lives INSIDE each flow: a per-flow
  tool-call budget (`MAX_TOOL_CALLS_PER_FLOW`, default 10 — a flow
  covers several map nodes, so the old flat 5 is too small) plus a
  repeat guard (`_ToolCallCap`, first in the flow agent's middleware
  list so its raise escapes the hook layer). The step cap counts node
  executions, not calls inside a flow — without this, a model
  repeating one failing call (2026-11-13: `get_booking` with a bad id,
  looped to the recursion limit) never finishes. A repeat of ANY
  earlier failed call in the execution is denied (the set, not just
  the immediate predecessor — interleaved repeats evaded the old
  guard); past the budget the flow stops with a diagnostic error and
  the turn ends. After each flow, `flow_trace` records which of the
  flow's steps ran and which are missing (checklist audit for the v5
  acceptance check, not a second router). `stop_completion` is scoped
  OUT of flow agents (the mission graph builds a flow middleware whose
  registry excludes it — a flow agent that stops without tools is
  usually asking the guest a question). The flow path does **not**
  render the prose plan block and does **not** attach the plan gate —
  the graph itself is the enforcement. `run_flow_subgraph` returns the
  same outcome shape as `run_deep_agent`, plus `visited_nodes` (flow
  ids) and `flow_trace`.
- **Node subgraph** (`runtime/subgraph/node.py`) — LEGACY, superseded
  by the flow compiler for the pilot; kept for its tests and as the
  fallback path. Same shape (one Deep Agent per map node, code router
  over `NEXT:` lines, fallback to next-in-order, `NEXT: END` honored
  only at terminal nodes). Both compilers live in `runtime/subgraph/`
  with their shared pieces (`ToolCallCap`, `tools_for_node`,
  `next_in_order`, `last_text`) in `common.py`.
- **Plan gate** (`runtime/plan_gate.py`, `PLAN_GATE_MISSIONS =
  {"hotel_reception"}` — currently no production mission attaches it,
  since hotel moved to the subgraph). An `AgentMiddleware` built from
  the run's parsed maps: it allows the current node's tools, the next
  tool node in order, branch targets of the current node and of the
  start node (re-route), and already-visited tool nodes; anything
  else is denied with a ToolMessage naming the allowed next tools.
  After 5 denials the run fails with `PlanGateExceededError`. Tools
  in no map node are allowed with a warning (fail-open; map coverage
  is enforced by test instead).

The two mechanisms share the map schema and the `plan_trace`
diagnostics below; the gate is the enforcement for missions still on
the free-form path, the subgraph is the enforcement for the pilot.

## Map schema

A map lives in the skill's frontmatter under `map:` (structured YAML,
machine-readable; the markdown body stays the human playbook):

```yaml
map:
  start: identify
  nodes:
    - id: identify
      goal: Parse asset_id and date range from the input.
      tools: []
      stop_when: asset_id missing or invalid — stop and report it.
    - id: postgres-faults
      goal: Search fault history for the asset in the date range.
      tools: [search_fault_history]
      branches:
        - when: no faults found — do not stop, continue to maintenance
          goto: postgres-maintenance
    - id: synthesize
      goal: Draft findings only from the collected evidence.
      tools: []
      stop_when: enough evidence collected.
```

Rules (`runtime/skills/map_plan.py`):

- `start` names the first node (defaults to the first node listed).
- The default flow is the **next node in list order**; a branch is a
  jump taken when its `when` condition holds. `when` is plain words —
  the agent decides whether it holds.
- A node with `tools: []` (or no `tools`) is a reasoning-only step.
- Unknown tool names are dropped with a warning (validated against the
  live tool registry when it can be loaded); unknown `goto` targets
  drop their branch. A malformed map logs a warning and is ignored —
  the skill then works exactly as before, map-less. A skill without a
  map is unaffected (`skill.map is None`).
- Unknown extra keys are ignored.

## How plan injection works

1. `runtime/skills/loader.py` parses the map at skill-load time and
   attaches it as `skill.map`.
2. On the **free-form path**, `deep_agent_node`
   (`runtime/mission_graph.py`) renders the maps of the selected skills
   (`render_plan`) and appends the plan block to whichever system
   prompt runs (pinned AgentVersion prompt, admin prompt, or default).
   The plan is skill-derived runtime context — like the skill files
   themselves — not part of the pinned prompt text. Skills without a
   map change nothing. With `enforced=True`, the header states the
   plan gate refuses out-of-step calls; the default header keeps the
   Option C "you may deviate" text. On the **subgraph path** no plan
   block is rendered at all — the compiled graph carries the plan.
3. The plan block states the contract: follow the plan; deviate only
   when the evidence justifies it and say why; report visited nodes
   and deviations at the end. The data-access pilot renders to ~370
   tokens (skills budget: 1,024).
4. The native deepagents SkillsMiddleware still discloses the full
   `SKILL.md` file (including the frontmatter map) on demand — the
   rendered plan in the system prompt is the operative copy.

## How plan_trace is computed

Deterministic, from the run's tool calls — no LLM cooperation needed
(`compute_plan_trace` in `runtime/skills/map_plan.py`, called from
`run_deep_agent` when `skill_maps` are passed):

- **visited**: node ids in the order their tools were first called. A
  node counts as visited when any of its tools was called — failed
  calls count (the agent attempted the node); grounding of
  `actions_taken` is separate and unchanged.
- **tools_by_node**: which tools were called per node, in call order.
  A tool claimed by two nodes counts for the first node in map order.
- **skipped**: node ids never visited, in map order.
- **deviations**: tools that belong to no node of that map.

Each mapped skill gets its own trace, keyed by skill name:
`result["plan_trace"]["data-access"]`. `deep_agent_node` also adds a
one-line summary per skill to the execution notes, e.g.:

> Plan (data-access): visited identify -> postgres-faults ->
> qdrant-documents; skipped postgres-sensors; deviation:
> calculate_failure_rate (not in map)

`plan_trace` is additive: `validate_results` checks only the declared
output-schema keys and ignores extra keys.

Compatibility: doubles predating `skill_maps` (older tests, eval
mocks) don't accept the kwarg — the node retries the call without it
and the run proceeds, just without a plan trace.

## Pilot: data-access

`skills/data-access/SKILL.md` — the first map skill, declared by
`fault_analysis` (alongside `fault-investigation` and
`report-writing`). Its map walks the three stores for one asset:

`identify` (parse + validate input) → `postgres-faults`
(`search_fault_history`) → `postgres-maintenance`
(`get_maintenance_records`) → `postgres-sensors` (`get_sensor_data`,
skipped when faults + maintenance are both empty) →
`graph-relations` (`query_asset_relationships`, Neo4j) →
`qdrant-documents` (`search_technical_documents`) → `synthesize`
(findings only from collected evidence; stop when all three stores
are unreachable or enough evidence is collected).

## Overlap with the retrieval hook (honest note)

`fault_analysis` still declares a `retrieval` config, so the
`retrieval_context` hook injects upfront data (faults, documents,
relations) into the agent's context at SessionStart — the thing the
map approach was meant to avoid. The map is for the Deep Agent path;
the hook also serves the capability path's context. If the map proves
itself, a follow-up can drop the retrieval config for
`fault_analysis` and let the agent pull everything through the map's
tools; that change is not made here.

## Writing a new map skill

1. Create `skills/<name>/SKILL.md` with frontmatter `name`,
   `description`, `keywords`, and a `map:` block using only real tool
   names (see `runtime/tools/` — unknown names are dropped).
2. Keep the markdown body short: when to use, the map pointer, rules,
   what not to do. The map is the plan; the body is the guardrails.
3. Declare the skill in the mission YAML (`execution.skills`) or rely
   on keyword matching (threshold 1 — keep keywords specific).
4. Add tests in `tests/test_skill_maps.py`: parsing (valid/invalid),
   rendering (compact — aim ≤ 600 tokens), and a trace test with a
   recording agent.
