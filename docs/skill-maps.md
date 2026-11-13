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

## Enforcement: subgraph + plan gate

One run uses exactly one enforcement mechanism, chosen in
`deep_agent_node` (`runtime/mission_graph.py`):

- **Skill subgraph** (`runtime/skill_subgraph.py`, pilot:
  `hotel_reception`). `build_skill_subgraph` compiles a `SkillMap`
  into a LangGraph: one node per map node, each running its own Deep
  Agent with **only that node's tools** (tool-less nodes run a plain
  model call). The model ends each step with a `NEXT: <node-id>` line
  and a **code router** validates it against the map (branch targets
  of the current node + next node in list order + resume nodes); an
  absent or invalid declaration falls back to next-in-order, never an
  arbitrary jump. The node's instruction lists each branch choice with
  its `when` condition, so the model decides on the condition, not a
  bare id. `NEXT: END` is honored only at a flow-terminal node
  (`terminal: true` in the map); anywhere else it falls back to
  next-in-order — the model judging a request "handled" mid-flow must
  not end the run. `NEXT: WAIT` ends the turn from any node: the node
  needs more information from the guest before the flow can continue.
  Resume nodes (e.g. `create-booking`) are enterable
  only when the session already has prior turns — so a fresh turn
  stops at the confirmation question and the guest's confirmation
  turn resumes into the booking. Step cap: 12 node executions per run.
  The subgraph path does **not** render the prose plan block and does
  **not** attach the plan gate — the graph itself is the enforcement.
  `run_skill_subgraph` returns the same outcome shape as
  `run_deep_agent`, plus `visited_nodes` (the graph's full path;
  `plan_trace` only sees nodes with tool calls).
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
