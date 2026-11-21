# Adding an agent (declarative harness)

Each mission declares runtime policy in a **`harness:`** block (merged with
`runtime/harness/defaults.yaml` and validated at seed time).

Configure **`HARNESS_SOURCE`** (`db` | `seed`, default `db` in production).

| Policy | Source of truth |
|--------|-----------------|
| RBAC (`user_runnable`), guest `/guest/chat` routing, builtin deny / flow flags in the global index | **`HARNESS_SOURCE=db`:** `missions.execution_config.harness` for rows in Postgres; per-row fallback to that mission's seed YAML only when the `harness` key is missing; invalid DB harness → closed defaults (deny off). **Empty table → no missions in index** (no RBAC grants). **`HARNESS_SOURCE=seed`:** all `seeds/missions/*.yaml`. |
| Per-execution deny / flow during a run | `execution_config.harness` on the mission snapshot (same as DB after seed) |
| Guest tool registration (`reply_to_guest`, `confirm_booking_create`) | **Image `seeds/missions/*.yaml` only** at process start (not DB) |
| No AgentVersion system prompt | The agent class's own fallback: `BaseAgent.fallback_system_prompt` reads the seed YAML; `AdminAgent` returns its own text (no `system_prompt_fallback` Literal anymore) |

`GET /health` reports `harness_index_degraded` when the DB index refresh failed
and a stale snapshot is served.

**`security.user_runnable` defaults to false.**

Docker images copy `seeds/` and `skills/` at build time — rebuild the image
(or mount the repo) after adding scaffolded files, then run `make seed`.

## Harness fields (mission YAML)

| Block | Purpose |
|-------|---------|
| `security.user_runnable` | JWT `user` role may `POST /executions` for this mission |
| `guest_channel.exposes_public_chat` | Route `POST /guest/chat` here (exactly one mission) |
| `builtin_tools.deny` | Hide/deny deepagents built-ins (hotel) |
| `builtin_tools.deny_message_template` | Hook message (`{tool_name}` placeholder only) when deny is true |
| `flow_subgraph.enabled` | Use the structured flow router when skills are mapped |
| `guest_output.register_reply_tools` | Register `reply_to_guest` / booking confirmation tools (seed-time only) |
| `guest_output.capability` | Registry capability for those synthetic tools |
| `notifications.notify_admin_actions` | SUCCEEDED runs of this mission notify Slack when output has a non-empty `actions_taken` (admin actions; failures notify for every mission) |
| `context_budget.reclaim_retrieval_for_history` | Opt-in: move unused retrieval slice into history trim budget |
| `output_files.required` | Opt-in: `stop_completion` nudges the agent to write top-level report files under `/output` in the live sandbox before stopping (uses the same export marker + find logic as file export) |
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