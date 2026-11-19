# Adding an agent (declarative harness)

Each mission declares runtime policy in a **`harness:`** block (merged with
`runtime/harness/defaults.yaml` and validated at seed time).

Configure **`HARNESS_SOURCE`** (`db` | `seed`, default `db` in production).

| Policy | Source of truth |
|--------|-----------------|
| RBAC (`user_runnable`), guest `/guest/chat` routing, builtin deny / flow flags in the global index | **`HARNESS_SOURCE=db`:** `missions.execution_config.harness` for rows in Postgres; per-row fallback to that mission's seed YAML only when the `harness` key is missing; invalid DB harness → closed defaults (deny off). **Empty table → no missions in index** (no RBAC grants). **`HARNESS_SOURCE=seed`:** all `seeds/missions/*.yaml`. |
| Per-execution deny / flow during a run | `execution_config.harness` on the mission snapshot (same as DB after seed) |
| Guest tool registration (`reply_to_guest`, `confirm_guest_booking`) | **Image `seeds/missions/*.yaml` only** at process start (not DB) |
| No AgentVersion system prompt | **Image `seeds/agents/*.yaml`:** `harness.system_prompt_fallback` (`default` / `admin` / `seed`) |

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
| `context_budget.reclaim_retrieval_for_history` | Opt-in: move unused retrieval slice into history trim budget |

Agent YAML: `harness.system_prompt_fallback`:

- `default` / `admin` — code constants (fault / admin agents)
- `seed` — use the agent file's `system_prompt` when no AgentVersion (hotel)

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