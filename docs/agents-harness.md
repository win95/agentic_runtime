# Adding an agent (declarative harness)

Each mission declares runtime policy in a **`harness:`** block (merged with
`runtime/harness/defaults.yaml` and validated at seed time). The runtime
reads the pinned `execution.harness` snapshot from Postgres — no Python
frozensets per agent.

## Harness fields (mission YAML)

| Block | Purpose |
|-------|---------|
| `security.user_runnable` | JWT `user` role may `POST /executions` for this mission |
| `guest_channel.exposes_public_chat` | Route `POST /guest/chat` here (exactly one mission) |
| `builtin_tools.deny` | Hide/deny deepagents built-ins (hotel) |
| `builtin_tools.deny_message_template` | Hook message (`{tool_name}` placeholder) when deny is true |
| `flow_subgraph.enabled` | Use the structured flow router when skills are mapped |
| `guest_output.register_reply_tools` | Register `reply_to_guest` / booking confirmation tools |
| `guest_output.capability` | Registry capability for those synthetic tools |

Agent YAML may include `harness.system_prompt_fallback: default|admin` for
drift checks against `runtime/deep_agent.py` constants (fault/admin only).

## Add a new agent (after)

1. `make new-agent NAME=my_agent` — agent + mission + `skills/.../SKILL.md`
2. Edit the new YAML: `allowed_tools`, objective, harness overrides if needed
3. `make seed-agent AGENT=my_agent MISSION=my_agent_mission`
4. Optional: map MCP tools in `.env` `MCP_TOOL_CAPABILITIES`
5. `python scripts/manual/agent_config_check.py` (harness + config checks)

**Five steps**, one Python package touched only when you add genuinely new
runtime behavior — not security, guest routing, builtin deny, flow pilot, or
guest-output registration.

## Before this refactor (roughly)

To add an agent you had to discover and edit many hardcoded touchpoints, for
example:

- `app/core/security.py` (`USER_RUNNABLE_MISSIONS`)
- `app/routers/guest.py` (`GUEST_MISSION_ID`)
- `runtime/deep_agent.py` (`NO_BUILTIN_TOOL_MISSIONS`)
- `runtime/subgraph/flow.py` (`FLOW_PILOT_MISSIONS`)
- `runtime/hooks/builtin.py` (`_BUILTIN_DENY_MISSIONS` + deny message)
- `runtime/tools/guest_channel_tools.py` (capability literal)
- Duplicate prompts / MLflow registration scripts
- Mission + agent seed YAML, skills map, MCP env JSON, manual verifier scripts

That was on the order of **12–15 edits across Python, YAML, env, and docs**
before the first seed — easy to miss a gate and get silent misconfiguration.

## Still code-keyed (not in `harness:` yet)

These remain allowlists or literals; extending them still means editing Python
or env unless a later harness pass covers them:

| Concern | Location |
|---------|----------|
| Capability subgraph (`fault_analysis` only) | `runtime/mission_graph.py` (`CAPABILITY_BUILDERS`, `capability_to_graph`) |
| Admin Slack notifications | `runtime/notifications/slack.py` (`ADMIN_MISSION_ID`) |
| Admin prompt fallback without AgentVersion | `mission_graph` (`agent_id == "admin_bot"`) |
| Tool → capability registration (fault/admin/Hotel MCP) | `runtime/tools/*`, `MCP_TOOL_CAPABILITIES` in `.env` |
| Manual / eval script mission constants | `scripts/manual/*.py`, `scripts/eval_mission.py`, etc. |

New agents that reuse existing capabilities and skills generally only need
YAML + seed; a **new capability subgraph** or **new registered tool family**
still needs registry/code work as today.