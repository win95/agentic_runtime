# Archived agents

Agents disabled and removed from the running app on **2026-11-27**,
on tran's decision ("Chốt, disable cả 2 theo kế hoạch"). Their code
lives here, out of the app's import/seed/skill paths, so nothing
starts with the app — but everything is restorable.

## What is here

Each agent folder keeps its files at their **original relative
paths**, so restoring is a plain copy back:

- `fault_investigator/` — the equipment-fault agent + `fault_analysis`
  mission: capability subgraph (`runtime/capabilities/`), fault tools
  (`runtime/tools/`), agent + mission + weekly-schedule YAML, the
  `fault-investigation` skill, the two manual run scripts, and the
  retrieval golden set.
- `hotel_receptionist/` — the front-desk agent + `hotel_reception`
  mission (the public guest API's mission): the agent class with its
  guest scoping (`runtime/agents/`), the guest-channel reply tools,
  agent + mission YAML, the `reception` skill, the manual run script
  + report, the mock hotel MCP server (17 tools), and the three
  hotel-only test files. **Note 2026-11-27:** the guest-channel reply
  tools (`runtime/tools/guest_channel_tools.py`) were copied back to
  live use the same day — they are shared guest infrastructure, not
  hotel-only: `/guest/chat` now routes to `agent_lark_me_mission`
  (option A, `reply_to_guest`). The archive copy stays here as the
  record.

## Why the database needs a separate step

Removing the files alone does **not** disable the agents, for two
reasons (both verified in code before the move):

1. `runtime/agents/registry.py::resolve_agent` falls back to
   `BaseAgent` for any agent id with no registered class, and the
   skill loader silently skips missing skills — so a mission row
   left in the DB would still run, degraded. For hotel that is
   worse than failing: it would run **without** the guest scoping
   `HotelReceptionistAgent` enforced.
2. `make seed` only upserts — it never deletes a mission whose YAML
   is gone.

So after pulling, run the cleanup script (it deletes, in FK order:
execution results → executions → schedules → agent versions →
agents → missions; execution *history* for these two missions goes
with them because `Execution.mission_id` is a foreign key):

```bash
python -m scripts.manual.disable_agents --dry-run   # show what would go
python -m scripts.manual.disable_agents --apply     # delete, then commit
```

Then restart the worker + API. `/guest/chat` no longer answers 503:
since 2026-11-27 it routes to `agent_lark_me_mission` (guest_channel
+ guest_output in its mission YAML, `reply_to_guest` restored to
live use) — the 503 only returns if no mission exposes public chat.

`make seed-demo` keeps working: the two `fault_analysis` demo
executions were removed from `seeds/demo_data.py` in the same change
(the system_admin demo execution stays).

## What was deliberately NOT archived

Shared infrastructure the remaining agents still use: the write
pipeline (`runtime/write_pipeline_runner.py` — used by `BaseAgent`,
the flow subgraph and the skill-map planner), the hybrid-retrieval
modules (fault was the pilot mission, the code is shared), the MCP
adapter + `MCP_SERVER_URL` compose wiring (its gate already no-ops
when no mission uses MCP tools), the guest router itself, the
`infra/repositories` store layer, and the docs (kept as history).

## Restore

```bash
# from the repo root, for one agent (example: hotel):
cp -r archive/agents/hotel_receptionist/* .
# or both:
cp -r archive/agents/fault_investigator/* .
cp -r archive/agents/hotel_receptionist/* .
make seed            # re-creates the mission + agent rows (+ versions)
# if the DB cleanup script was run, re-seed demo data too if wanted:
make seed-demo
# restart worker + api; hotel restores the guest API with it
```

No code change is needed to restore — the three import removals
(`runtime/agents/__init__.py`, `runtime/mission_graph.py`,
`runtime/tools/__init__.py`) are undone by copying the files back,
since the moved modules self-register on import.
