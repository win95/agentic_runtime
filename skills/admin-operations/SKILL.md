---
name: admin-operations
description: Administer the agentic-runtime platform with the admin tools — inspect executions, missions and agents, check dependency health, and manage missions and executions. Read before write; destructive actions need the exact id.
keywords: admin mission execution agent health cancel dispatch platform tenant deploy
---

# Admin Operations

## When to Use
- The mission is `system_admin`: the request asks about the platform itself — executions, missions, agents, or dependency health — or asks to change them.

## Steps
1. Read before write: `get_executions` / `get_missions` / `list_agents` first, then act on what they return.
2. Check health with `system_health` when the request is about the platform being up or down; report each dependency (database, redis, qdrant, mlflow, minio) separately.
3. Destructive actions (`delete_mission`, `cancel_execution`) require the exact id named in the request. If the id is missing or ambiguous, refuse that action and ask for the exact id.
4. `delete_mission` is refused when the mission still has executions (409 rule) — report the execution count and do not cascade.
5. `dispatch_execution` returns a QUEUED execution id; check its outcome later with `get_executions` (pass the execution_id) instead of assuming it succeeded.

## What NOT to Do
- Never invent ids, counts, or statuses — report exactly what the tools return, including their errors.
- Never work around a tool refusal by guessing.
- Never delete a mission's execution history to force a delete.
