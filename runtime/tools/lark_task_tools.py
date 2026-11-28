"""Lark Task tools (Layer 5) — for the agent_lark_me task-reminder flow.

Native Lark Tasks (Task API v2) via the official ``lark-oapi`` SDK,
wrapped by ``runtime/integrations/lark_client.py`` — same lazy-client
/ structured-error pattern as ``lark_tools.py``: importing this module
never fails when the SDK or credentials are missing; the agent gets a
structured error dict instead.

Capability: ``lark_tasks``. All three tools are read-only. The tools
are only visible to missions that declare the ``lark_tasks``
capability (registry ``check``); read tools are not gated by the
agent's ``allowed_tools`` (Layer 1 read-open — see
``runtime/tools/registry.py``), but the mission capability gate still
applies, so add ``lark_tasks`` to the mission's ``capabilities`` to
enable them. No ``tool_budget`` entry is needed for reads.

"In-process" (design doc §3.6) = ``status == "todo"`` among the
unfinished tasks ``lark_list_tasks`` returns by default
(``completed=False``); the due filter (overdue / due today) is applied
by the skill flow, not by these tools. Due is returned in two forms:
``due`` (UTC ISO 8601 string, e.g. "2026-11-28T02:00:00+00:00") and
``due_ms`` (raw millisecond timestamp, for exact due comparison);
either is None when the task has no due date. Member ids are open_ids
(the client passes ``user_id_type="open_id"``) — the id to DM with
``lark_send_message`` (``receive_id_type="open_id"``).
"""

from __future__ import annotations

from langchain_core.tools import tool

from runtime.integrations import lark_client
from runtime.tools.registry import registry


def _err(name: str, result: dict) -> dict:
    """Attach the tool name + provenance to a client error dict."""
    out = dict(result)
    out.setdefault("tool", name)
    out.setdefault("stub", False)
    out.setdefault("provenance", f"lark:{name.removeprefix('lark_')}")
    return out


@tool
def lark_list_tasklists(limit: int = 100) -> dict:
    """List the Lark tasklists the agent's app can read.

    Read-only. Follows pages until ``limit`` tasklists are collected
    (capped at 100, same spirit as ``lark_get_history``) or the pages
    end. Returns {"tasklists": [...], "count": n, "truncated": bool};
    each tasklist is {guid, name, creator, owner, members, url,
    created_at, updated_at}. Use first in the reminder flow to learn
    which tasklists exist (scope: all_tasklists).
    """
    result = lark_client.list_tasklists(limit=limit)
    if "error" in result:
        return _err("lark_list_tasklists", result)
    return {**result, "stub": False, "provenance": "lark:list_tasklists"}


@tool
def lark_list_tasks(completed: bool = False, limit: int = 100) -> dict:
    """List Lark tasks (unfinished by default), paginated to a cap.

    Read-only. ``completed=False`` (the default) returns only
    unfinished tasks — the reminder flow's "in-process" set is
    ``status == "todo"`` among these. Follows pages until ``limit``
    tasks are collected (capped at 100, same spirit as
    ``lark_get_history``) or the pages end. Returns {"tasks": [...],
    "count": n, "truncated": bool, "completed": bool}; each task is
    {guid, summary, due (ISO string), due_ms, due_is_all_day, status,
    completed_at, members, creator, url, tasklists, created_at,
    updated_at}. Members carry open_ids (``members[].id``) for the
    reminder DM. Pass ``completed=True`` to list finished tasks.
    """
    result = lark_client.list_tasks(completed=completed, limit=limit)
    if "error" in result:
        return _err("lark_list_tasks", result)
    return {**result, "stub": False, "provenance": "lark:list_tasks"}


@tool
def lark_get_task(task_guid: str) -> dict:
    """Read one Lark task by guid.

    Read-only. Returns {"task": {...}} in the same shape as one entry
    of ``lark_list_tasks`` (guid, summary, due, status, members, url,
    ...). Use to re-read a task before reminding about it, or to get
    the ``url`` / members of a task found elsewhere.
    """
    result = lark_client.get_task(task_guid=task_guid)
    if "error" in result:
        return _err("lark_get_task", result)
    return {**result, "stub": False, "provenance": "lark:get_task"}


# Register the 3 tools under the lark_tasks capability.
# All read-only (permission lark:read); no destructive targets, no
# approval gate — reads are the Layer 1 read-open set.
registry.register(
    lark_list_tasklists, capability="lark_tasks", permission="lark:read",
    timeout_seconds=30, read_only=True,
)
registry.register(
    lark_list_tasks, capability="lark_tasks", permission="lark:read",
    timeout_seconds=60, read_only=True,
)
registry.register(
    lark_get_task, capability="lark_tasks", permission="lark:read",
    timeout_seconds=30, read_only=True,
)
