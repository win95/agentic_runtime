"""LarkAgent — the Lark agent's own policy (agent_lark_me, Step 3).

One rule: a Lark tool call may only touch this run's own chat — the
chat the user is talking in (``input["chat_id"]``) or a group this run
created. Everything else is refused with a structured error dict, so
the agent sees it as a tool result, not a crash.

Mechanism (mirrors the hotel agent's guest scope, but per-run, not
per-session): the agent instance is fresh per run (``resolve_agent``),
and the Mission Graph sets ``execution_input`` on it before the tool
list is built. ``scope_tool`` wraps each Lark tool with a call-time
check against that input:

- the allowed chats start as ``{input chat_id}`` and grow by every
  group ``lark_create_group`` creates in this run;
- ``lark_update_group`` (rename) is allowed for the run's own chat
  (the resolve run's input chat IS the incident group) and for
  groups this run created — never for any other chat;
- ``lark_send_message`` to a user (``user_id`` / ``open_id``) is allowed
  only to the input sender;
- ``lark_reply_message`` may only reply in a thread this run takes
  part in: the input message / root, or a message this run sent.

Remind mode (``input["mode"] == "remind_tasks"``, §3.5 of the
harness-loops design): the run has no input chat — its recipients
come from ``input["recipients"]`` instead:

- ``"task_assignees"`` — ``lark_send_message`` to a user is allowed
  only for ids this run itself read from the Task API (assignees of
  todo tasks returned by ``lark_list_tasks`` / ``lark_get_task``).
  The recipient set grows by the SAME post-tool result-feedback
  mechanism as ``lark_create_group`` grows the allowed chats: the
  task-tool wrapper invokes the tool, then harvests member ids from
  the successful result into ``_lark_reminder_recipients``.
- ``{"type": "fixed_user", "open_id": ...}`` — that one user is
  seeded into the reminder set at init, like the sender is.
- ``{"type": "fixed_chat", "chat_id": ...}`` — that one chat is
  seeded into the allowed chats at init, like the input chat is.

The wrapper is a copy — the registered tool itself is never mutated
(same pattern as the asset scope in ``runtime/tools/registry.py``).
No ``guest_reply_tool`` override: replies go through the
``lark_reply_message`` registry tool (one send path, D-LARK-09), not
through the guest ``reply_to_guest`` tool.
"""

from __future__ import annotations

from runtime.agents.base import BaseAgent
from runtime.agents.registry import register_agent_class

LARK_TOOLS = frozenset(
    {
        "lark_create_group",
        "lark_send_message",
        "lark_get_group",
        "lark_upload_image",
        "lark_add_members",
        "lark_reply_message",
        "lark_get_history",
        "lark_update_group",
    }
)

# Tools whose target is a chat_id argument.
_CHAT_TARGET_TOOLS = frozenset(
    {"lark_get_group", "lark_add_members", "lark_get_history"}
)

# Task-reading tools (capability ``lark_tasks``, built by the P2
# tools work — see the task-remind flow in skills/agent-lark).
# Read-only: no chat target, so the wrapper passes them through and
# — in remind mode only — harvests assignee ids from their results
# into _lark_reminder_recipients (same post-tool result-feedback
# mechanism as lark_create_group, below).
LARK_TASK_TOOLS = frozenset(
    {"lark_list_tasklists", "lark_list_tasks", "lark_get_task"}
)

# Calendar free/busy (capability ``lark_calendar``) — sender calendar only.
LARK_CALENDAR_TOOLS = frozenset({"lark_calendar_freebusy"})

# Member-id keys accepted when harvesting assignees from a task
# result. The Task API v2 task shape carries members as objects with
# an ``id`` (an open_id) and may also surface open_id / user_id /
# union_id directly; accept all of them, never the task's own guid.
_TASK_MEMBER_ID_KEYS = ("open_id", "user_id", "union_id", "id", "member_id")


def _tasks_from_result(result) -> list:
    """Extract the task dicts from a task-tool result, any shape.

    Accepts a list of tasks, a single task dict, or a dict wrapping
    the tasks under ``tasks`` / ``items`` / ``task`` / ``data``
    (nested one level). An error result yields no tasks.
    """
    if isinstance(result, list):
        return [t for t in result if isinstance(t, dict)]
    if not isinstance(result, dict) or "error" in result:
        return []
    for key in ("tasks", "items"):
        value = result.get(key)
        if isinstance(value, list):
            return [t for t in value if isinstance(t, dict)]
    for key in ("task", "data"):
        value = result.get(key)
        if isinstance(value, dict):
            nested = _tasks_from_result(value)
            if nested:
                return nested
            if key == "task":
                return [value]
            return []
        if isinstance(value, list):
            return [t for t in value if isinstance(t, dict)]
    if "members" in result or "summary" in result or "guid" in result:
        return [result]
    return []


class LarkAgent(BaseAgent):
    """The Lark bot — one chat per run, groups only it creates."""

    def __init__(self, agent_id: str, mission: dict, agent_version=None):
        super().__init__(agent_id, mission, agent_version)
        # Per-run scope state (fresh instance per run — nothing here
        # survives into another run). Seeded lazily on the first
        # scope_tool call, from execution_input.
        self._lark_scope_seeded = False
        self._lark_allowed_chats: set[str] = set()
        self._lark_created_chats: set[str] = set()
        self._lark_own_message_ids: set[str] = set()
        self._lark_sender_ids: set[str] = set()
        self._lark_input_message_ids: set[str] = set()
        # Remind mode only: user ids this run may DM as task
        # assignees — seeded from fixed_user input, or grown in-run
        # from Task API results (see _harvest_task_assignees).
        self._lark_reminder_recipients: set[str] = set()

    def _seed_lark_scope(self) -> None:
        """Seed the allowed set from this run's input (once per run)."""
        if self._lark_scope_seeded:
            return
        self._lark_scope_seeded = True
        inp = self.execution_input or {}
        chat_id = inp.get("chat_id")
        if isinstance(chat_id, str) and chat_id:
            self._lark_allowed_chats.add(chat_id)
        sender = inp.get("sender_id") or {}
        if isinstance(sender, dict):
            for key in ("open_id", "user_id", "union_id"):
                value = sender.get(key)
                if isinstance(value, str) and value:
                    self._lark_sender_ids.add(value)
        for key in ("message_id", "root_id"):
            value = inp.get(key)
            if isinstance(value, str) and value:
                self._lark_input_message_ids.add(value)
        if self._is_remind_tasks():
            recipients = inp.get("recipients")
            if isinstance(recipients, dict):
                rtype = recipients.get("type")
                if rtype == "fixed_chat" or (
                    rtype is None and "chat_id" in recipients
                ):
                    chat_id = recipients.get("chat_id")
                    if isinstance(chat_id, str) and chat_id:
                        self._lark_allowed_chats.add(chat_id)
                if rtype == "fixed_user" or (
                    rtype is None
                    and ("open_id" in recipients or "user_id" in recipients)
                ):
                    for key in ("open_id", "user_id", "union_id"):
                        value = recipients.get(key)
                        if isinstance(value, str) and value:
                            self._lark_reminder_recipients.add(value)
                # "task_assignees" seeds nothing: the set grows
                # in-run from Task API results (see scope_tool).

    def _is_remind_tasks(self) -> bool:
        """True when this run is a scheduled task reminder."""
        return (self.execution_input or {}).get("mode") == "remind_tasks"

    def _remind_is_task_assignees(self) -> bool:
        """True when remind recipients are the tasks' own assignees.

        ``recipients`` absent defaults to ``"task_assignees"`` (the
        §7(a) default for the first loop — the skill reads the same
        default, so scope and skill agree).
        """
        if not self._is_remind_tasks():
            return False
        recipients = (self.execution_input or {}).get("recipients")
        return recipients is None or recipients == "task_assignees"

    def _harvest_task_assignees(self, result) -> None:
        """Add assignees of todo tasks in a task-tool result.

        Only called in remind mode with recipients == task_assignees.
        The result shape is the task tools' contract (task fields
        guid/summary/due/completed_at/status/members/url): a task
        list under ``tasks`` / ``items`` / ``data``, or a single task
        under ``task`` / ``data`` / at the top level. Member entries
        may be plain id strings or dicts. Only members of tasks whose
        status is todo (or absent — the tool already filtered) are
        admitted, so a done or future task never authorizes a DM.
        A result carrying ``"error"`` harvests nothing.
        """
        tasks = _tasks_from_result(result)
        for task in tasks:
            if not isinstance(task, dict):
                continue
            status = task.get("status")
            if status is not None and str(status).lower() != "todo":
                continue
            members = task.get("members")
            if members is None:
                members = task.get("assignees") or task.get("assignee")
            if isinstance(members, (str, dict)):
                members = [members]
            if not isinstance(members, list):
                continue
            for member in members:
                if isinstance(member, str):
                    if member:
                        self._lark_reminder_recipients.add(member)
                elif isinstance(member, dict):
                    for key in _TASK_MEMBER_ID_KEYS:
                        value = member.get(key)
                        if isinstance(value, str) and value:
                            self._lark_reminder_recipients.add(value)

    def _refuse(self, tool_name: str, detail: str) -> dict:
        return {
            "error": f"{tool_name}: {detail}; refused",
            "tool": tool_name,
            "stub": False,
            "provenance": f"lark:{tool_name}",
        }

    def scope_tool(self, tool, session_id: str, mission_id: str = ""):
        """Wrap a Lark tool with this run's chat scope.

        Tools outside ``LARK_TOOLS`` and ``LARK_TASK_TOOLS`` pass
        through unchanged.
        """
        name = getattr(tool, "name", "") or ""
        if (
            name not in LARK_TOOLS
            and name not in LARK_TASK_TOOLS
            and name not in LARK_CALENDAR_TOOLS
        ):
            return tool
        self._seed_lark_scope()

        def _guarded(**kwargs):
            if name in LARK_CALENDAR_TOOLS:
                open_id = next(
                    (
                        sid
                        for sid in self._lark_sender_ids
                        if isinstance(sid, str) and sid.startswith("ou_")
                    ),
                    next(iter(self._lark_sender_ids), None),
                )
                if not isinstance(open_id, str) or not open_id:
                    return self._refuse(
                        name,
                        "no sender open_id in this run's input — only the "
                        "chatting user's calendar may be queried",
                    )
                kwargs.pop("user_open_id", None)
                from runtime.tools import lark_calendar_tools

                return lark_calendar_tools.run_lark_calendar_freebusy(
                    user_open_id=open_id,
                    range_name=kwargs.get("range", "today"),
                    start=kwargs.get("start"),
                    end=kwargs.get("end"),
                    timezone=kwargs.get("timezone", "Asia/Saigon"),
                )
            # Task reads have no chat target — allowed in any mode
            # (read-only). In remind mode with task_assignees
            # recipients, the successful result grows the reminder
            # set: the same post-tool result-feedback mechanism
            # lark_create_group uses below to grow the allowed
            # chats (invoke, then read chat_id / member ids out of
            # the result and add them to run state).
            if name in LARK_TASK_TOOLS:
                result = tool.invoke(kwargs)
                if self._remind_is_task_assignees():
                    self._harvest_task_assignees(result)
                return result
            # Reads / member adds target a chat_id argument.
            if name in _CHAT_TARGET_TOOLS:
                target = kwargs.get("chat_id")
                if not isinstance(target, str) or target not in self._lark_allowed_chats:
                    return self._refuse(
                        name, f"chat_id '{target}' is outside this run's chat"
                    )
                return tool.invoke(kwargs)
            if name == "lark_update_group":
                target = kwargs.get("chat_id")
                # Rename is allowed for the run's own chat (the
                # resolve run's input chat IS the incident group)
                # and for groups this run created — nothing else.
                # _lark_allowed_chats is exactly {input chat} +
                # created groups, so one membership check covers
                # both (the created set is checked explicitly so
                # the intent stays readable if the sets diverge).
                if not isinstance(target, str) or (
                    target not in self._lark_created_chats
                    and target not in self._lark_allowed_chats
                ):
                    return self._refuse(
                        name,
                        f"chat_id '{target}' is not this run's chat "
                        "and was not created by this run — only the "
                        "run's own chat or groups it created may be "
                        "renamed",
                    )
                return tool.invoke(kwargs)
            if name == "lark_send_message":
                receive_type = kwargs.get("receive_id_type", "chat_id")
                receive_id = kwargs.get("receive_id")
                if receive_type == "chat_id":
                    if (
                        not isinstance(receive_id, str)
                        or receive_id not in self._lark_allowed_chats
                    ):
                        return self._refuse(
                            name, f"chat_id '{receive_id}' is outside this run's chat"
                        )
                elif receive_type in ("user_id", "open_id"):
                    if not isinstance(receive_id, str) or (
                        receive_id not in self._lark_sender_ids
                        and receive_id not in self._lark_reminder_recipients
                    ):
                        return self._refuse(
                            name,
                            f"user '{receive_id}' is not the user this run "
                            "is talking to",
                        )
                result = tool.invoke(kwargs)
                if isinstance(result, dict) and "error" not in result:
                    mid = result.get("message_id")
                    if isinstance(mid, str) and mid:
                        self._lark_own_message_ids.add(mid)
                return result
            if name == "lark_reply_message":
                target = kwargs.get("message_id")
                allowed = self._lark_input_message_ids | self._lark_own_message_ids
                if not isinstance(target, str) or target not in allowed:
                    return self._refuse(
                        name,
                        f"message_id '{target}' is not a thread this run "
                        "takes part in",
                    )
                result = tool.invoke(kwargs)
                if isinstance(result, dict) and "error" not in result:
                    mid = result.get("message_id")
                    if isinstance(mid, str) and mid:
                        self._lark_own_message_ids.add(mid)
                return result
            if name == "lark_create_group":
                result = tool.invoke(kwargs)
                if isinstance(result, dict) and "error" not in result:
                    chat_id = result.get("chat_id")
                    if isinstance(chat_id, str) and chat_id:
                        self._lark_created_chats.add(chat_id)
                        self._lark_allowed_chats.add(chat_id)
                return result
            # lark_upload_image has no chat target — allowed.
            return tool.invoke(kwargs)

        return tool.model_copy(update={"func": _guarded})


register_agent_class("agent_lark_me", LarkAgent)
