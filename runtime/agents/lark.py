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

    def _refuse(self, tool_name: str, detail: str) -> dict:
        return {
            "error": f"{tool_name}: {detail}; refused",
            "tool": tool_name,
            "stub": False,
            "provenance": f"lark:{tool_name}",
        }

    def scope_tool(self, tool, session_id: str, mission_id: str = ""):
        """Wrap a Lark tool with this run's chat scope.

        Tools outside ``LARK_TOOLS`` pass through unchanged.
        """
        name = getattr(tool, "name", "") or ""
        if name not in LARK_TOOLS:
            return tool
        self._seed_lark_scope()

        def _guarded(**kwargs):
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
                    if (
                        not isinstance(receive_id, str)
                        or receive_id not in self._lark_sender_ids
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
