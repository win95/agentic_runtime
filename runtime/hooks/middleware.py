"""Hook middleware — the in-agent half of the hooks system.

``HookMiddleware`` is a ``langchain`` ``AgentMiddleware`` (verified
against the installed ``langchain`` package: ``before_agent`` /
``after_agent`` / ``wrap_tool_call`` / ``after_model`` with
``@hook_config(can_jump_to=["model"])`` for the jump back to the model).
It translates hook decisions into agent behavior:

- ``before_agent`` — SessionStart: handlers may return
  ``additional_context``, injected as a system message (the Hooks v2
  load-context pattern).
- ``before_model`` — context budget: trims the oldest complete
  conversation turns (AIMessage + its ToolMessages) when the turn
  total exceeds the effective history budget (HISTORY_TOKENS minus
  the history/long-term-memory blocks rendered for the run,
  ``reserved_history_tokens``), via ``RemoveMessage`` — system/human
  messages are never removed. Before the removal goes out, the
  trimmed turns are encoded into long-term memory
  (``runtime/long_term_memory.py``) so a later execution can
  retrieve them by keyword — encode-then-remove, never the reverse.
  A trimming failure never fails the run.
- ``wrap_tool_call`` — PreToolUse: a ``deny`` returns an error
  ``ToolMessage`` and the tool never runs. PostToolUse /
  PostToolUseFailure: ``feedback`` text is appended to the tool result
  the agent sees. A tool that raises is turned into an error
  ``ToolMessage`` instead of crashing the run (same "error evidence is
  still evidence" rule as the capability graphs).
- ``after_model`` — Stop: fires only when the agent's last message has
  no tool calls (it is about to stop). A ``continue`` decision returns
  ``{"messages": [nudge], "jump_to": "model"}`` — one more model turn,
  capped per run.
- ``after_agent`` — SessionEnd for the agent run (the worker also
  dispatches SessionEnd for the whole execution).

One instance is built per run in ``deep_agent_node`` and keeps the
run's hook record in ``events`` (one dict per dispatch: event, tool,
decision actions, message) so the node can surface interventions in
the execution state. Like the token/audit handler, a middleware
failure never fails the run — and like it, the middleware touches
the DB only through one best-effort writer: the long-term-memory
encoder, whose failure is logged while the trim still goes ahead.
"""

from __future__ import annotations

import logging
import threading

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from runtime.context_budget import (
    HISTORY_TOKENS,
    group_turns,
    message_tokens,
    trim_history,
)
from runtime.hooks.events import HookContext, HookEvent
from runtime.hooks.registry import HookRegistry
from runtime.long_term_memory import encode_texts, turn_text

logger = logging.getLogger(__name__)

# Subagent hooks: tool-level events only — derived from BUILTIN_HOOKS so new
# PreToolUse/PostToolUse handlers apply automatically.
SUBAGENT_EXCLUDED_HOOK_EVENTS = frozenset(
    {
        HookEvent.SESSION_START,
        HookEvent.SESSION_END,
        HookEvent.USER_PROMPT_SUBMIT,
        HookEvent.STOP,
        HookEvent.SUBAGENT_START,
        HookEvent.SUBAGENT_STOP,
        HookEvent.NOTIFICATION,
        HookEvent.PERMISSION_REQUEST,
        HookEvent.PRE_COMPACT,
    }
)
SUBAGENT_TOOL_HOOK_EVENTS = frozenset(
    {
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_USE,
        HookEvent.POST_TOOL_USE_FAILURE,
    }
)


def builtin_subagent_hook_names() -> list[str]:
    """Built-in hook names safe for subagents (tool enforcement only)."""
    from runtime.hooks.builtin import BUILTIN_HOOKS

    names: list[str] = []
    for name, pairs in BUILTIN_HOOKS.items():
        events = {event for event, _handler in pairs}
        if events & SUBAGENT_EXCLUDED_HOOK_EVENTS:
            continue
        if events & SUBAGENT_TOOL_HOOK_EVENTS:
            names.append(name)
    return names


class SharedRunCounters:
    """Thread-safe counters shared by main agent + subagent middleware."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tool_calls_made = 0

    def increment_tool_calls(self) -> int:
        with self._lock:
            self.tool_calls_made += 1
            return self.tool_calls_made

    def snapshot_tool_calls(self) -> int:
        with self._lock:
            return self.tool_calls_made


class HookMiddleware(AgentMiddleware):
    """Per-run hook middleware for one Deep Agent invocation."""

    def __init__(
        self,
        registry: HookRegistry,
        *,
        execution_id: str = "",
        mission_id: str = "",
        tenant: str = "default",
        agent_id: str = "",
        mission: dict | None = None,
        input_data: dict | None = None,
        evidence_count: int = 0,
        max_continues: int = 3,
        session_id: str = "",
        allowed_tools: list | None = None,
        capabilities: list | None = None,
        reserved_history_tokens: int = 0,
        history_budget_tokens: int | None = None,
        shared_counters: SharedRunCounters | None = None,
        events_sink: list | None = None,
        subagent_mode: bool = False,
    ):
        self.registry = registry
        self.execution_id = execution_id
        self.mission_id = mission_id
        self.tenant = tenant
        self.agent_id = agent_id
        self.mission = mission or {}
        self.input_data = input_data or {}
        self.evidence_count = evidence_count
        self.max_continues = max_continues
        self.session_id = session_id
        self.allowed_tools = list(allowed_tools or [])
        self.capabilities = list(capabilities or [])
        # History/LTM blocks rendered into the system prompt for this
        # run (see runtime/context_budget.py): the trim budget for the
        # run's own turns is the mission's effective history budget
        # (HISTORY_TOKENS, or HISTORY + RETRIEVAL for missions without
        # pre-retrieval — see effective_history_tokens) minus these,
        # so the history slice is shared instead of overspent. 0 keeps
        # the old behaviour (turns alone may use the whole slice).
        self.reserved_history_tokens = max(0, int(reserved_history_tokens or 0))
        self.history_budget_tokens = (
            int(history_budget_tokens)
            if history_budget_tokens
            else HISTORY_TOKENS
        )
        self.shared_counters = shared_counters or SharedRunCounters()
        self.events_sink = events_sink
        self.subagent_mode = subagent_mode
        self.tool_calls_made = 0
        self.continues = 0
        self.trimmed_messages = 0
        self.trimmed_tokens = 0
        self.encoded_turns = 0
        self.events: list[dict] = []

    @classmethod
    def for_subagent(cls, parent: HookMiddleware) -> HookMiddleware:
        """Tool-enforcement middleware for the general-purpose subagent."""
        sub_registry = parent.registry.with_only(builtin_subagent_hook_names())
        return cls(
            sub_registry,
            execution_id=parent.execution_id,
            mission_id=parent.mission_id,
            tenant=parent.tenant,
            agent_id=parent.agent_id,
            mission=parent.mission,
            input_data=parent.input_data,
            evidence_count=0,
            max_continues=0,
            session_id=parent.session_id,
            allowed_tools=list(parent.allowed_tools),
            capabilities=list(parent.capabilities),
            reserved_history_tokens=0,
            history_budget_tokens=HISTORY_TOKENS,
            shared_counters=parent.shared_counters,
            events_sink=parent.events,
            subagent_mode=True,
        )

    # -- helpers ------------------------------------------------------
    def _base_context(self) -> HookContext:
        return HookContext(
            execution_id=self.execution_id,
            mission_id=self.mission_id,
            tenant=self.tenant,
            agent_id=self.agent_id,
            mission=self.mission,
            input=self.input_data,
            session_id=self.session_id,
            allowed_tools=list(self.allowed_tools),
            capabilities=list(self.capabilities),
        )

    def _record(self, event: HookEvent, decisions, **extra) -> None:
        record = {
            "event": event.value,
            "decisions": [d.action for d in decisions],
            "messages": [d.message for d in decisions if d.message],
            **extra,
        }
        target = self.events_sink if self.events_sink is not None else self.events
        target.append(record)

    # -- SessionStart --------------------------------------------------
    def before_agent(self, state, runtime):  # noqa: ARG002 - runtime unused
        if self.subagent_mode:
            return None
        context = self._base_context()
        decisions = self.registry.dispatch(HookEvent.SESSION_START, context)
        self._record(HookEvent.SESSION_START, decisions)
        extra = "\n\n".join(
            d.additional_context for d in decisions if d.additional_context
        )
        if extra:
            return {"messages": [SystemMessage(content=extra)]}
        return None

    # -- Context budget: trim history before each model call ----------
    def before_model(self, state, runtime):  # noqa: ARG002 - runtime unused
        try:
            messages = (state or {}).get("messages") or []
            budget = max(
                0, self.history_budget_tokens - self.reserved_history_tokens
            )
            remove = trim_history(messages, budget)
            if not remove:
                return None
            # Encode-then-remove on the main agent only — subagent turns
            # can contain untrusted web/tool text and must not enter LTM.
            if not self.subagent_mode:
                self.encoded_turns += encode_texts(
                    self.session_id,
                    self.execution_id,
                    [turn_text(turn) for turn in group_turns(remove)],
                    source="trim",
                )
            remove_ids = [getattr(m, "id", "") or "" for m in remove]
            removed_tokens = sum(message_tokens(m) for m in remove)
            self.trimmed_messages += len(remove)
            self.trimmed_tokens += removed_tokens
            logger.info(
                "Context budget: trimmed %d message(s) (~%d tokens) "
                "to fit the %d-token history budget "
                "(%d reserved for history/LTM blocks)",
                len(remove),
                removed_tokens,
                budget,
                self.reserved_history_tokens,
            )
            # Messages carry ids once they are in the graph state; a
            # missing id cannot be removed by id, so it is left alone.
            return {
                "messages": [RemoveMessage(id=mid) for mid in remove_ids if mid]
            }
        except Exception as exc:  # noqa: BLE001 — trimming never breaks a run
            logger.debug("Context budget trim failed (non-fatal): %s", exc)
            return None

    # -- Pre/PostToolUse + PostToolUseFailure --------------------------
    def wrap_tool_call(self, request, handler):
        tool_call = request.tool_call or {}
        tool_name = str(tool_call.get("name") or "")
        tool_input = dict(tool_call.get("args") or {})
        tool_call_id = str(tool_call.get("id") or "")

        pre = self._base_context()
        pre.tool_name = tool_name
        pre.tool_input = tool_input
        decisions = self.registry.dispatch(HookEvent.PRE_TOOL_USE, pre)
        self._record(HookEvent.PRE_TOOL_USE, decisions, tool=tool_name)
        for decision in decisions:
            if decision.action == "deny":
                logger.info("Hook denied tool call %s: %s", tool_name, decision.message)
                return ToolMessage(
                    content=f"Hook denied tool call '{tool_name}': {decision.message}",
                    tool_call_id=tool_call_id,
                    status="error",
                )

        try:
            result = handler(request)
        except Exception as exc:  # noqa: BLE001 — tool errors are evidence
            self.shared_counters.increment_tool_calls()
            if not self.subagent_mode:
                self.tool_calls_made += 1
            failure = self._base_context()
            failure.tool_name = tool_name
            failure.tool_input = tool_input
            failure.error = str(exc)
            fail_decisions = self.registry.dispatch(
                HookEvent.POST_TOOL_USE_FAILURE, failure
            )
            self._record(
                HookEvent.POST_TOOL_USE_FAILURE, fail_decisions, tool=tool_name
            )
            feedback = "\n\n".join(
                d.message for d in fail_decisions if d.action == "feedback" and d.message
            )
            content = f"Tool '{tool_name}' failed: {exc}"
            if feedback:
                content += f"\n\n{feedback}"
            return ToolMessage(content=content, tool_call_id=tool_call_id, status="error")

        self.shared_counters.increment_tool_calls()
        if not self.subagent_mode:
            self.tool_calls_made += 1
        result_content = result.content if isinstance(result, ToolMessage) else ""
        is_error = isinstance(result, ToolMessage) and result.status == "error"
        post_event = (
            HookEvent.POST_TOOL_USE_FAILURE if is_error else HookEvent.POST_TOOL_USE
        )
        post = self._base_context()
        post.tool_name = tool_name
        post.tool_input = tool_input
        post.tool_response = result_content if isinstance(result_content, str) else ""
        post.error = post.tool_response if is_error else ""
        post_decisions = self.registry.dispatch(post_event, post)
        self._record(post_event, post_decisions, tool=tool_name)
        feedback = "\n\n".join(
            d.message for d in post_decisions if d.action == "feedback" and d.message
        )
        if feedback and isinstance(result, ToolMessage) and isinstance(
            result.content, str
        ):
            return ToolMessage(
                content=f"{result.content}\n\n{feedback}",
                tool_call_id=result.tool_call_id,
                status=result.status,
            )
        return result

    # -- Stop (Ralph loop) ---------------------------------------------
    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):  # noqa: ARG002 - runtime unused
        if self.subagent_mode:
            return None
        messages = (state or {}).get("messages") or []
        last = messages[-1] if messages else None
        # Only the stop case reaches this hook: while the agent still
        # has tool calls, it continues to the tools node on its own.
        if last is None or getattr(last, "tool_calls", None):
            return None
        context = self._base_context()
        context.messages = list(messages)
        context.data["tool_calls_made"] = self.shared_counters.snapshot_tool_calls()
        context.data["evidence_count"] = self.evidence_count
        context.data["continues"] = self.continues
        context.data["max_continues"] = self.max_continues
        decisions = self.registry.dispatch(HookEvent.STOP, context)
        self._record(HookEvent.STOP, decisions)
        for decision in decisions:
            if decision.action == "continue" and self.continues < self.max_continues:
                self.continues += 1
                logger.info(
                    "Hook Stop continues turn %d/%d: %s",
                    self.continues,
                    self.max_continues,
                    decision.message,
                )
                return {
                    "messages": [HumanMessage(content=decision.message)],
                    "jump_to": "model",
                }
        return None

    # -- SessionEnd (agent run) -----------------------------------------
    def after_agent(self, state, runtime):  # noqa: ARG002 - unused
        if self.subagent_mode:
            return None
        context = self._base_context()
        context.reason = "agent_finished"
        decisions = self.registry.dispatch(HookEvent.SESSION_END, context)
        self._record(HookEvent.SESSION_END, decisions)
        return None
