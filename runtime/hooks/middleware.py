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
  total exceeds HISTORY_TOKENS, via ``RemoveMessage`` — system/human
  messages are never removed. A trimming failure never fails the run.
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
failure never fails the run — but unlike that handler it does not
touch the DB at all.
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from runtime.context_budget import HISTORY_TOKENS, message_tokens, trim_history
from runtime.hooks.events import HookContext, HookEvent
from runtime.hooks.registry import HookRegistry

logger = logging.getLogger(__name__)


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
        self.tool_calls_made = 0
        self.continues = 0
        self.trimmed_messages = 0
        self.trimmed_tokens = 0
        self.events: list[dict] = []

    # -- helpers ------------------------------------------------------
    def _base_context(self) -> HookContext:
        return HookContext(
            execution_id=self.execution_id,
            mission_id=self.mission_id,
            tenant=self.tenant,
            agent_id=self.agent_id,
            mission=self.mission,
            input=self.input_data,
        )

    def _record(self, event: HookEvent, decisions, **extra) -> None:
        self.events.append(
            {
                "event": event.value,
                "decisions": [d.action for d in decisions],
                "messages": [d.message for d in decisions if d.message],
                **extra,
            }
        )

    # -- SessionStart --------------------------------------------------
    def before_agent(self, state, runtime):  # noqa: ARG002 - runtime unused
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
            remove = trim_history(messages, HISTORY_TOKENS)
            if not remove:
                return None
            remove_ids = [getattr(m, "id", "") or "" for m in remove]
            removed_tokens = sum(message_tokens(m) for m in remove)
            self.trimmed_messages += len(remove)
            self.trimmed_tokens += removed_tokens
            logger.info(
                "Context budget: trimmed %d message(s) (~%d tokens) "
                "to fit the %d-token history budget",
                len(remove),
                removed_tokens,
                HISTORY_TOKENS,
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
        messages = (state or {}).get("messages") or []
        last = messages[-1] if messages else None
        # Only the stop case reaches this hook: while the agent still
        # has tool calls, it continues to the tools node on its own.
        if last is None or getattr(last, "tool_calls", None):
            return None
        context = self._base_context()
        context.messages = list(messages)
        context.data["tool_calls_made"] = self.tool_calls_made
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
        context = self._base_context()
        context.reason = "agent_finished"
        decisions = self.registry.dispatch(HookEvent.SESSION_END, context)
        self._record(HookEvent.SESSION_END, decisions)
        return None
