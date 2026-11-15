"""Shared pieces for the subgraph compilers (``runtime/subgraph/``).

Both compilers — the flow compiler (``flow.py``, Direction C, current)
and the legacy node compiler (``node.py``) — build Deep Agents with a
per-execution tool-call budget + repeat guard, pick tools by name, walk
the map in list order, and read the last text of a message list. Those
pieces live here so neither compiler owns what the other imports; the
legacy node module used to own them, which is backwards now that the
flow compiler is the current path.
"""

from __future__ import annotations

import json
import logging

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from runtime.skills.map_plan import SkillMap

logger = logging.getLogger(__name__)

# Default tool-call budget for one agent execution (5): the node
# compiler's historical default. The step cap counts node executions,
# not tool calls inside a node — an agent runs until the model stops
# calling tools, so a model repeating one failing call (2026-11-13:
# get_booking with a bad id, looped until the LangGraph recursion
# limit) never reaches its NEXT:/FLOW: line and the step cap never sees
# it. This budget is the per-execution backstop. 5 is generous for the
# reception map: the heaviest node (booking-lookup) legitimately needs
# at most find_guest + get_booking; every other node needs 1–2 calls.
DEFAULT_TOOL_CALL_BUDGET = 5


class ToolCallBudgetExceededError(Exception):
    """An agent spent its per-execution tool-call budget."""


def _call_key(name: str, args: dict) -> tuple[str, str]:
    try:
        args_json = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        args_json = repr(sorted(args.items()))
    return (name, args_json)


def _error_in_result(result) -> str | None:
    """Error text if a tool result is a failure, else None.

    Two shapes count as failure: a ToolMessage with status="error"
    (raised tools, hook denials) and a successful result carrying an
    {"error": ...} dict — the mock MCP convention (get_booking returns
    {"error": "Booking not found: ..."} as data). The second shape is
    the one the booking-061 loop hid behind: no exception, so no
    PostToolUseFailure feedback ever fired and the agent just saw a
    normal-looking tool result.
    """
    content = result.content if isinstance(result, ToolMessage) else result
    if isinstance(result, ToolMessage) and result.status == "error":
        return str(content)[:500]
    if isinstance(content, dict) and isinstance(content.get("error"), str):
        return content["error"][:500]
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
            return parsed["error"][:500]
    return None


class ToolCallCap(AgentMiddleware):
    """Per-agent-execution tool-call budget + repeat guard (2026-11-13).

    One instance serves one agent execution (one node agent in the
    legacy node compiler, one flow agent in the flow compiler);
    ``reset()`` is called before each execution, so the budget is per
    execution, not per run — an agent the flow legitimately revisits
    gets a fresh budget each time.

    Two rules, checked in order on every call:
    1. Budget: past ``max_calls`` calls in this execution, raise
       ``ToolCallBudgetExceededError``. Raising (not denying) ends
       the agent immediately — a denial would still cost a model turn
       per repeat, and a small model can burn the whole recursion
       limit on denials. The raise must escape the hook layer, so
       this middleware goes FIRST in the agent's middleware list
       (first = outermost): HookMiddleware converts exceptions from
       inner layers into error ToolMessages, which would swallow it.
    2. Repeat: a call identical to ANY earlier FAILED call in this
       execution (same name, same args) is denied with a message
       naming the alternative, without executing. The set (not just
       the immediate predecessor): a model that interleaves another
       call in between has not changed strategy — the booking-061
       loop evaded the predecessor-only guard by alternating
       get_booking / find_guest. A success never clears the set: a
       (name, args) that failed stays denied for the rest of the
       execution even if other calls succeed in between.
    """

    def __init__(self, max_calls: int = DEFAULT_TOOL_CALL_BUDGET):
        self.max_calls = max_calls
        self.calls = 0
        self.exceeded = False
        self._failed: dict[tuple[str, str], str] = {}

    def reset(self) -> None:
        self.calls = 0
        self.exceeded = False
        self._failed = {}

    def wrap_tool_call(self, request, handler):
        tool_call = request.tool_call or {}
        name = str(tool_call.get("name") or "")
        args = dict(tool_call.get("args") or {})
        tool_call_id = str(tool_call.get("id") or "")
        self.calls += 1
        if self.calls > self.max_calls:
            self.exceeded = True
            raise ToolCallBudgetExceededError(
                f"tool-call budget ({self.max_calls}) exceeded at '{name}'"
            )
        key = _call_key(name, args)
        if key in self._failed:
            return ToolMessage(
                content=(
                    f"You already called '{name}' with these exact arguments "
                    f"and it failed ({self._failed[key]}). Do not repeat it — "
                    f"call a different tool, use different arguments, or ask "
                    f"the guest for the missing information first."
                ),
                tool_call_id=tool_call_id,
                status="error",
            )
        result = handler(request)
        error = _error_in_result(result)
        if error is not None:
            self._failed[key] = error
        return result


def next_in_order(skill_map: SkillMap, node_id: str) -> str | None:
    ids = skill_map.node_ids
    try:
        i = ids.index(node_id)
    except ValueError:
        return None
    return ids[i + 1] if i + 1 < len(ids) else None


def tools_for_node(tools: list, node_tools: list[str]) -> list:
    """The tool objects for one map node, by name. A named tool that is
    not in the run's tool list is dropped with a warning — the node then
    runs with fewer tools, and the outcome shows what actually ran."""
    by_name = {getattr(t, "name", ""): t for t in tools}
    out = []
    for name in node_tools:
        tool = by_name.get(name)
        if tool is None:
            logger.warning("Subgraph: node tool %r is not available in this run", name)
            continue
        out.append(tool)
    return out


def last_text(messages: list) -> str:
    for message in reversed(messages or []):
        content = getattr(message, "content", "") or ""
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            joined = " ".join(t for t in texts if t)
            if joined:
                return joined
    return ""
