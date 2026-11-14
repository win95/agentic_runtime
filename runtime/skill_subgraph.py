"""Skill-map subgraphs (Phase 3 — the new architecture's core).

The proposal's reframe: most missions are workflows with agentic nodes,
not free-form agents — a *deterministic spine* (code owns routing, the
allowlist, retry and approval) with *agentic nodes* (the model decides
content inside a node). The plan gate (Phase 3a, ``runtime/plan_gate.py``)
enforces a map at the call boundary; the subgraph compiles the map into
the structure itself, so an out-of-order tool call is not denied — it is
unreachable, because the node agent only HAS its node's tools.

Compilation: one LangGraph node per map node, wired by the map's own
transition table.

- A node with tools runs a Deep Agent built with ONLY that node's tools
  (``runtime.deep_agent.build_deep_agent``), with the node's goal as its
  step instruction. A tool-less node (``route``, ``booking-dates``,
  ``confirm-booking``) runs a plain model call with no tools.
- After each node, a code router — not the model — picks the next node
  from the map: the node's declared ``NEXT: <node-id>`` line, validated
  against the allowed set (branch targets of the current node + the next
  node in order + resume nodes, below). An absent or invalid declaration
  falls back to the next node in order, never to an arbitrary jump.
  ``NEXT: END`` ends the turn only at a flow-terminal node (``terminal:
  true`` in the map — e.g. the report nodes); anywhere else it is
  treated like an invalid declaration and falls back, because the
  model judging the request "handled" mid-flow is exactly the
  premature-END failure. ``NEXT: WAIT`` ends the turn from any node:
  the node needs more information from the guest before the flow can
  continue (e.g. dates missing at booking-dates).
- ``resume_nodes`` is the multi-turn rule: from the map's start node, a
  declared jump to a resume node is allowed only when the session has
  prior turns. For hotel that is ``create-booking`` — the guest's
  confirmation turn resumes where the previous turn stopped at
  ``confirm-booking``. The map itself gains no such branch, so the gate
  (and any free-form run) keeps refusing a first-call ``create_booking``.
  The route node's instruction states the rule; the router is the
  backstop.
- A step cap (``MAX_NODE_STEPS``) ends the run with a diagnostic error
  instead of cycling forever — the map's branches allow loops by design
  (sold out -> re-pick dates -> check again).
- A per-node tool-call budget (``MAX_TOOL_CALLS_PER_NODE``, default 5)
  plus an exact-repeat guard (``_NodeToolCallCap``) end a node whose
  agent keeps calling tools: the step cap counts node executions, not
  calls inside a node, so without this a model repeating one failing
  call loops until the LangGraph recursion limit. A repeat of the
  immediately preceding failed call is denied; past the budget the
  node stops with a diagnostic error and the turn ends.

The gate and the subgraph are two enforcement mechanisms for the same
map; a run uses ONE. When the subgraph runs, the plan gate is NOT
attached (there is nothing left for it to deny), and the rendered plan
block is not appended to the prompt (the map is compiled, not prose).
``SUBGRAPH_PILOT_MISSIONS`` is the rollout scope — hotel only, for now;
``fault_analysis`` keeps its free-form agent until its data-access map
is verified against this compiler.
"""

from __future__ import annotations

import json
import logging
import re

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from runtime.deep_agent import MAX_INPUT_JSON_BYTES, build_deep_agent, outcome_from_messages
from runtime.llm_factory import build_chat_model
from runtime.security import wrap_data
from runtime.skills.map_plan import SkillMap

logger = logging.getLogger(__name__)

# Rollout scope (pilot): hotel only. The mechanism is generic; the next
# mission adopts it once its map is verified against the compiler.
SUBGRAPH_PILOT_MISSIONS = frozenset({"hotel_reception"})

# Node executions per run. The map's branches allow cycles by design
# (e.g. sold out -> pick other dates -> check again), so the cap is the
# backstop against a model that keeps looping.
MAX_NODE_STEPS = 12

# Tool calls per node execution. The step cap above counts node
# executions, not tool calls inside a node — a node agent runs until
# the model stops calling tools, so a model repeating one failing call
# (2026-11-13: get_booking with a bad id, looped until the LangGraph
# recursion limit) never reaches NEXT: and the step cap never sees it.
# This budget is the per-node backstop. 5 is generous for the reception
# map: the heaviest node (booking-lookup) legitimately needs at most
# find_guest + get_booking; every other node needs 1–2 calls.
MAX_TOOL_CALLS_PER_NODE = 5


class NodeToolCallBudgetExceededError(Exception):
    """A node agent spent its per-node tool-call budget."""


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


class _NodeToolCallCap(AgentMiddleware):
    """Per-node tool-call budget + exact-repeat guard (2026-11-13).

    One instance serves one node agent; ``reset()`` is called before
    each node execution, so the budget is per execution, not per run —
    a node the flow legitimately revisits (sold out -> re-pick dates)
    gets a fresh budget each time.

    Two rules, checked in order on every call:
    1. Budget: past ``max_calls`` calls in this execution, raise
       ``NodeToolCallBudgetExceededError``. Raising (not denying) ends
       the node immediately — a denial would still cost a model turn
       per repeat, and a small model can burn the whole recursion
       limit on denials. The raise must escape the hook layer, so
       this middleware goes FIRST in the node agent's middleware list
       (first = outermost): HookMiddleware converts exceptions from
       inner layers into error ToolMessages, which would swallow it.
    2. Exact repeat: a call identical to the immediately preceding
       FAILED call (same name, same args) is denied with a message
       naming the alternative, without executing. Only the immediate
       predecessor counts — a model that tried something else in
       between has changed strategy, and the budget still bounds it.
    """

    def __init__(self, max_calls: int = MAX_TOOL_CALLS_PER_NODE):
        self.max_calls = max_calls
        self.calls = 0
        self.exceeded = False
        self._last_failed: tuple[str, str] | None = None
        self._last_error: str = ""

    def reset(self) -> None:
        self.calls = 0
        self.exceeded = False
        self._last_failed = None
        self._last_error = ""

    def wrap_tool_call(self, request, handler):
        tool_call = request.tool_call or {}
        name = str(tool_call.get("name") or "")
        args = dict(tool_call.get("args") or {})
        tool_call_id = str(tool_call.get("id") or "")
        self.calls += 1
        if self.calls > self.max_calls:
            self.exceeded = True
            raise NodeToolCallBudgetExceededError(
                f"tool-call budget ({self.max_calls}) exceeded at '{name}'"
            )
        key = _call_key(name, args)
        if self._last_failed is not None and key == self._last_failed:
            return ToolMessage(
                content=(
                    f"You already called '{name}' with these exact arguments "
                    f"and it failed ({self._last_error}). Do not repeat it — "
                    f"call a different tool, or use different arguments, or "
                    f"end your reply with NEXT: WAIT if you need information "
                    f"from the guest first."
                ),
                tool_call_id=tool_call_id,
                status="error",
            )
        result = handler(request)
        error = _error_in_result(result)
        if error is not None:
            self._last_failed = key
            self._last_error = error
        else:
            self._last_failed = None
        return result

# The model declares its branch choice as a line in its final text:
#   NEXT: <node-id>     or     NEXT: END     or     NEXT: WAIT
# END ends the turn only at a terminal node (see _route); WAIT ends
# it from any node (the node needs more guest input to continue).
NEXT_RE = re.compile(r"^NEXT:\s*([A-Za-z0-9_-]+)\s*$", re.MULTILINE)


class SubgraphState(TypedDict, total=False):
    """Graph state. A TypedDict (not a bare dict): LangGraph only keeps
    state keys the schema declares — with ``StateGraph(dict)`` the
    passthrough keys (session_history, input_data, history_block) were
    silently dropped after the first node, so the router could never
    see the session history and the resume rule never fired."""

    messages: list
    current: str | None
    visited: list[str]
    declared_next: str | None
    steps: int
    errors: list[str]
    input_data: dict
    session_history: dict
    history_block: str


def use_skill_subgraph(mission_id: str | None, mapped_skills) -> bool:
    """True when the run should compile the skill map instead of running
    one free-form agent: a pilot mission with at least one mapped skill."""
    return mission_id in SUBGRAPH_PILOT_MISSIONS and bool(mapped_skills)


def parse_next(text: str) -> str | None:
    """The node's declared next step (a node id, or "END"), or None."""
    if not text:
        return None
    match = NEXT_RE.search(text)
    return match.group(1).strip() if match else None


def _next_in_order(skill_map: SkillMap, node_id: str) -> str | None:
    ids = skill_map.node_ids
    try:
        i = ids.index(node_id)
    except ValueError:
        return None
    return ids[i + 1] if i + 1 < len(ids) else None


def allowed_next(
    skill_map: SkillMap,
    current_id: str,
    session_turns: int = 0,
    resume_nodes: frozenset[str] = frozenset(),
) -> set[str]:
    """Node ids the router accepts from ``current_id``: the current
    node's branch targets, the next node in list order, and — only from
    the start node, only with prior session turns — the resume nodes."""
    node = skill_map.node_by_id(current_id)
    if node is None:
        return set()
    allowed = {branch.goto for branch in node.branches}
    nxt = _next_in_order(skill_map, current_id)
    if nxt is not None:
        allowed.add(nxt)
    if current_id == skill_map.start and session_turns > 0:
        allowed |= {r for r in resume_nodes if skill_map.node_by_id(r) is not None}
    return allowed


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


def _last_text(messages: list) -> str:
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


def build_skill_subgraph(
    skill_map: SkillMap,
    tools: list,
    system_prompt: str,
    model_config=None,
    middleware=None,
    backend=None,
    skills=None,
    files=None,
    callbacks=None,
    trace_metadata: dict | None = None,
    trace_tags: list[str] | None = None,
    resume_nodes: frozenset[str] = frozenset(),
    node_executor=None,
    max_steps: int = MAX_NODE_STEPS,
    max_tool_calls_per_node: int = MAX_TOOL_CALLS_PER_NODE,
):
    """Compile a SkillMap into a LangGraph whose nodes are the map's
    nodes and whose edges are the map's transition table.

    ``node_executor`` is a test seam: ``executor(node, state, messages)``
    returning the new message list, replacing the real Deep Agent /
    plain-model call. Production passes None.
    """
    agent_cache: dict[str, object] = {}
    model_cache: dict[str, object] = {}
    cap_cache: dict[str, _NodeToolCallCap] = {}

    def node_agent(node):
        if node.id not in agent_cache:
            node_tools = tools_for_node(tools, node.tools)
            tool_names = ", ".join(t for t in node.tools) or "none"
            # The cap goes FIRST (outermost): its budget raise must
            # escape HookMiddleware, which converts inner exceptions
            # into error ToolMessages — a swallowed raise would leave
            # the model looping on error messages instead of stopping.
            cap = _NodeToolCallCap(max_calls=max_tool_calls_per_node)
            cap_cache[node.id] = cap
            agent_cache[node.id] = build_deep_agent(
                node_tools,
                system_prompt=(
                    f"{system_prompt}\n\nYou are executing ONE step of a plan: "
                    f"node '{node.id}' — {node.goal}. You have ONLY these tools: "
                    f"{tool_names}. Do not call any other tool."
                ),
                skills=skills,
                model_config=model_config,
                middleware=[cap, *(middleware or [])],
                backend=backend,
            )
        return agent_cache[node.id]

    def plain_model():
        if "model" not in model_cache:
            model_cache["model"] = build_chat_model(model_config)
        return model_cache["model"]

    def node_instruction(state: dict, node, allowed: set[str]) -> str:
        input_json = json.dumps(
            state.get("input_data") or {}, ensure_ascii=False
        )[:MAX_INPUT_JSON_BYTES]
        parts = [
            f"Step '{node.id}': {node.goal}",
            f"Execution input:\n{wrap_data('execution_input', input_json)}",
        ]
        history_block = state.get("history_block") or ""
        if history_block:
            parts.append(f"Session history (prior turns):\n{history_block}")
        if node.tools:
            parts.append(
                "Call the tools for this step as needed, then answer the guest "
                "in your reply. Your tools are only this step's tools."
            )
        else:
            parts.append("Answer in text only — you have no tools in this step.")
        # The branch conditions are part of the choice: the model
        # decides with the actual `when` text, not a bare node id
        # (route once sent a quote request to booking-lookup because the
        # ids alone gave it nothing to decide on).
        hints = {b.goto: b.when for b in node.branches}
        choices = ", ".join(
            f"{nid} ({hints[nid]})" if nid in hints else nid
            for nid in sorted(allowed)
        ) or "END"
        if node.terminal:
            ending = (
                "or NEXT: END when the guest's request is fully handled in "
                "this turn, or NEXT: WAIT when you need more information "
                "from the guest before continuing."
            )
        else:
            ending = (
                "or NEXT: WAIT when you need more information from the "
                "guest before continuing. Do not end the turn with "
                "NEXT: END here — the request is not fully handled until "
                "a later step."
            )
        parts.append(
            "End your reply with exactly one line: NEXT: <node-id> — choose "
            f"from: {choices} — {ending}"
        )
        return "\n\n".join(parts)

    def make_node_fn(node):
        def _run_node(state: dict) -> dict:
            steps = int(state.get("steps") or 0) + 1
            session_turns = len(
                (state.get("session_history") or {}).get("turns") or []
            )
            allowed = allowed_next(
                skill_map, node.id,
                session_turns=session_turns, resume_nodes=resume_nodes,
            )
            messages = list(state.get("messages") or []) + [
                HumanMessage(content=node_instruction(state, node, allowed))
            ]
            config: dict = {}
            if callbacks:
                config["callbacks"] = callbacks
            if trace_metadata:
                config["metadata"] = trace_metadata
            if trace_tags:
                config["tags"] = trace_tags
            errors = list(state.get("errors") or [])
            if node_executor is not None:
                new_messages = node_executor(node, state, messages)
            elif node.tools:
                agent = node_agent(node)
                cap_cache[node.id].reset()
                payload: dict = {"messages": messages}
                if files:
                    payload["files"] = files
                try:
                    if config:
                        result = agent.invoke(payload, config=config)
                    else:
                        result = agent.invoke(payload)
                    new_messages = result.get("messages") or []
                except NodeToolCallBudgetExceededError as exc:
                    # The node agent spent its tool-call budget: stop
                    # the node here instead of letting the model burn
                    # the recursion limit on further calls. NEXT: WAIT
                    # ends the turn from any node (see _route).
                    errors.append(
                        f"Tool-call budget ({max_tool_calls_per_node} calls) "
                        f"exceeded at node '{node.id}': {exc}"
                    )
                    new_messages = messages + [
                        AIMessage(content=(
                            f"Step '{node.id}' stopped: the tool-call budget "
                            f"for this step is exhausted.\nNEXT: WAIT"
                        ))
                    ]
            else:
                model = plain_model()
                reply = (
                    model.invoke(messages, config=config)
                    if config
                    else model.invoke(messages)
                )
                new_messages = messages + [reply]
            if steps >= max_steps:
                errors.append(
                    f"Plan step cap ({max_steps} nodes) reached at '{node.id}'"
                )
            return {
                "messages": new_messages,
                "current": node.id,
                "visited": list(state.get("visited") or []) + [node.id],
                "declared_next": parse_next(_last_text(new_messages)),
                "steps": steps,
                "errors": errors,
            }

        return _run_node

    def _route(state: dict) -> str:
        if int(state.get("steps") or 0) >= max_steps:
            return "END"
        declared = state.get("declared_next")
        current = state.get("current") or skill_map.start
        if declared == "WAIT":
            # The node needs more information from the guest before the
            # flow can continue: end the turn, never force it forward.
            return "END"
        if declared == "END":
            # END is only honored at a flow-terminal node. Anywhere else
            # it means the model judged the request handled mid-flow
            # (task-details ending the run after merely restating the
            # issue, with no task created) — treat it like an invalid
            # declaration and fall through to the next node in order.
            node = skill_map.node_by_id(current)
            if node is not None and node.terminal:
                return "END"
            declared = None
        session_turns = len(
            (state.get("session_history") or {}).get("turns") or []
        )
        allowed = allowed_next(
            skill_map, current,
            session_turns=session_turns, resume_nodes=resume_nodes,
        )
        if declared is not None and declared in allowed:
            # A resume node (e.g. create-booking) is enterable only with
            # prior turns — never on a fresh turn, even when declared:
            # the confirmation turn resumes a prior turn's flow, a fresh
            # turn must stop at the confirmation step instead.
            if declared in resume_nodes and session_turns == 0:
                return "END"
            return declared
        nxt = _next_in_order(skill_map, current)
        # Same guard for the fallback: never fall through into a resume
        # node on a fresh turn (confirm-booking -> create-booking would
        # book before the guest confirms). End the turn instead.
        if nxt is not None and nxt in resume_nodes and session_turns == 0:
            return "END"
        return nxt if nxt is not None else "END"

    graph = StateGraph(SubgraphState)
    for node in skill_map.nodes:
        graph.add_node(node.id, make_node_fn(node))
    graph.add_edge(START, skill_map.start)
    path_map = {node.id: node.id for node in skill_map.nodes}
    path_map["END"] = END
    for node in skill_map.nodes:
        graph.add_conditional_edges(node.id, _route, path_map)
    return graph.compile()


def run_skill_subgraph(
    compiled,
    mission: dict,
    input_data: dict,
    session_history: dict | None = None,
    history_block: str = "",
    callbacks=None,
    skill_maps: list[SkillMap] | None = None,
    trace_metadata: dict | None = None,
    trace_tags: list[str] | None = None,
) -> dict:
    """Invoke a compiled skill subgraph for one mission input.

    Returns the same outcome shape as ``run_deep_agent`` (see
    ``outcome_from_messages``), with graph-level diagnostics (e.g. the
    step cap) merged into ``errors`` ahead of the tool errors.
    """
    objective = mission.get("objective", "")
    input_json = json.dumps(input_data, ensure_ascii=False)[:MAX_INPUT_JSON_BYTES]
    prompt = (
        f"Mission: {mission.get('mission_id')}\n"
        f"Objective: {objective}\n"
        f"Input:\n{wrap_data('execution_input', input_json)}\n\n"
        "Follow the plan step by step and return a concise summary of "
        "what was done, citing the evidence gathered."
    )
    initial: dict = {
        "messages": [HumanMessage(content=prompt)],
        "current": None,
        "visited": [],
        "declared_next": None,
        "steps": 0,
        "errors": [],
        "input_data": input_data,
        "session_history": session_history or {},
        "history_block": history_block,
    }
    config: dict = {}
    if callbacks:
        config["callbacks"] = callbacks
    if trace_metadata:
        config["metadata"] = trace_metadata
    if trace_tags:
        config["tags"] = trace_tags
    result = compiled.invoke(initial, config=config) if config else compiled.invoke(initial)
    messages = result.get("messages") or []
    outcome = outcome_from_messages(messages, skill_maps=skill_maps)
    # The graph's own visited path (every node, tool-less included) —
    # plan_trace only sees nodes with tool calls, so the path is
    # reported separately for the execution notes and tests.
    outcome["visited_nodes"] = list(result.get("visited") or [])
    graph_errors = list(result.get("errors") or [])
    if graph_errors:
        outcome["errors"] = graph_errors + outcome["errors"]
    return outcome
