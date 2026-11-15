"""Flow-agent subgraphs (Direction C, 2026-11-15).

The v4b live run (109 pass / 32 fail) showed the node compiler's
failure mode is structural, not a parse bug: one LangGraph node per
map node, routed by a ``NEXT:`` text line the model writes inside the
same message that answers the guest, means every step is a routing
decision, the control line leaks into the answer, and one bad parse
walks the map in list order across flow boundaries (a failed
``report-booking`` created a task the guest never asked for). The
fix-proposal set patched each symptom separately (tolerant regex,
stop-on-invalid, terminal guard, answer stripping); this compiler
replaces the contract instead.

Direction C: one agent per FLOW, not per node.

- Flows are derived from the map, not declared: each branch target of
  the start (``route``) node starts a flow; the flow walks the map in
  list order until a terminal node (inclusive) or a resume node
  (exclusive — a resume node starts its own flow, see below). Nodes
  the walk never reaches become single-node flows, so every map node
  belongs to exactly one flow. For hotel_reception that gives 8 flows:
  booking-quote (booking-dates, check-availability, quote-price,
  confirm-booking), booking-create (create-booking, report-booking),
  booking-payment (booking-payment, collect-payment, confirm-payment,
  report-payment), task (task-details, find-staff, create-task,
  assign-task, report-task), and the four single-node lookups
  (policy-lookup, booking-lookup, booking-change, task-status).
- The router runs once per flow boundary, not once per node: a plain
  model call (the map's start node) declares ``FLOW: <flow-id>`` — or
  ``FLOW: END`` / ``FLOW: WAIT``. The choice set is 7-8 flow ids with
  their branch ``when`` conditions, not 20 nodes. After a flow ends,
  the router runs again so a multi-request turn can enter a second
  flow; a flow already visited this turn is never re-entered.
- There is no ``NEXT:`` inside a flow. The flow agent is a Deep Agent
  with the UNION of its flow's tools and its steps as a checklist in
  the system prompt; it works through them in order and stops when its
  reply is ready. The answer channel is separate from the control
  channel: each flow writes its reply text to a dedicated ``answer``
  state key, and the outcome summary comes from that key — never from
  the message list, so no control line can leak into the guest's
  answer. (The route node's replies are not in ``flow_messages`` at
  all, so ``outcome_from_messages`` never sees them; and when no flow
  produced an answer the summary stays empty instead of falling back
  to the raw message list.) Each flow agent's input is isolated: it
  sees only the mission prompt, earlier flows' answer texts, and its
  own flow instruction — never the router's instruction or its
  ``FLOW:`` declarations, which the model would otherwise copy into
  its reply.
- An invalid or missing flow declaration ENDS the turn with an error
  recorded — there is no "next in order" fallback, because in a flat
  list of independent flows there is no meaningful next. This is the
  deliberate contract change: a routing failure is a stopped turn,
  not a walk into another flow. "Invalid" includes ambiguous: a reply
  with two ``FLOW:`` declarations, or an "or ``FLOW:``" alternative,
  is no choice at all — the parser does not silently take the first
  token.
- The write gate is structural, not prompt-only. ``resume_nodes``
  (``create-booking``) are flow boundaries: the booking-create flow is
  enterable only when the session has prior turns, and its tools are
  only ``create_booking`` + the report step — the booking-quote flow
  does not HAVE ``create_booking`` at all, so the agent cannot call
  it before the guest confirms. The router enforces the prior-turns
  condition; the tool allowlist enforces the rest.
- The code-logic fixes from the v4b root-cause synthesis live here
  too, shared with the legacy compiler: the per-flow tool-call budget
  (``MAX_TOOL_CALLS_PER_FLOW``, default 10 — a 6-node flow legitimately
  needs more than the old flat 5) with the set-based repeat guard
  (``ToolCallCap``), and ``stop_completion`` is scoped OUT of flow
  agents (the mission graph builds a flow middleware whose registry
  excludes it — a flow agent that stops without tools is usually
  asking the guest a question, not failing to gather evidence).
- Checklist verification is an audit, not a second router: after each
  flow, ``flow_trace`` records which of the flow's steps ran (a step
  counts as done when its tools were called; a tool-less step counts
  when the flow produced answer text) and which are missing. The run
  still completes — the trace is for the v5 acceptance check, not for
  steering the model mid-turn.

``FLOW_PILOT_MISSIONS`` is the rollout scope — hotel only, for now.
The node compiler (``runtime/subgraph/node.py``) stays as the
fallback path; a run uses ONE.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from runtime.deep_agent import MAX_INPUT_JSON_BYTES, build_deep_agent, outcome_from_messages
from runtime.llm_factory import build_chat_model
from runtime.security import wrap_data
from runtime.skills.map_plan import SkillMap
from runtime.subgraph.common import (
    ToolCallBudgetExceededError,
    ToolCallCap,
    last_text,
    next_in_order,
    tools_for_node,
)

logger = logging.getLogger(__name__)

# Rollout scope (pilot): hotel only. The mechanism is generic; the next
# mission adopts it once its map is verified against the compiler.
FLOW_PILOT_MISSIONS = frozenset({"hotel_reception"})

# Node executions per run (route calls + flow executions). A
# multi-request turn is route -> flow -> route -> flow, so 12 still
# leaves headroom over the old node compiler's cap.
MAX_FLOW_STEPS = 12

# Tool calls per flow execution. A flow covers several map nodes, so
# the flat 5 of the node compiler is too small: the v4b run showed
# policy-lookup alone spending 5 calls on search/read retries, and a
# booking flow legitimately needs check + quote + (on the confirm
# turn) create. 10 covers the observed paths with headroom; the
# budget still bounds a looping agent.
MAX_TOOL_CALLS_PER_FLOW = 10

# The route node's declaration. Tolerant by design (the v4b root
# cause): the first token after FLOW: is the choice — bold markers
# (the model writes **FLOW:**, colon inside the bold), indentation
# and trailing explanation after the token are all accepted, because
# the model writes prose around its declaration. Tolerant is NOT
# ambiguous: a reply that declares two flows ("FLOW: a or FLOW: b")
# or hedges with an alternative ("... or FLOW: b") is not a vote for
# the first token — it is no choice at all, and the router ends the
# turn instead of misrouting on it (v5 review, point 1).
FLOW_RE = re.compile(r"(?m)^[ \t]*(?:\*\*)?FLOW\s*:\s*(?:\*\*)?\s*([A-Za-z0-9_-]+)")
_FLOW_COUNT_RE = re.compile(r"\bFLOW\s*:", re.IGNORECASE)
_OR_FLOW_RE = re.compile(r"\bor\s+FLOW\s*:", re.IGNORECASE)


def _is_ambiguous_flow_declaration(text: str | None) -> bool:
    """True when the reply declares more than one flow, or offers a
    FLOW: as an alternative ("... or FLOW: b") — either way the model
    did not make one choice."""
    if not text:
        return False
    if _OR_FLOW_RE.search(text):
        return True
    return len(_FLOW_COUNT_RE.findall(text)) > 1


@dataclass
class Flow:
    """One flow: a chain of map nodes run by a single agent.

    ``id`` is the entry node id — it is also the token the router
    declares (``FLOW: booking-dates``). ``resume_gated`` marks flows
    whose entry is a resume node: enterable only with prior turns.
    """

    id: str
    nodes: list = field(default_factory=list)
    resume_gated: bool = False

    @property
    def tools(self) -> list[str]:
        """Union of the steps' tools, order-preserving, deduplicated."""
        seen: list[str] = []
        for node in self.nodes:
            for name in node.tools or []:
                if name not in seen:
                    seen.append(name)
        return seen


def derive_flows(
    skill_map: SkillMap,
    resume_nodes: frozenset[str] = frozenset(),
) -> list[Flow]:
    """Group a map's nodes into flows.

    Entries are the start node's branch targets (in branch order),
    then any resume node not already an entry — a resume node starts
    its own flow so the write gate is a flow boundary, not a prompt
    rule. Each flow walks the map in list order until a terminal node
    (inclusive) or a resume node (exclusive). Unreachable nodes become
    single-node flows so every node belongs to exactly one flow.
    """
    start = skill_map.node_by_id(skill_map.start)
    entries: list[str] = []
    if start is not None:
        entries = [b.goto for b in start.branches if b.goto != skill_map.start]
    for node_id in resume_nodes:
        if skill_map.node_by_id(node_id) is not None and node_id not in entries:
            entries.append(node_id)
    covered: set[str] = set()
    flows: list[Flow] = []
    for entry in entries:
        if entry in covered:
            continue
        nodes = []
        cur: str | None = entry
        while cur is not None:
            if cur in covered:
                logger.warning(
                    "Flow: node %r already in another flow; stopping flow %r",
                    cur,
                    entry,
                )
                break
            node = skill_map.node_by_id(cur)
            if node is None:
                logger.warning("Flow: node %r not in map; stopping flow %r", cur, entry)
                break
            if node.id in resume_nodes and node.id != entry:
                # A resume node starts its own flow (its own entry,
                # above) — never walk into it from another flow.
                break
            nodes.append(node)
            covered.add(node.id)
            if node.terminal:
                break
            cur = next_in_order(skill_map, node.id)
        if nodes:
            flows.append(Flow(id=entry, nodes=nodes, resume_gated=entry in resume_nodes))
    for node in skill_map.nodes:
        # The start node is the router, never a flow member.
        if node.id == skill_map.start:
            continue
        if node.id not in covered:
            logger.warning(
                "Flow: node %r not reachable from any route branch; running it as its own flow",
                node.id,
            )
            flows.append(
                Flow(
                    id=node.id,
                    nodes=[node],
                    resume_gated=node.id in resume_nodes,
                )
            )
            covered.add(node.id)
    # A branch target that is neither inside its flow nor another
    # flow's entry is a jump this compiler cannot express — warn, do
    # not silently drop it (for hotel every branch target is in-flow
    # or an entry).
    entry_ids = {f.id for f in flows}
    for flow in flows:
        in_flow = {n.id for n in flow.nodes}
        for node in flow.nodes:
            for branch in node.branches:
                if branch.goto not in in_flow and branch.goto not in entry_ids:
                    logger.warning(
                        "Flow: branch %r -> %r in flow %r points outside "
                        "any flow; the flow agent cannot take it",
                        node.id,
                        branch.goto,
                        flow.id,
                    )
    return flows


def parse_flow(text: str | None) -> str | None:
    """The flow id the route node declared, or None — no FLOW: line,
    or an ambiguous one (two declarations, or an "or FLOW:"
    alternative: the model did not make one choice)."""
    if not text:
        return None
    if _is_ambiguous_flow_declaration(text):
        return None
    match = FLOW_RE.search(text)
    return match.group(1) if match else None


def use_flow_subgraph(mission_id: str | None, mapped_skills) -> bool:
    """True when the mission is a flow pilot and has a mapped skill."""
    return mission_id in FLOW_PILOT_MISSIONS and bool(mapped_skills)


def _tool_names_called(messages: list) -> set[str]:
    names: set[str] = set()
    for message in messages or []:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict) and call.get("name"):
                names.add(str(call["name"]))
        if isinstance(message, ToolMessage) and getattr(message, "name", ""):
            names.add(str(message.name))
    return names


def _flow_answer(added: list) -> str:
    """The flow agent's reply text — the guest's answer.

    The last AI message with genuine text, with any FLOW: line
    stripped (a flow agent is never told to write one, so this is
    defense only). A budget diagnostic is not an answer.
    """
    for message in reversed(added or []):
        if not isinstance(message, AIMessage):
            continue
        content = getattr(message, "content", "") or ""
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            content = " ".join(t for t in texts if t)
        if not isinstance(content, str) or not content.strip():
            continue
        if content.strip().startswith("Flow '") and "stopped:" in content:
            continue
        stripped = "\n".join(
            line for line in content.splitlines() if not FLOW_RE.match(line)
        ).strip()
        if stripped:
            return stripped
    return ""


def _trace_flow(flow: Flow, messages: list, answer: str) -> dict:
    """Checklist audit for one flow execution.

    A step counts as done when its tools were called; a tool-less
    step (asking the guest a question) counts when the flow produced
    answer text. Missing steps are reported, not enforced — the trace
    is for the v5 acceptance check.
    """
    called = _tool_names_called(messages)
    done: list[str] = []
    missing: list[str] = []
    for node in flow.nodes:
        node_tools = node.tools or []
        if not node_tools:
            (done if answer else missing).append(node.id)
        elif any(t in called for t in node_tools):
            done.append(node.id)
        else:
            missing.append(node.id)
    return {"flow": flow.id, "steps_completed": done, "steps_missing": missing}


class FlowState(TypedDict, total=False):
    """Graph state. ``answer`` is the guest-facing reply channel —
    separate from ``messages``, so no control line can leak into it.
    ``flow_messages`` holds only flow-agent output (never the route
    node's replies), so the outcome never derives a summary from a
    routing message. ``flow_answers`` holds each flow's guest-facing
    reply text in order — the only thing a later flow in the same turn
    sees of an earlier flow (see the isolation in ``_run_flow``)."""

    messages: list
    current: str | None
    visited: list[str]
    declared_flow: str | None
    route_next: str | None
    steps: int
    errors: list[str]
    input_data: dict
    session_history: dict
    history_block: str
    answer: str
    flow_messages: list
    flow_trace: list[dict]
    flow_answers: list[str]


def build_flow_subgraph(
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
    flow_executor=None,
    max_steps: int = MAX_FLOW_STEPS,
    max_tool_calls_per_flow: int = MAX_TOOL_CALLS_PER_FLOW,
):
    """Compile a SkillMap into a LangGraph of flow agents + one router.

    Graph shape: START -> route -> (flow | END); each flow -> route
    (re-route for a second request) or END at the step cap. The
    router picks a flow once per flow boundary; a flow already visited
    this turn is never re-entered.

    ``flow_executor`` is a test seam: ``executor(flow, state,
    messages)`` returning the new message list, replacing the real
    Deep Agent call. Production passes None.
    """
    flows = derive_flows(skill_map, resume_nodes=resume_nodes)
    by_id = {flow.id: flow for flow in flows}
    start = skill_map.node_by_id(skill_map.start)
    when_by_entry = {b.goto: b.when for b in (start.branches if start else [])}
    agent_cache: dict[str, object] = {}
    model_cache: dict[str, object] = {}
    cap_cache: dict[str, ToolCallCap] = {}

    def flow_agent(flow: Flow):
        if flow.id not in agent_cache:
            flow_tools = tools_for_node(tools, flow.tools)
            tool_names = ", ".join(flow.tools) or "none"
            steps = "\n".join(
                f"{i + 1}. {node.id}: {node.goal}"
                + (
                    f" (tools: {', '.join(node.tools)})"
                    if node.tools
                    else " (text only — no tools)"
                )
                for i, node in enumerate(flow.nodes)
            )
            extra = ""
            if flow.resume_gated:
                extra = (
                    "\n\nThis flow runs only because the guest already "
                    "confirmed in a previous turn — create the booking now "
                    "and report it."
                )
            # The cap goes FIRST (outermost): its budget raise must
            # escape HookMiddleware, which converts inner exceptions
            # into error ToolMessages — a swallowed raise would leave
            # the model looping on error messages instead of stopping.
            cap = ToolCallCap(max_calls=max_tool_calls_per_flow)
            cap_cache[flow.id] = cap
            agent_cache[flow.id] = build_deep_agent(
                flow_tools,
                system_prompt=(
                    f"{system_prompt}\n\nYou are executing ONE flow of a "
                    f"plan: flow '{flow.id}'. Work through these steps in "
                    f"order, skipping any that do not apply to this "
                    f"request:\n{steps}{extra}\n\nYou have ONLY these "
                    f"tools: {tool_names}. Do not call any other tool. "
                    f"Call the tools for a step as needed, then answer "
                    f"the guest in your reply. If you need information "
                    f"from the guest, ask in your reply and stop — your "
                    f"reply ends this flow's turn."
                ),
                skills=skills,
                model_config=model_config,
                middleware=[cap, *(middleware or [])],
                backend=backend,
            )
        return agent_cache[flow.id]

    def route_model():
        if "model" not in model_cache:
            model_cache["model"] = build_chat_model(model_config)
        return model_cache["model"]

    def route_instruction(state: dict) -> str:
        input_json = json.dumps(state.get("input_data") or {}, ensure_ascii=False)[
            :MAX_INPUT_JSON_BYTES
        ]
        parts = [
            "Route the guest's request to ONE flow. Read the "
            "conversation: what the flows already did this turn is in "
            "their replies — route only what is still unhandled.",
            f"Execution input:\n{wrap_data('execution_input', input_json)}",
        ]
        history_block = state.get("history_block") or ""
        if history_block:
            parts.append(f"Session history (prior turns):\n{history_block}")
        choices = []
        for flow in flows:
            when = when_by_entry.get(flow.id)
            if when is None and flow.resume_gated:
                when = (
                    "the guest confirmed a booking in a previous turn "
                    "(only when the session has prior turns)"
                )
            choices.append(f"- {flow.id} ({when})" if when else f"- {flow.id}")
        parts.append("Flows (choose one):\n" + "\n".join(choices))
        visited = state.get("visited") or []
        parts.append(
            "Already visited this turn: "
            + (", ".join(visited) if visited else "none")
            + " — do not pick a visited flow again."
        )
        parts.append(
            "End your reply with exactly one line: FLOW: <flow-id> — or "
            "FLOW: END when nothing remains to do, or FLOW: WAIT when "
            "you need more information from the guest before routing."
        )
        return "\n\n".join(parts)

    def flow_instruction(state: dict, flow: Flow) -> str:
        input_json = json.dumps(state.get("input_data") or {}, ensure_ascii=False)[
            :MAX_INPUT_JSON_BYTES
        ]
        parts = [
            f"Flow '{flow.id}': work through its steps in order.",
            f"Execution input:\n{wrap_data('execution_input', input_json)}",
        ]
        history_block = state.get("history_block") or ""
        if history_block:
            parts.append(f"Session history (prior turns):\n{history_block}")
        return "\n\n".join(parts)

    def _config() -> dict:
        config: dict = {}
        if callbacks:
            config["callbacks"] = callbacks
        if trace_metadata:
            config["metadata"] = trace_metadata
        if trace_tags:
            config["tags"] = trace_tags
        return config

    def make_route_fn():
        def _run_route(state: dict) -> dict:
            steps = int(state.get("steps") or 0) + 1
            messages = list(state.get("messages") or []) + [
                HumanMessage(content=route_instruction(state))
            ]
            config = _config()
            model = route_model()
            reply = model.invoke(messages, config=config) if config else model.invoke(messages)
            new_messages = messages + [reply]
            reply_text = last_text(new_messages)
            declared = parse_flow(reply_text)
            errors = list(state.get("errors") or [])
            # The router decides here; the edge function only reads
            # route_next. Every invalid case ends the turn with an
            # error recorded — there is no "next in order" fallback:
            # in a flat list of independent flows there is no
            # meaningful next, and a silent walk is how a failed
            # report-booking created a task the guest never asked for.
            route_next: str | None = None
            if declared is None:
                if _is_ambiguous_flow_declaration(reply_text):
                    errors.append(
                        "Router declared an ambiguous flow (more than "
                        "one FLOW: line, or 'or FLOW:'); ending turn"
                    )
                else:
                    errors.append("Router declared no flow (no FLOW: line); ending turn")
            elif declared in ("END", "WAIT"):
                route_next = "END"
            elif declared not in by_id:
                errors.append(f"Router declared unknown flow '{declared}'; ending turn")
            elif declared in (state.get("visited") or []):
                errors.append(
                    f"Router declared flow '{declared}' already visited this turn; ending turn"
                )
            elif by_id[declared].resume_gated:
                session_turns = len((state.get("session_history") or {}).get("turns") or [])
                if session_turns == 0:
                    errors.append(
                        f"Router declared resume-gated flow '{declared}' "
                        "on a fresh turn; ending turn"
                    )
                else:
                    route_next = declared
            else:
                route_next = declared
            if steps >= max_steps:
                errors.append(f"Plan step cap ({max_steps} nodes) reached at 'route'")
                route_next = "END"
            return {
                "messages": new_messages,
                "current": "route",
                "visited": list(state.get("visited") or []),
                "declared_flow": declared,
                "route_next": route_next,
                "steps": steps,
                "errors": errors,
            }

        return _run_route

    def make_flow_fn(flow: Flow):
        def _run_flow(state: dict) -> dict:
            steps = int(state.get("steps") or 0) + 1
            # Context isolation (v5 review, point 3): the flow agent
            # must NOT see the router's instruction or its FLOW:
            # declarations — the model copies them into its own reply
            # ("FLOW: check" as the whole answer). It gets only the
            # mission prompt, what earlier flows already told the
            # guest this turn (their answer texts, never their raw
            # messages), and its own flow instruction. The full
            # history stays in state["messages"] for the router and
            # the trace — only the agent's INPUT is isolated.
            history = list(state.get("messages") or [])
            instruction = HumanMessage(content=flow_instruction(state, flow))
            prior_answers = list(state.get("flow_answers") or [])
            isolated: list = history[:1]
            if prior_answers:
                isolated.append(
                    HumanMessage(
                        content=(
                            "Earlier flows this turn already told the "
                            "guest:\n" + "\n---\n".join(prior_answers)
                        )
                    )
                )
            isolated.append(instruction)
            config = _config()
            errors = list(state.get("errors") or [])
            if flow_executor is not None:
                new_messages = flow_executor(flow, state, isolated)
            else:
                agent = flow_agent(flow)
                cap_cache[flow.id].reset()
                payload: dict = {"messages": isolated}
                if files:
                    payload["files"] = files
                try:
                    if config:
                        result = agent.invoke(payload, config=config)
                    else:
                        result = agent.invoke(payload)
                    new_messages = result.get("messages") or []
                except ToolCallBudgetExceededError as exc:
                    # The flow agent spent its tool-call budget: stop
                    # the flow here instead of letting the model burn
                    # the recursion limit on further calls. The
                    # diagnostic is an error, not the guest's answer.
                    errors.append(
                        f"Tool-call budget ({max_tool_calls_per_flow} calls) "
                        f"exceeded at flow '{flow.id}': {exc}"
                    )
                    new_messages = isolated + [
                        AIMessage(
                            content=(
                                f"Flow '{flow.id}' stopped: the tool-call "
                                f"budget for this flow is exhausted."
                            )
                        )
                    ]
            added = new_messages[len(isolated) :] if len(new_messages) >= len(isolated) else []
            answer = _flow_answer(added)
            if steps >= max_steps:
                errors.append(f"Plan step cap ({max_steps} nodes) reached at flow '{flow.id}'")
            return {
                "messages": history + [instruction] + added,
                "current": flow.id,
                "visited": list(state.get("visited") or []) + [flow.id],
                "declared_flow": None,
                "route_next": None,
                "steps": steps,
                "errors": errors,
                "answer": answer or state.get("answer") or "",
                "flow_messages": list(state.get("flow_messages") or []) + added,
                "flow_trace": list(state.get("flow_trace") or [])
                + [_trace_flow(flow, new_messages, answer)],
                "flow_answers": prior_answers + ([answer] if answer else []),
            }

        return _run_flow

    def _route_flow(state: dict) -> str:
        if int(state.get("steps") or 0) >= max_steps:
            return "END"
        nxt = state.get("route_next")
        return nxt if nxt in by_id else "END"

    def _after_flow(state: dict) -> str:
        # At the step cap there is no point running the router again
        # for one more model call — end the turn.
        return "END" if int(state.get("steps") or 0) >= max_steps else "route"

    graph = StateGraph(FlowState)
    graph.add_node("route", make_route_fn())
    for flow in flows:
        graph.add_node(flow.id, make_flow_fn(flow))
    graph.add_edge(START, "route")
    path_map = {flow.id: flow.id for flow in flows}
    path_map["END"] = END
    graph.add_conditional_edges("route", _route_flow, path_map)
    for flow in flows:
        graph.add_conditional_edges(flow.id, _after_flow, {"route": "route", "END": END})
    return graph.compile()


def run_flow_subgraph(
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
    """Invoke a compiled flow subgraph for one mission input.

    Returns the same outcome shape as ``run_deep_agent`` (see
    ``outcome_from_messages``), with the flow answer as the summary
    and graph-level diagnostics merged into ``errors`` ahead of the
    tool errors. ``visited_nodes`` lists flow ids (not map nodes);
    ``flow_trace`` is the per-flow checklist audit.
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
        "declared_flow": None,
        "route_next": None,
        "steps": 0,
        "errors": [],
        "input_data": input_data,
        "session_history": session_history or {},
        "history_block": history_block,
        "answer": "",
        "flow_messages": [],
        "flow_trace": [],
        "flow_answers": [],
    }
    config: dict = {}
    if callbacks:
        config["callbacks"] = callbacks
    if trace_metadata:
        config["metadata"] = trace_metadata
    if trace_tags:
        config["tags"] = trace_tags
    result = compiled.invoke(initial, config=config) if config else compiled.invoke(initial)
    flow_messages = result.get("flow_messages") or []
    outcome = outcome_from_messages(flow_messages, skill_maps=skill_maps)
    # The answer channel, not the message list: the summary is the
    # last flow agent's reply text. The route node's "FLOW: END" line
    # is never in flow_messages, so it cannot leak into the summary.
    answer = (result.get("answer") or "").strip()
    if answer:
        outcome["summary"] = answer
    else:
        # No flow produced a guest-facing reply: never fall back to
        # the raw message list — a flow agent's reply can be a bare
        # control line ("FLOW: check"), and that line must not become
        # the guest's answer (v5 review, point 2). An empty summary
        # plus the recorded errors says honestly that nothing was
        # answered.
        outcome["summary"] = ""
    # The graph's own visited path (flow ids) — plan_trace only sees
    # flows with tool calls, so the path is reported separately.
    outcome["visited_nodes"] = list(result.get("visited") or [])
    outcome["flow_trace"] = list(result.get("flow_trace") or [])
    graph_errors = list(result.get("errors") or [])
    if graph_errors:
        outcome["errors"] = graph_errors + outcome["errors"]
    return outcome
