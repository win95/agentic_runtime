"""Structured flow router — selection, budget, reprompt, error surfacing.

No live LLM: exercises ``runtime/subgraph/flow.py`` helpers and the
compiled graph through the ``route_invoker`` seam.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from runtime.context_budget import ROUTE_INSTRUCTION_TOKENS
from runtime.mission_graph import _errors_explain_empty_evidence, validate_results
from runtime.skills.map_plan import MapBranch, MapNode, SkillMap
from runtime.subgraph.flow import (
    ROUTE_TOOL_NAME,
    ROUTER_CLARIFICATION_REPLY,
    append_unique_errors,
    build_flow_subgraph,
    build_route_instruction,
    derive_flows,
    eligible_flow_ids,
    parse_route_declaration,
    route_instruction_token_count,
    route_tool_spec,
    run_flow_subgraph,
)


def _map(nodes, start="route"):
    return SkillMap(
        skill_name="test-flow",
        start=start,
        nodes=[
            MapNode(
                id=n["id"],
                goal=n.get("goal", n["id"]),
                tools=n.get("tools", []),
                branches=[
                    MapBranch(when=b["when"], goto=b["goto"]) for b in n.get("branches", [])
                ],
                terminal=bool(n.get("terminal", False)),
                resume=bool(n.get("resume", False)),
            )
            for n in nodes
        ],
    )


FLOW_MAP = _map(
    [
        {
            "id": "route",
            "branches": [
                {"when": "guest wants to book", "goto": "check"},
                {"when": "guest wants a task", "goto": "find"},
            ],
        },
        {"id": "check", "goal": "Collect stay dates", "tools": ["t_check"], "terminal": True},
        {"id": "find", "goal": "Collect task details", "tools": ["t_find"], "terminal": True},
        {
            "id": "create",
            "goal": "Create after confirm",
            "tools": ["t_create"],
            "terminal": True,
            "resume": True,
        },
    ]
)


def _tool_call(flow_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "name": ROUTE_TOOL_NAME,
                "args": {"flow_id": flow_id},
                "type": "tool_call",
            }
        ],
    )


def test_eligible_flow_ids_excludes_visited():
    flows = derive_flows(FLOW_MAP)
    state = {"visited": ["check"]}
    assert eligible_flow_ids(flows, state) == ["find", "create"]


def test_append_unique_errors_dedupes():
    errors = ["Router declared flow 'x' already visited this turn; ending turn"]
    append_unique_errors(errors, [errors[0]])
    assert len(errors) == 1


def test_route_tool_spec_enum_from_map():
    spec = route_tool_spec(["check", "find"])
    enum = spec["function"]["parameters"]["properties"]["flow_id"]["enum"]
    assert set(enum) == {"check", "find", "END", "WAIT"}


def test_parse_route_declaration_prefers_tool_over_text():
    msg = _tool_call("check")
    assert parse_route_declaration(msg, frozenset({"check", "END", "WAIT"})) == "check"
    text_msg = AIMessage(content="prose\nFLOW: find")
    assert parse_route_declaration(
        text_msg, frozenset({"check", "find", "END", "WAIT"})
    ) == "find"


def test_parse_route_declaration_rejects_unknown_tool_enum():
    msg = _tool_call("not-a-flow")
    assert parse_route_declaration(msg, frozenset({"check", "END", "WAIT"})) is None


def test_route_instruction_respects_token_budget():
    long_goal = "x" * 500
    big_map = _map(
        [
            {"id": "route", "branches": [{"when": "a", "goto": f"f{i}"} for i in range(30)]},
            *[
                {"id": f"f{i}", "goal": long_goal, "terminal": True}
                for i in range(30)
            ],
        ]
    )
    from runtime.subgraph.flow import derive_flows

    flows = derive_flows(big_map)
    when = {b.goto: b.when for b in big_map.node_by_id("route").branches}
    state = {"input_data": {"message": "hi"}, "visited": [], "history_block": ""}
    assert (
        route_instruction_token_count(state, flows, when)
        <= ROUTE_INSTRUCTION_TOKENS
    )


def test_flow_subgraph_structured_route_success():
    calls = iter([_tool_call("check"), _tool_call("END")])

    def route_invoker(messages, tool_spec, config):  # noqa: ARG001
        return next(calls)

    def flow_executor(flow, state, messages):  # noqa: ARG001
        return messages + [AIMessage(content=f"ran {flow.id}")]

    compiled = build_flow_subgraph(
        FLOW_MAP,
        tools=[],
        system_prompt="base",
        flow_executor=flow_executor,
        route_invoker=route_invoker,
    )
    out = run_flow_subgraph(
        compiled,
        {"mission_id": "test", "objective": "test"},
        {"message": "book"},
        skill_maps=[FLOW_MAP],
    )
    assert "check" in (out.get("visited_nodes") or [])
    assert not any("Router declared no flow" in e for e in out.get("errors") or [])


def test_analytis_router_runs_research_then_report():
    """The real analytis map: router picks research, then report, then END.

    A visited flow is never re-entered, so after research the only
    eligible flow is report — the router cannot loop back, and the
    report flow (no tavily tools) cannot start new research.
    """
    from runtime.skills import load_skills

    skill_map = {s.name: s for s in load_skills()}["agent-analytis"].map
    assert skill_map is not None
    calls = iter(
        [_tool_call("clarify-scope"), _tool_call("report-generate"), _tool_call("END")]
    )

    def route_invoker(messages, tool_spec, config):  # noqa: ARG001
        return next(calls)

    def flow_executor(flow, state, messages):  # noqa: ARG001
        return messages + [AIMessage(content=f"ran {flow.id}")]

    compiled = build_flow_subgraph(
        skill_map,
        tools=[],
        system_prompt="base",
        flow_executor=flow_executor,
        route_invoker=route_invoker,
    )
    out = run_flow_subgraph(
        compiled,
        {"mission_id": "agent_analytis_mission", "objective": "research"},
        {"message": "nghiên cứu giá cà phê robusta"},
        skill_maps=[skill_map],
    )
    assert out.get("visited_nodes") == ["clarify-scope", "report-generate"]
    assert not any("Router declared" in e for e in out.get("errors") or [])


def test_run_flow_subgraph_surfaces_empty_turn_without_flows():
    class EndOnly:
        def invoke(self, initial, config=None):  # noqa: ARG002
            return {
                **initial,
                "visited": [],
                "errors": [],
                "answer": "",
                "flow_messages": [],
                "flow_trace": [],
            }

    out = run_flow_subgraph(
        EndOnly(),
        {"mission_id": "test", "objective": "test"},
        {"message": "maintenance task room 201"},
    )
    assert any("Turn ended without executing any flow" in e for e in out.get("errors") or [])


def test_flow_subgraph_reprompt_then_clarification():
    seq = iter(
        [
            AIMessage(content="Flow 'quote-price': unsure"),
            AIMessage(content="still no tool"),
        ]
    )

    def route_invoker(messages, tool_spec, config):  # noqa: ARG001
        return next(seq)

    compiled = build_flow_subgraph(
        FLOW_MAP,
        tools=[],
        system_prompt="base",
        route_invoker=route_invoker,
    )
    out = run_flow_subgraph(
        compiled,
        {"mission_id": "test", "objective": "test"},
        {"message": "help"},
        skill_maps=[FLOW_MAP],
    )
    assert out.get("summary") == ROUTER_CLARIFICATION_REPLY
    assert any("Router declared no flow" in e for e in out.get("errors") or [])


def test_validate_results_skips_no_evidence_when_router_failed():
    state = {
        "errors": ["Router declared no flow (no FLOW: line); ending turn"],
        "evidence": [],
        "result": {},
        "mission": {"output_schema": {"properties": {"answer": {}}}},
    }
    assert _errors_explain_empty_evidence(state["errors"])
    out = validate_results(state)
    assert "No evidence collected" not in out["errors"]
    assert "Router declared no flow" in out["errors"][0]


def test_validate_results_still_flags_no_evidence_without_router_errors():
    out = validate_results(
        {
            "errors": [],
            "evidence": [],
            "result": {},
            "mission": {"output_schema": {"properties": {"answer": {}}}},
        }
    )
    assert "No evidence collected" in out["errors"]


def test_build_route_instruction_compact_lines():
    from runtime.subgraph.flow import derive_flows

    flows = derive_flows(FLOW_MAP)
    when = {b.goto: b.when for b in FLOW_MAP.node_by_id("route").branches}
    text = build_route_instruction(
        {"input_data": {}, "visited": []},
        flows,
        when,
    )
    assert "select_flow" in text
    assert "- check:" in text or "- check\n" in text or "- check:" in text
    assert json.dumps(when)[:20] not in text  # not dumping raw map JSON


def test_build_route_instruction_keeps_full_catalog_when_history_is_long():
    """Catalog ``when`` lines must not collapse to ~24 chars under long history."""
    from runtime.subgraph.flow import derive_flows
    from runtime.skills import load_skills

    skill_map = {s.name: s for s in load_skills()}["agent-lark"].map
    flows = derive_flows(skill_map)
    when = {b.goto: b.when for b in skill_map.node_by_id("route").branches}
    long_when = when.get("task-query") or ""
    cal_when = when.get("calendar-query") or ""
    assert len(long_when) > 60, "fixture: task-query when text should be descriptive"
    assert len(cal_when) > 40, "fixture: calendar-query when text should be descriptive"

    history_block = "\n".join(
        f"Turn {i}: user asked about something; assistant replied at length."
        for i in range(80)
    )
    state = {
        "input_data": {"message": "liệt kê tasklist của tôi", "chat_id": "oc_1"},
        "visited": [],
        "history_block": history_block,
    }
    text = build_route_instruction(state, flows, when)
    assert (
        route_instruction_token_count(state, flows, when)
        <= ROUTE_INSTRUCTION_TOKENS
    )
    assert "task-query:" in text
    assert "calendar-query:" in text
    assert long_when[:40] in text, "catalog must keep full task-query when text"
    assert cal_when[:30] in text, "catalog must keep calendar-query when text"
    assert "...task-query: input có mode == remi" not in text