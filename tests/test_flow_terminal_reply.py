"""Terminal flows with reply_tools end the turn without a second router call."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from runtime.skills.map_plan import MapBranch, MapNode, SkillMap
from runtime.subgraph.flow import (
    ROUTE_TOOL_NAME,
    build_flow_subgraph,
    run_flow_subgraph,
)


def _map():
    return SkillMap(
        skill_name="reply-skill",
        start="route",
        reply_tools=["lark_reply_message"],
        nodes=[
            MapNode(
                id="route",
                branches=[
                    MapBranch(when="task question", goto="task-query"),
                    MapBranch(when="general chat", goto="answer"),
                ],
            ),
            MapNode(
                id="task-query",
                goal="Query tasks",
                tools=["lark_list_tasklists", "lark_reply_message"],
                terminal=True,
            ),
            MapNode(
                id="answer",
                goal="General answer",
                tools=["lark_reply_message"],
                terminal=True,
            ),
        ],
    )


def _route_call(flow_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "r1",
                "name": ROUTE_TOOL_NAME,
                "args": {"flow_id": flow_id},
                "type": "tool_call",
            }
        ],
    )


def test_terminal_reply_skips_second_router():
    route_calls = 0

    def route_invoker(messages, tool_spec, config):  # noqa: ARG001
        nonlocal route_calls
        route_calls += 1
        if route_calls == 1:
            return _route_call("task-query")
        return _route_call("answer")

    def flow_executor(flow, state, messages):  # noqa: ARG001
        return messages + [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "t1",
                        "name": "lark_reply_message",
                        "args": {"message_id": "om_1", "text": "done"},
                    }
                ],
            ),
            ToolMessage(
                content=json.dumps({"message_id": "om_2", "stub": False}),
                tool_call_id="t1",
                name="lark_reply_message",
            ),
        ]

    compiled = build_flow_subgraph(
        _map(),
        tools=[],
        system_prompt="sys",
        flow_executor=flow_executor,
        route_invoker=route_invoker,
    )
    out = run_flow_subgraph(
        compiled,
        {"mission_id": "m", "objective": "o"},
        {"message": "tasks?"},
        skill_maps=[_map()],
    )
    assert route_calls == 1
    assert out.get("visited_nodes") == ["task-query"]


def test_terminal_flow_without_reply_still_reroutes():
    """Terminal flow with reply_tools configured but no reply sent -> router again."""
    route_calls = []

    def route_invoker(messages, tool_spec, config):  # noqa: ARG001
        route_calls.append(1)
        if len(route_calls) == 1:
            return _route_call("task-query")
        return _route_call("END")

    def flow_executor(flow, state, messages):  # noqa: ARG001
        return messages + [AIMessage(content="forgot to call reply tool")]

    compiled = build_flow_subgraph(
        _map(),
        tools=[],
        system_prompt="sys",
        flow_executor=flow_executor,
        route_invoker=route_invoker,
    )
    out = run_flow_subgraph(
        compiled,
        {"mission_id": "m", "objective": "o"},
        {"message": "tasks?"},
        skill_maps=[_map()],
    )
    assert len(route_calls) == 2
    assert out.get("visited_nodes") == ["task-query"]