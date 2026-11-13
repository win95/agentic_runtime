"""Skill subgraph tests (Phase 3) — no LLM, no live infra.

The router and the NEXT protocol are tested through the node-executor
seam (fake executors returning canned messages); one integration test
drives the REAL node functions with a fake model / fake node agents
(monkeypatched build_deep_agent / build_chat_model) to prove the
tool-filtering and the end-to-end outcome; one wiring test proves the
Mission Graph takes the subgraph path for hotel_reception.
"""

import re

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool

from app.services import mission_service
from runtime import deep_agent as deep_agent_module
from runtime import skill_subgraph as skill_subgraph_module
from runtime.mission_graph import run_mission
from runtime.skill_subgraph import (
    MAX_NODE_STEPS,
    allowed_next,
    build_skill_subgraph,
    parse_next,
    run_skill_subgraph,
    tools_for_node,
    use_skill_subgraph,
)
from runtime.skills.map_plan import MapBranch, MapNode, SkillMap
from seeds import load_missions


def _map(nodes, start="route"):
    return SkillMap(skill_name="test-flow", start=start, nodes=[
        MapNode(id=n["id"], goal=n.get("goal", n["id"]),
                tools=n.get("tools", []),
                branches=[MapBranch(when=b["when"], goto=b["goto"])
                          for b in n.get("branches", [])])
        for n in nodes
    ])


FLOW_MAP = _map([
    {"id": "route", "branches": [
        {"when": "book", "goto": "check"},
        {"when": "task", "goto": "find"},
    ]},
    {"id": "check", "tools": ["t_check"]},
    {"id": "quote", "tools": ["t_quote"]},
    {"id": "find", "tools": ["t_find"]},
    {"id": "create", "tools": ["t_create"]},
    {"id": "report"},
])


def _executor_script(script):
    """A node executor that returns canned messages per node id.

    ``script`` maps node id -> (extra messages to append, declared NEXT).
    The declared NEXT is appended as a text line to the last message.
    """
    def execute(node, state, messages):
        extra, nxt = script.get(node.id, ([], None))
        out = list(messages) + list(extra)
        if nxt is not None:
            out.append(AIMessage(content=f"NEXT: {nxt}"))
        return out

    return execute


def _run(skill_map, script, **kwargs):
    compiled = build_skill_subgraph(
        skill_map, tools=[], system_prompt="base",
        node_executor=_executor_script(script), **kwargs,
    )
    return run_skill_subgraph(
        compiled,
        {"mission_id": "test", "objective": "test"},
        {"message": "hi"},
        skill_maps=[skill_map],
    )


# --- NEXT parsing ----------------------------------------------------

def test_parse_next():
    assert parse_next("done\nNEXT: check") == "check"
    assert parse_next("NEXT: END") == "END"
    assert parse_next("no next line here") is None
    assert parse_next("") is None


# --- allowed transitions ---------------------------------------------

def test_allowed_next():
    assert allowed_next(FLOW_MAP, "route") == {"check", "find"}
    assert allowed_next(FLOW_MAP, "check") == {"quote"}
    assert allowed_next(FLOW_MAP, "report") == set()
    # Resume rule: create is reachable from route only with history.
    assert "create" not in allowed_next(FLOW_MAP, "route", session_turns=0,
                                         resume_nodes={"create"})
    assert "create" in allowed_next(FLOW_MAP, "route", session_turns=2,
                                     resume_nodes={"create"})


# --- router behaviour (fake executors) -------------------------------

def test_router_follows_declared_next():
    result = _run(FLOW_MAP, {
        "route": ([AIMessage(content="routing")], "check"),
        "check": ([AIMessage(content="checked")], "quote"),
        "quote": ([AIMessage(content="quoted")], "END"),
    })
    assert result["visited_nodes"] == ["route", "check", "quote"]


def test_router_falls_back_to_next_in_order_on_bad_next():
    result = _run(FLOW_MAP, {
        "route": ([AIMessage(content="routing")], "create"),  # not allowed
        "check": ([AIMessage(content="checked")], "END"),
    })
    # Invalid NEXT is ignored: route falls back to the next node (check).
    assert result["visited_nodes"] == ["route", "check"]


def test_router_resume_rule_needs_history():
    script = {
        "route": ([AIMessage(content="confirming")], "create"),
        # Without the resume jump the fallback stops at check, so the
        # run never reaches create by walking the map in order.
        "check": ([AIMessage(content="checked")], "END"),
        "create": ([AIMessage(content="created")], "END"),
    }
    # No prior turns: the resume jump is refused, route falls back.
    result = _run(FLOW_MAP, script)
    assert "create" not in result["visited_nodes"]
    # With prior turns: the same declaration resumes at create.
    compiled = build_skill_subgraph(
        FLOW_MAP, tools=[], system_prompt="base",
        node_executor=_executor_script(script),
        resume_nodes={"create"},
    )
    result = run_skill_subgraph(
        compiled,
        {"mission_id": "test", "objective": "test"},
        {"message": "ok"},
        session_history={"turns": [{"id": "t1"}]},
        skill_maps=[FLOW_MAP],
    )
    assert result["visited_nodes"] == ["route", "create"]


def test_router_never_books_before_confirmation():
    """The confirmation safety net, in code: on a fresh turn the router
    never enters create-booking — neither by declaration nor by
    falling through from confirm-booking. The turn ends at confirm."""
    confirm_map = _map([
        {"id": "route", "branches": [{"when": "book", "goto": "confirm"}]},
        {"id": "confirm"},
        {"id": "create", "tools": ["t_create"]},
        {"id": "report"},
    ])
    script = {
        "route": ([AIMessage(content="routing")], "confirm"),
        # The model forgets to stop and declares the booking node.
        "confirm": ([AIMessage(content="please confirm")], "create"),
        "create": ([AIMessage(content="created")], "END"),
    }
    result = _run(confirm_map, script, resume_nodes={"create"})
    assert result["visited_nodes"] == ["route", "confirm"]
    # With prior turns (the confirmation turn) the same declaration
    # is allowed through.
    compiled = build_skill_subgraph(
        confirm_map, tools=[], system_prompt="base",
        node_executor=_executor_script(script),
        resume_nodes={"create"},
    )
    result = run_skill_subgraph(
        compiled,
        {"mission_id": "test", "objective": "test"},
        {"message": "ok, đặt đi"},
        session_history={"turns": [{"id": "t1"}]},
        skill_maps=[confirm_map],
    )
    assert result["visited_nodes"] == ["route", "confirm", "create"]


def test_step_cap_ends_a_loop():
    loop_map = _map([
        {"id": "route", "branches": [{"when": "again", "goto": "check"}]},
        {"id": "check", "branches": [{"when": "again", "goto": "route"}]},
    ])
    result = _run(loop_map, {
        "route": ([AIMessage(content="r")], "check"),
        "check": ([AIMessage(content="c")], "route"),
    }, max_steps=3)
    assert "step cap" in " ".join(result["errors"])
    assert len(result["visited_nodes"]) == 3


# --- tool filtering ---------------------------------------------------

@langchain_tool
def t_check() -> str:
    """Check availability."""
    return "ok"


@langchain_tool
def t_quote() -> str:
    """Quote a price."""
    return "ok"


def test_tools_for_node_filters_by_name():
    tools = tools_for_node([t_check, t_quote], ["t_quote", "t_missing"])
    assert [t.name for t in tools] == ["t_quote"]


# --- end-to-end with fake model / fake node agents -------------------

class _FakeNodeAgent:
    def __init__(self, canned):
        self.canned = canned

    def invoke(self, payload, config=None):
        return {"messages": list(payload["messages"]) + list(self.canned)}


class _FakeModel:
    def __init__(self, canned_by_node):
        self.canned_by_node = canned_by_node

    def invoke(self, messages, config=None):
        last = messages[-1].content if messages else ""
        match = re.search(r"Step '([A-Za-z0-9_-]+)'", last)
        node_id = match.group(1) if match else "route"
        return self.canned_by_node[node_id]


def _install_fakes(monkeypatch, canned_by_node):
    def fake_build_deep_agent(tools, system_prompt=None, skills=None,
                              model_config=None, middleware=None, backend=None):
        match = re.search(r"node '([A-Za-z0-9_-]+)'", system_prompt or "")
        node_id = match.group(1) if match else "route"
        return _FakeNodeAgent(canned_by_node[node_id])

    monkeypatch.setattr(
        skill_subgraph_module, "build_deep_agent", fake_build_deep_agent)
    monkeypatch.setattr(
        skill_subgraph_module, "build_chat_model",
        lambda cfg=None: _FakeModel(canned_by_node))


def test_run_end_to_end_with_fake_model(monkeypatch):
    """The real node functions + router, with a fake model: the check
    node's agent gets ONLY t_check, the tool call lands in
    actions_taken, and the plan trace records the visited nodes."""
    small = _map([
        {"id": "route"},
        {"id": "check", "tools": ["t_check"]},
        {"id": "report"},
    ])
    canned = {
        "route": AIMessage(content="guest wants to book\nNEXT: check"),
        "check": [
            AIMessage(content="", tool_calls=[
                {"name": "t_check", "args": {}, "id": "c1", "type": "tool_call"},
            ]),
            ToolMessage(content='{"available": true}', tool_call_id="c1"),
            AIMessage(content="room available\nNEXT: report"),
        ],
        "report": AIMessage(content="all done\nNEXT: END"),
    }
    _install_fakes(monkeypatch, canned)
    seen_tools: dict = {}

    orig_build = skill_subgraph_module.build_deep_agent

    def spy_build(tools, **kwargs):
        match = re.search(r"node '([A-Za-z0-9_-]+)'", kwargs.get("system_prompt") or "")
        seen_tools[match.group(1)] = [t.name for t in tools]
        return orig_build(tools, **kwargs)

    monkeypatch.setattr(skill_subgraph_module, "build_deep_agent", spy_build)
    compiled = build_skill_subgraph(
        small, tools=[t_check], system_prompt="base")
    result = run_skill_subgraph(
        compiled, {"mission_id": "test", "objective": "test"},
        {"message": "book a room"}, skill_maps=[small])
    assert seen_tools == {"check": ["t_check"]}
    assert result["actions_taken"] == ["t_check"]
    assert result["visited_nodes"] == ["route", "check", "report"]
    # plan_trace only sees nodes with tool calls: route/report are
    # tool-less, so only check appears there.
    assert result["plan_trace"]["test-flow"]["visited"] == ["check"]
    assert result["summary"] == "all done\nNEXT: END"
    assert result["errors"] == []


# --- scope decision ---------------------------------------------------

def test_use_skill_subgraph():
    assert use_skill_subgraph("hotel_reception", ["reception"]) is True
    assert use_skill_subgraph("fault_analysis", ["data-access"]) is False
    assert use_skill_subgraph("hotel_reception", []) is False
    assert use_skill_subgraph(None, ["reception"]) is False


# --- Mission Graph wiring --------------------------------------------

def test_deep_agent_node_runs_subgraph_for_hotel(db_session, monkeypatch):
    """hotel_reception compiles its map instead of running one free-form
    agent: the builder gets the reception map, the node middleware
    carries the hook middleware + built-in exclusion (and no plan gate
    — the subgraph IS the enforcement), and the confirmation-turn
    resume node is create-booking."""
    load_missions.main()
    mission = mission_service.get_mission(db_session, "hotel_reception")
    mission_dict = mission_service.mission_to_dict(mission)

    seen: dict = {}

    def fake_build_subgraph(skill_map, tools, system_prompt=None, **kwargs):
        seen["skill_map"] = skill_map
        seen["middleware"] = list(kwargs.get("middleware") or [])
        seen["resume_nodes"] = kwargs.get("resume_nodes")
        seen["system_prompt"] = system_prompt or ""
        return object()

    def fake_run_subgraph(compiled, mission_arg, input_data, **kwargs):
        seen["run_kwargs"] = kwargs
        return {
            "summary": "ok",
            "evidence": [{
                "source": "deep_agent", "kind": "summary",
                "summary": "ok", "stub": False,
            }],
            "actions_taken": [],
            "errors": [],
        }

    monkeypatch.setattr(
        skill_subgraph_module, "build_skill_subgraph", fake_build_subgraph)
    monkeypatch.setattr(
        skill_subgraph_module, "run_skill_subgraph", fake_run_subgraph)

    final = run_mission(mission_dict, {"message": "hi"}, "exec_hotel_subgraph")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert seen["skill_map"].skill_name == "reception"
    middleware = seen["middleware"]
    assert any(type(m).__name__ == "HookMiddleware" for m in middleware)
    exclusions = [m for m in middleware
                  if type(m).__name__ == "_ToolExclusionMiddleware"]
    assert len(exclusions) == 1
    assert exclusions[0]._excluded == deep_agent_module.BUILTIN_TOOL_NAMES
    assert not any(type(m).__name__ == "PlanGate" for m in middleware)
    assert seen["resume_nodes"] == {"create-booking"}
    # The map is compiled, not rendered as prose: no ENFORCED header.
    assert "ENFORCED" not in seen["system_prompt"]
    assert seen["run_kwargs"]["skill_maps"][0].skill_name == "reception"
    assert MAX_NODE_STEPS == 12  # the loop cap, pinned here
