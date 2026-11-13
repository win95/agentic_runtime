"""Plan gate tests (Phase 3a) — no LLM, no live infra.

The gate is exercised with fake tool-call requests against the REAL
reception skill map (loaded from disk), plus the node-wiring test that
the Mission Graph appends exactly one PlanGate for hotel_reception.
"""

import pathlib

import pytest
import yaml
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage

from app.services import mission_service
from runtime import deep_agent as deep_agent_module
from runtime.mission_graph import run_mission
from runtime.plan_gate import MAX_DENIALS, PlanGate, PlanGateExceededError, plan_gate
from runtime.skills.map_plan import parse_skill_map, render_plan
from seeds import load_missions

SKILL_PATH = pathlib.Path(__file__).resolve().parents[1] / "skills" / "reception" / "SKILL.md"

# The 14-tool MCP contract (docs/hotel-reception.md) — the map must
# cover every one, or the gate has no opinion about it (fail-open).
HOTEL_TOOLS = {
    "check_room_availability",
    "get_room_rates",
    "find_guest",
    "register_guest",
    "create_booking",
    "get_booking",
    "cancel_booking",
    "change_booking_dates",
    "list_staff",
    "create_task",
    "assign_task",
    "get_task",
    "search_policy_documents",
    "read_policy_document",
}


def _reception_map():
    """The reception map parsed straight from the skill file, WITHOUT the
    loader's known-tools validation: in tests no MCP server is connected,
    so load_skills() would drop all 14 MCP tools from the map nodes (the
    production node parses the map after load_mcp_tools(), so it keeps
    them — see the comment in test_hotel_reception.py)."""
    text = SKILL_PATH.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(text.split("---\n", 2)[1])
    skill_map = parse_skill_map(frontmatter, "reception")
    assert skill_map is not None
    return skill_map


def _request(name: str, call_id: str = "call-1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": call_id},
        tool=None, state={}, runtime=None,
    )


def _handler(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="ok", tool_call_id=request.tool_call["id"])


def _gate(**kwargs) -> PlanGate:
    return PlanGate([_reception_map()], mission_id="hotel_reception", **kwargs)


def _allowed(gate: PlanGate, name: str) -> bool:
    result = gate.wrap_tool_call(_request(name), _handler)
    return isinstance(result, ToolMessage) and result.content == "ok"


def test_gate_denies_skip_ahead_then_allows_after_list_staff():
    """The exact task-scenario symptom (2026-11-12): create_task before
    list_staff is denied, and the denial names list_staff."""
    gate = _gate()
    denied = gate.wrap_tool_call(_request("create_task"), _handler)
    assert isinstance(denied, ToolMessage)
    assert denied.status == "error"
    assert "create_task" in denied.content
    assert "list_staff" in denied.content
    assert gate.denied_count == 1
    # After the denied call the plan has not advanced: create_task is
    # still denied, list_staff is the way forward.
    assert not _allowed(gate, "create_task")
    assert _allowed(gate, "list_staff")
    assert _allowed(gate, "create_task")
    assert _allowed(gate, "assign_task")
    assert gate.denied_count == 2


def test_gate_allows_booking_flow_in_order():
    """Booking flow passes through the tool-less confirm-booking node:
    quote-price -> create-booking is a legal next step."""
    gate = _gate()
    assert _allowed(gate, "check_room_availability")
    assert _allowed(gate, "get_room_rates")
    assert _allowed(gate, "create_booking")


def test_gate_allows_each_flow_from_route():
    """From the start node every flow's first tool is reachable."""
    gate = _gate()
    assert _allowed(gate, "check_room_availability")
    gate = _gate()
    assert _allowed(gate, "list_staff")
    gate = _gate()
    assert _allowed(gate, "search_policy_documents")
    assert _allowed(gate, "read_policy_document")
    gate = _gate()
    assert _allowed(gate, "find_guest")
    assert _allowed(gate, "register_guest")
    assert _allowed(gate, "get_booking")
    gate = _gate()
    assert _allowed(gate, "get_task")


def test_gate_allows_reroute_after_a_flow():
    """After finishing the booking flow the agent can re-route to a
    policy lookup in the same execution (the classifier is always
    re-enterable)."""
    gate = _gate()
    assert _allowed(gate, "check_room_availability")
    assert _allowed(gate, "get_room_rates")
    assert _allowed(gate, "create_booking")
    assert _allowed(gate, "search_policy_documents")
    assert _allowed(gate, "read_policy_document")


def test_gate_allows_revisit():
    """Going back to an already-visited node is allowed."""
    gate = _gate()
    assert _allowed(gate, "list_staff")
    assert _allowed(gate, "create_task")
    assert _allowed(gate, "list_staff")


def test_gate_denies_assign_before_create():
    """assign_task before create_task is denied even after list_staff."""
    gate = _gate()
    assert _allowed(gate, "list_staff")
    denied = gate.wrap_tool_call(_request("assign_task"), _handler)
    assert denied.status == "error"
    assert "assign_task" in denied.content
    assert _allowed(gate, "create_task")
    assert _allowed(gate, "assign_task")


def test_gate_allows_tool_in_no_map():
    """A tool no map claims is allowed (fail-open with a warning) — a
    missing map entry must not break a live run."""
    gate = _gate()
    assert _allowed(gate, "some_future_tool")
    assert gate.denied_count == 0


def test_gate_cap_raises_after_max_denials():
    gate = _gate(max_denials=2)
    assert gate.wrap_tool_call(_request("create_task"), _handler).status == "error"
    assert gate.wrap_tool_call(_request("create_task"), _handler).status == "error"
    with pytest.raises(PlanGateExceededError, match="out-of-plan"):
        gate.wrap_tool_call(_request("create_task"), _handler)
    assert gate.denied_count == 3
    assert MAX_DENIALS == 5  # the production cap, pinned here


def test_reception_map_covers_all_14_contract_tools():
    """Coverage is enforced by test, not at runtime: every contract tool
    must appear in at least one map node, or the gate cannot see it."""
    skill_map = _reception_map()
    covered = {tool for node in skill_map.nodes for tool in node.tools}
    assert HOTEL_TOOLS <= covered


def test_plan_gate_scope():
    """The gate is hotel-only (pilot); fault_analysis keeps its advisory
    map, and a mission with no maps gets no gate."""
    skill_map = _reception_map()
    assert plan_gate([skill_map], "fault_analysis") is None
    assert plan_gate([skill_map], "system_admin") is None
    assert plan_gate([], "hotel_reception") is None
    assert plan_gate(None, "hotel_reception") is None
    assert isinstance(plan_gate([skill_map], "hotel_reception"), PlanGate)


def test_render_plan_enforced_header():
    skill_map = _reception_map()
    enforced = render_plan(skill_map, enforced=True)
    assert "ENFORCED" in enforced
    assert "plan gate" in enforced
    advisory = render_plan(skill_map)
    assert "may deviate" in advisory
    assert "ENFORCED" not in advisory


def test_deep_agent_node_adds_plan_gate_for_hotel(db_session, monkeypatch):
    """The Mission Graph appends exactly one PlanGate for hotel_reception,
    after the hook middleware and the built-in exclusion."""
    load_missions.main()
    mission = mission_service.get_mission(db_session, "hotel_reception")
    mission_dict = mission_service.mission_to_dict(mission)

    seen: dict = {}

    def fake_build_deep_agent(tools, system_prompt=None, skills=None, model_config=None,
                              middleware=None, backend=None):
        seen["middleware"] = list(middleware or [])
        seen["system_prompt"] = system_prompt or ""
        return object()

    def fake_run_deep_agent(agent, mission_arg, input_data, callbacks=None, files=None):
        return {
            "summary": "ok",
            "evidence": [{
                "source": "deep_agent", "kind": "summary",
                "summary": "ok", "stub": False,
            }],
            "actions_taken": [],
            "errors": [],
        }

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build_deep_agent)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run_deep_agent)

    final = run_mission(mission_dict, {"message": "hi"}, "exec_hotel_gate_wiring")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    middleware = seen.get("middleware") or []
    gates = [m for m in middleware if type(m).__name__ == "PlanGate"]
    assert len(gates) == 1
    assert gates[0].mission_id == "hotel_reception"
    # The rendered plan tells the model the gate is enforced.
    assert "ENFORCED" in seen["system_prompt"]
