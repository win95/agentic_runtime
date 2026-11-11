"""Item 5 (2026-11-11): actions_taken must be grounded in real tool
outcomes, not in calls made or model prose.

A refused/failed tool call (cancelled:false, refused:true, an "error"
value, or an error ToolMessage) is NOT listed as taken — it goes into
the errors list as a factual line, and deep_agent_node surfaces those
errors in the result (tool_failures) even when the run SUCCEEDs.
No live LLM or DB needed: the agent is a fake returning canned
AIMessage / ToolMessage pairs.
"""

import json

from langchain_core.messages import AIMessage, ToolMessage

from runtime import deep_agent as deep_agent_module
from runtime.deep_agent import run_deep_agent
from runtime.mission_graph import deep_agent_node

MISSION = {
    "mission_id": "system_admin",
    "agent_id": "admin_bot",
    "objective": "Administer the platform.",
    "input_schema": {
        "type": "object",
        "properties": {"request": {"type": "string"}},
        "required": ["request"],
    },
    "execution": {"mode": "autonomous"},
    "capabilities": ["system_admin"],
    "output_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "actions_taken": {"type": "array"},
            "evidence": {"type": "array"},
        },
    },
}


class _FakeAgent:
    def __init__(self, messages):
        self._messages = messages

    def invoke(self, state, config=None):  # noqa: ARG002
        return {"messages": self._messages}


def _ai_with_call(name, call_id, summary="Done."):
    return AIMessage(
        content=summary,
        tool_calls=[
            {"name": name, "args": {}, "id": call_id, "type": "tool_call"}
        ],
    )


def _tool_result(call_id, payload, status="success"):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return ToolMessage(content=content, tool_call_id=call_id, status=status)


def test_cancelled_false_not_listed_as_taken():
    """The reported bug: the bot claimed "cancelled successfully"
    when the tool returned cancelled:false."""
    agent = _FakeAgent([
        _ai_with_call("cancel_execution", "call-1", ""),
        _tool_result("call-1", {
            "execution_id": "exec_done", "status": "SUCCEEDED",
            "cancelled": False,
        }),
        # Real runs end with the agent's closing message.
        AIMessage(content="Cancelled successfully."),
    ])
    outcome = run_deep_agent(agent, MISSION, {"request": "cancel exec_done"})
    assert outcome["actions_taken"] == []
    assert len(outcome["errors"]) == 1
    assert "cancel_execution" in outcome["errors"][0]
    assert "cancelled: false" in outcome["errors"][0]
    # The prose summary is not rewritten — only the structured
    # outcome is authoritative.
    assert "Cancelled successfully" in outcome["summary"]


def test_refused_delete_not_listed_as_taken():
    agent = _FakeAgent([
        _ai_with_call("delete_mission", "call-1", "Deleted the mission."),
        _tool_result("call-1", {
            "deleted": False, "refused": True,
            "reason": "Mission has 2 execution(s)", "mission_id": "m1",
        }),
    ])
    outcome = run_deep_agent(agent, MISSION, {"request": "delete m1"})
    assert outcome["actions_taken"] == []
    assert len(outcome["errors"]) == 1
    assert "delete_mission" in outcome["errors"][0]
    assert "2 execution(s)" in outcome["errors"][0]


def test_error_result_not_listed_as_taken():
    agent = _FakeAgent([
        _ai_with_call("dispatch_execution", "call-1"),
        _tool_result("call-1", {"error": "Mission not found"}),
    ])
    outcome = run_deep_agent(agent, MISSION, {"request": "run nope"})
    assert outcome["actions_taken"] == []
    assert "Mission not found" in outcome["errors"][0]


def test_error_status_tool_message_not_listed_as_taken():
    agent = _FakeAgent([
        _ai_with_call("get_sensor_data", "call-1"),
        _tool_result("call-1", "Tool 'get_sensor_data' failed: boom", status="error"),
    ])
    outcome = run_deep_agent(agent, MISSION, {"request": "x"})
    assert outcome["actions_taken"] == []
    assert "boom" in outcome["errors"][0]


def test_successful_and_unanswered_calls_still_listed():
    agent = _FakeAgent([
        _ai_with_call("list_missions", "call-1"),
        _tool_result("call-1", {"missions": []}),
        _ai_with_call("system_health", "call-2"),
        # No ToolMessage for call-2 (e.g. a mock agent): nothing
        # contradicts the call, so it still counts.
    ])
    outcome = run_deep_agent(agent, MISSION, {"request": "x"})
    assert outcome["actions_taken"] == ["list_missions", "system_health"]
    assert outcome["errors"] == []


def test_node_surfaces_tool_failures_in_result(monkeypatch):
    """deep_agent_node puts grounded failures in result["tool_failures"]
    (visible even on SUCCEEDED runs) instead of failing validation
    via state errors."""
    outcome = {
        "summary": "Cancelled the execution.",
        "evidence": [
            {"source": "deep_agent", "kind": "summary",
             "summary": "Cancelled the execution.", "stub": False}
        ],
        "actions_taken": [],
        "errors": ["Tool 'cancel_execution' reported failure: cancelled: false"],
    }

    def fake_build(*args, **kwargs):
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None):
        return dict(outcome)

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)
    state = {
        "execution_id": "exec-ground-1", "mission_id": "system_admin",
        "mission": MISSION, "input": {"request": "cancel exec_done"},
        "mode": "autonomous", "plan": [], "evidence": [], "result": None,
        "status": "RUNNING", "errors": [], "notes": [],
    }
    out = deep_agent_node(state)
    assert out.get("errors") == []
    assert out["result"]["tool_failures"] == [
        "Tool 'cancel_execution' reported failure: cancelled: false"
    ]
    assert out["result"]["actions_taken"] == []
    assert any("Tool failures reported" in note for note in out["notes"])
