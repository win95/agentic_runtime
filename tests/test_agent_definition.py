"""MDA response_format / context_schema (docs/agent-definition.md).

Main-feature coverage only: schema resolution, context validation,
the build wiring, the structured-response fold, and the mission-level
result merge.
"""

import pytest
from langchain_core.messages import AIMessage

from runtime import deep_agent as deep_agent_module
from runtime.agent_context import (
    resolve_response_format,
    validate_agent_context,
)
from runtime.deep_agent import build_deep_agent, run_deep_agent
from runtime.mission_graph import run_mission

SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "answer": {"type": "string"}},
}


# --- Resolution + validation ----------------------------------------


def test_resolve_response_format():
    assert resolve_response_format(None, SCHEMA) is None
    assert resolve_response_format(False, SCHEMA) is None
    assert resolve_response_format(True, SCHEMA) == SCHEMA
    explicit = {"type": "object", "properties": {"answer": {}}}
    assert resolve_response_format(explicit, SCHEMA) == explicit
    with pytest.raises(ValueError):
        resolve_response_format(True, {"type": "object"})
    with pytest.raises(ValueError):
        resolve_response_format(True, None)
    with pytest.raises(ValueError):
        resolve_response_format({"type": "string"}, SCHEMA)
    with pytest.raises(ValueError):
        resolve_response_format("summary", SCHEMA)


def test_validate_agent_context():
    good = {
        "mission_id": "fault_analysis",
        "execution_id": "exec_1",
        "tenant": "default",
        "capabilities": ["fault_analysis"],
        "input": {"asset_id": "O21"},
        "execution_config": {"mode": "hybrid"},
        "session_id": "",
        "user_id": None,
    }
    assert validate_agent_context(good) == []
    bad = dict(good, execution_id=123, capabilities="fault_analysis", input=[])
    problems = validate_agent_context(bad)
    assert len(problems) == 3
    assert validate_agent_context("nope") == ["agent context must be a dict"]


# --- Build wiring ----------------------------------------------------


def test_build_deep_agent_passes_response_format(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(deep_agent_module, "get_chat_model", lambda: object())
    import deepagents

    def fake_create(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create)
    build_deep_agent([], system_prompt="x", response_format=SCHEMA)
    assert seen.get("response_format") == SCHEMA
    assert seen.get("interrupt_on") is None
    assert seen.get("checkpointer") is None


# --- Structured-response fold ---------------------------------------


class _StubAgent:
    def __init__(self, result):
        self._result = result

    def invoke(self, input_payload, config=None):
        return self._result


def test_run_deep_agent_folds_structured_response():
    agent = _StubAgent(
        {
            "messages": [AIMessage(content="prose summary")],
            "structured_response": {"summary": "structured", "answer": "A"},
        }
    )
    outcome = run_deep_agent(agent, {"mission_id": "t", "objective": "t"}, {})
    assert outcome["summary"] == "structured"
    assert outcome["structured_response"] == {"summary": "structured", "answer": "A"}


def test_run_deep_agent_without_structured_response():
    agent = _StubAgent({"messages": [AIMessage(content="prose only")]})
    outcome = run_deep_agent(agent, {"mission_id": "t", "objective": "t"}, {})
    assert outcome["summary"] == "prose only"
    assert "structured_response" not in outcome


# --- Mission level ----------------------------------------------------


def _stubbed_structured_run(monkeypatch):
    def fake_build(*args, **kwargs):
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None, **kwargs):
        return {
            "summary": "prose summary",
            "evidence": [
                {"source": "deep_agent", "kind": "summary", "summary": "x", "stub": False}
            ],
            "actions_taken": [],
            "errors": [],
            "structured_response": {
                "summary": "structured summary",
                "answer": "structured answer",
            },
        }

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)


def test_mission_structured_response_wins(monkeypatch):
    _stubbed_structured_run(monkeypatch)
    mission = {
        "mission_id": "fault_analysis",
        "agent_id": "fault_investigator",
        "objective": "Test response_format",
        "capabilities": ["fault_analysis"],
        "execution": {"mode": "autonomous", "response_format": True},
        "output_schema": {"properties": {"summary": {}, "answer": {}}},
    }
    final = run_mission(mission, {"asset_id": "O21"}, "exec_structured_1")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert final["result"]["answer"] == "structured answer"
    assert final["result"]["summary"] == "structured summary"


def test_mission_flow_plus_response_format_fails_fast(monkeypatch):
    _stubbed_structured_run(monkeypatch)
    mission = {
        "mission_id": "fault_analysis",
        "agent_id": "fault_investigator",
        "objective": "Test response_format",
        "capabilities": ["fault_analysis"],
        "execution": {"mode": "autonomous", "response_format": True},
        "output_schema": {"properties": {"summary": {}, "answer": {}}},
    }
    monkeypatch.setattr(
        "runtime.subgraph.flow.use_flow_subgraph", lambda *a, **k: True
    )
    final = run_mission(mission, {"asset_id": "O21"}, "exec_structured_flow")
    assert final["status"] == "FAILED"
    assert any("response_format" in e for e in final.get("errors") or [])
