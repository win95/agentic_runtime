"""Human-in-the-loop interrupts (docs/interrupt.md).

Main-feature coverage only: config parsing, resume command building,
the deep-agent pause/resume round trip with a fake model, and the
mission-graph pause/resume round trip with a stubbed agent runner.
"""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr

from runtime import deep_agent as deep_agent_module
from runtime.deep_agent import build_deep_agent, run_deep_agent
from runtime.interrupt import (
    ALLOWED_DECISIONS,
    build_resume_command,
    extract_interrupt_requests,
    interrupt_payload,
    parse_interrupt_on,
)
from runtime.mission_graph import build_mission_graph, run_mission

MISSION = {
    "mission_id": "fault_analysis",
    "agent_id": "fault_investigator",
    "objective": "Test interrupt_on",
    "capabilities": ["fault_analysis"],
    "execution": {"mode": "autonomous", "interrupt_on": {"wipe_cache": True}},
    "output_schema": {"properties": {"summary": {}, "answer": {}}},
}
INPUT = {"asset_id": "O21"}


# --- Config parsing -------------------------------------------------


def test_parse_interrupt_on_none():
    assert parse_interrupt_on(None) is None
    assert parse_interrupt_on({}) is None


def test_parse_interrupt_on_bool_and_config():
    parsed = parse_interrupt_on(
        {
            "delete_mission": True,
            "cancel_execution": {
                "allowed_decisions": ["approve", "reject"],
                "description": "Irreversible.",
            },
        }
    )
    assert parsed == {
        "delete_mission": True,
        "cancel_execution": {
            "allowed_decisions": ["approve", "reject"],
            "description": "Irreversible.",
        },
    }


def test_parse_interrupt_on_rejects_bad_shapes():
    with pytest.raises(ValueError):
        parse_interrupt_on("wipe_cache")
    with pytest.raises(ValueError):
        parse_interrupt_on({"wipe_cache": "yes"})
    with pytest.raises(ValueError):
        parse_interrupt_on({"wipe_cache": {"when": "never"}})
    with pytest.raises(ValueError):
        parse_interrupt_on({"wipe_cache": {"allowed_decisions": ["edit"]}})
    with pytest.raises(ValueError):
        parse_interrupt_on({"wipe_cache": {"allowed_decisions": []}})


# --- Resume command + extraction -------------------------------------


def test_build_resume_command_approve_and_reject():
    # Command serializes the resume value to a plain dict (it must be
    # checkpoint-serializable); the middleware parses it back on resume.
    approve = build_resume_command("approve", 2)
    assert approve.resume == {
        "decisions": [{"type": "approve"}, {"type": "approve"}]
    }
    reject = build_resume_command("reject", 1)
    assert reject.resume == {
        "decisions": [{"type": "reject", "message": "Rejected by human reviewer"}]
    }
    with pytest.raises(ValueError):
        build_resume_command("edit", 1)
    with pytest.raises(ValueError):
        build_resume_command("approve", 0)


def test_extract_interrupt_requests_from_dict_shape():
    value = {
        "action_requests": [
            {"name": "wipe_cache", "args": {"path": "/tmp/x"}},
            {"name": "other_tool", "args": {}},
        ]
    }
    assert extract_interrupt_requests(value) == [
        {"tool": "wipe_cache", "args": {"path": "/tmp/x"}},
        {"tool": "other_tool", "args": {}},
    ]
    assert extract_interrupt_requests({"nope": True}) == []
    assert interrupt_payload([{"tool": "wipe_cache", "args": {}}]) == {
        "requests": [{"tool": "wipe_cache", "args": {}}],
        "allowed_decisions": list(ALLOWED_DECISIONS),
    }


# --- Deep-agent round trip (real HITL middleware, fake model) --------


class _ScriptedChatModel(BaseChatModel):
    """Returns canned messages in order (this langchain-core's
    FakeListChatModel only takes strings, not tool-call messages)."""

    _responses: list = PrivateAttr(default_factory=list)

    def __init__(self, responses, **kwargs):
        super().__init__(**kwargs)
        self._responses = list(responses)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=self._responses.pop(0))]
        )

    def bind_tools(self, tools, **kwargs):
        # BaseChatModel.bind_tools raises NotImplementedError; the
        # scripted model ignores the binding and returns its canned
        # messages (same pattern as the repo's fake models).
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"


def _tool_call_message():
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "wipe_cache",
                "args": {"path": "/tmp/x"},
            }
        ],
    )


def _agent_with_fake_model(monkeypatch, responses, checkpointer):
    calls: list[str] = []

    @tool
    def wipe_cache(path: str) -> str:
        """Wipe a cache path (test tool)."""
        calls.append(path)
        return "wiped"

    model = _ScriptedChatModel(responses=responses)
    monkeypatch.setattr(deep_agent_module, "get_chat_model", lambda: model)
    agent = build_deep_agent(
        [wipe_cache],
        system_prompt="You are a test agent.",
        interrupt_on={"wipe_cache": True},
        checkpointer=checkpointer,
    )
    return agent, calls


def test_deep_agent_pauses_then_approves(monkeypatch):
    checkpointer = InMemorySaver()
    agent, calls = _agent_with_fake_model(
        monkeypatch, [_tool_call_message(), AIMessage(content="done")], checkpointer
    )
    mission = {"mission_id": "test", "objective": "test"}
    thread_id = "exec_interrupt_approve:deep-agent"
    paused = run_deep_agent(
        agent, mission, {}, checkpointer_thread_id=thread_id
    )
    assert paused.get("interrupted") is True
    assert paused["interrupt_requests"] == [
        {"tool": "wipe_cache", "args": {"path": "/tmp/x"}}
    ]
    assert calls == []  # gated tool did NOT run before approval
    resumed = run_deep_agent(
        agent,
        mission,
        {},
        resume_decision="approve",
        resume_request_count=1,
        checkpointer_thread_id=thread_id,
    )
    assert resumed.get("interrupted") is not True
    assert calls == ["/tmp/x"]
    assert resumed.get("summary") == "done"


def test_deep_agent_reject_skips_tool(monkeypatch):
    checkpointer = InMemorySaver()
    agent, calls = _agent_with_fake_model(
        monkeypatch, [_tool_call_message(), AIMessage(content="skipped")], checkpointer
    )
    mission = {"mission_id": "test", "objective": "test"}
    thread_id = "exec_interrupt_reject:deep-agent"
    paused = run_deep_agent(
        agent, mission, {}, checkpointer_thread_id=thread_id
    )
    assert paused.get("interrupted") is True
    resumed = run_deep_agent(
        agent,
        mission,
        {},
        resume_decision="reject",
        resume_request_count=1,
        checkpointer_thread_id=thread_id,
    )
    assert resumed.get("interrupted") is not True
    assert calls == []
    assert resumed.get("summary") == "skipped"


# --- Mission-graph round trip (stubbed agent runner) -----------------


def _stubbed_agent(monkeypatch):
    """Fake build/run: first call pauses, resume call finishes."""
    calls = {"n": 0}

    def fake_build(*args, **kwargs):
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None, **kwargs):
        calls["n"] += 1
        if kwargs.get("resume_decision") is None:
            return {
                "summary": "",
                "evidence": [],
                "actions_taken": [],
                "errors": [],
                "interrupted": True,
                "interrupt_requests": [
                    {"tool": "wipe_cache", "args": {"path": "/tmp/x"}}
                ],
            }
        assert kwargs["resume_decision"] == "approve"
        return {
            "summary": "resumed and done",
            "evidence": [
                {"source": "deep_agent", "kind": "summary", "summary": "done", "stub": False}
            ],
            "actions_taken": ["wipe_cache"],
            "errors": [],
        }

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)
    return calls


def test_mission_pauses_then_resumes(monkeypatch):
    _stubbed_agent(monkeypatch)
    paused = run_mission(dict(MISSION), dict(INPUT), "exec_interrupt_mission")
    assert paused["status"] == "INTERRUPTED", f"errors: {paused.get('errors')}"
    assert paused["interrupt"]["requests"] == [
        {"tool": "wipe_cache", "args": {"path": "/tmp/x"}}
    ]
    resumed = run_mission(
        dict(MISSION),
        dict(INPUT),
        "exec_interrupt_mission",
        resume_decision="approve",
    )
    assert resumed["status"] == "SUCCEEDED", f"errors: {resumed.get('errors')}"
    assert resumed["result"]["answer"] == "resumed and done"


def test_mission_flow_plus_interrupt_fails_fast(monkeypatch):
    mission = dict(MISSION)
    mission["execution"] = {
        "mode": "autonomous",
        "interrupt_on": {"wipe_cache": True},
    }
    # Force the flow branch: a mapped skill + harness flag. The import
    # happens inside deep_agent_node, so patch the source module.
    monkeypatch.setattr(
        "runtime.subgraph.flow.use_flow_subgraph", lambda *a, **k: True
    )
    final = run_mission(mission, dict(INPUT), "exec_interrupt_flow")
    assert final["status"] == "FAILED"
    assert any("flow_subgraph" in e for e in final.get("errors") or [])


def test_graph_compiles_with_checkpointer():
    graph = build_mission_graph(checkpointer=InMemorySaver())
    assert graph is not None
    assert hasattr(graph, "invoke")


# --- Resume endpoint ------------------------------------------------


def test_resume_endpoint(client, db_session, monkeypatch):
    from app import models
    from app.services import execution_service

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        execution_service,
        "dispatch_resume",
        lambda execution_id, decision: calls.append((execution_id, decision)),
    )
    db_session.add(
        models.Execution(
            id="exec_resume_1",
            mission_id="fault_analysis",
            input={},
            status="INTERRUPTED",
            user_id=None,
        )
    )
    db_session.commit()
    # Bad decision -> 422 (validated before the handler runs).
    response = client.post(
        "/executions/exec_resume_1/resume", json={"decision": "edit"}
    )
    assert response.status_code == 422
    response = client.post(
        "/executions/exec_resume_1/resume", json={"decision": "approve"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "QUEUED"
    assert calls == [("exec_resume_1", "approve")]
    # No longer INTERRUPTED -> 409.
    response = client.post(
        "/executions/exec_resume_1/resume", json={"decision": "approve"}
    )
    assert response.status_code == 409
    # Unknown execution -> 404.
    response = client.post("/executions/nope/resume", json={"decision": "approve"})
    assert response.status_code == 404
