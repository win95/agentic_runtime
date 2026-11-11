"""Session-tracing tests — LangSmith metadata/tags + MLflow run tags.

No LLM, no live services: the LangSmith side is asserted on the
``RunnableConfig`` the graph/agent invoke actually receives, and the
MLflow side is asserted against a fake ``mlflow`` module (the real
client is never contacted).
"""

import contextlib
import sys
import types

import pytest

import runtime.deep_agent as deep_agent_module
import runtime.mission_graph as mission_graph_module
import runtime.tracing as tracing_module
import seeds.load_missions as load_missions
from app.services import mission_service
from runtime.mission_graph import run_mission
from runtime.tracing import (
    build_run_metadata,
    build_run_tags,
    mlflow_execution_run,
)

# --- Builders ---


def test_build_run_metadata_includes_session_and_agent():
    metadata = build_run_metadata(
        "exec-1", "fault_analysis", "fault_investigator", "sess-abc"
    )
    assert metadata == {
        "execution_id": "exec-1",
        "mission_id": "fault_analysis",
        "agent_id": "fault_investigator",
        "session_id": "sess-abc",
    }


def test_build_run_metadata_omits_absent_values():
    metadata = build_run_metadata("exec-1", "fault_analysis")
    assert metadata == {"execution_id": "exec-1", "mission_id": "fault_analysis"}
    assert build_run_tags("fault_analysis") == ["mission:fault_analysis"]


def test_build_run_tags_form():
    assert build_run_tags("fault_analysis", "fault_investigator", "sess-abc") == [
        "mission:fault_analysis",
        "agent:fault_investigator",
        "session:sess-abc",
    ]


# --- MLflow execution run ---


class _FakeActiveRun:
    """Minimal stand-in for mlflow's ActiveRun context manager."""

    def __init__(self, calls):
        self._calls = calls

    def __enter__(self):
        self._calls.append(("enter",))
        return self

    def __exit__(self, *exc):
        self._calls.append(("exit",))
        return False


def _install_fake_mlflow(monkeypatch, calls, start_raises=False):
    fake = types.ModuleType("mlflow")

    def _start_run(*, run_name, nested):
        calls.append(("start_run", run_name, nested))
        if start_raises:
            raise RuntimeError("tracking server down")
        return _FakeActiveRun(calls)

    def _set_tag(key, value):
        calls.append(("set_tag", key, value))

    fake.start_run = _start_run
    fake.set_tag = _set_tag
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setattr(tracing_module, "is_mlflow_tracing_enabled", lambda: True)
    return fake


def test_mlflow_execution_run_disabled_yields_none(monkeypatch):
    monkeypatch.setattr(tracing_module, "is_mlflow_tracing_enabled", lambda: False)
    with mlflow_execution_run("exec-1", "fault_analysis", None, "sess-1") as run:
        assert run is None


def test_mlflow_execution_run_tags_and_closes(monkeypatch):
    calls: list = []
    _install_fake_mlflow(monkeypatch, calls)
    seen = []
    with mlflow_execution_run(
        "exec-1", "fault_analysis", "fault_investigator", "sess-abc"
    ) as run:
        assert run is not None
        seen.append("body")
    assert seen == ["body"]
    assert ("start_run", "execution-exec-1", True) in calls
    assert ("set_tag", "execution_id", "exec-1") in calls
    assert ("set_tag", "mission_id", "fault_analysis") in calls
    assert ("set_tag", "agent_id", "fault_investigator") in calls
    assert ("set_tag", "session_id", "sess-abc") in calls
    assert ("enter",) in calls
    assert ("exit",) in calls
    # The run is closed even though the body succeeded.
    assert calls.index(("exit",)) > calls.index(("enter",))


def test_mlflow_execution_run_start_failure_is_best_effort(monkeypatch):
    calls: list = []
    _install_fake_mlflow(monkeypatch, calls, start_raises=True)
    ran = []
    with mlflow_execution_run("exec-1", "fault_analysis") as run:
        assert run is None
        ran.append(True)
    assert ran == [True]
    assert ("start_run", "execution-exec-1", True) in calls


def test_mlflow_execution_run_does_not_swallow_body_errors(monkeypatch):
    calls: list = []
    _install_fake_mlflow(monkeypatch, calls)
    with pytest.raises(ValueError, match="boom"):
        with mlflow_execution_run("exec-1", "fault_analysis"):
            raise ValueError("boom")
    # The run is still closed on the way out.
    assert ("exit",) in calls


# --- LangSmith config on the graph invoke ---


@contextlib.contextmanager
def _no_checkpointer():
    yield None


def test_run_mission_passes_trace_config(monkeypatch):
    """graph.invoke receives metadata + tags carrying the session id."""
    monkeypatch.setattr(tracing_module, "is_mlflow_tracing_enabled", lambda: False)
    monkeypatch.setattr(
        mission_graph_module, "_resolve_agent_version", lambda agent_id: (None, False)
    )
    monkeypatch.setattr(mission_graph_module, "get_checkpointer", _no_checkpointer)
    captured: dict = {}

    class _FakeGraph:
        def invoke(self, state, config=None):
            captured["config"] = config
            return dict(state, status="SUCCEEDED")

    monkeypatch.setattr(mission_graph_module, "mission_graph", _FakeGraph())
    mission = {"mission_id": "fault_analysis", "agent_id": "fault_investigator"}
    final = run_mission(
        mission,
        {"asset_id": "O21"},
        "exec-trace-1",
        session_id="sess-trace-1",
    )
    assert final["status"] == "SUCCEEDED"
    config = captured["config"]
    assert config["metadata"]["execution_id"] == "exec-trace-1"
    assert config["metadata"]["mission_id"] == "fault_analysis"
    assert config["metadata"]["agent_id"] == "fault_investigator"
    assert config["metadata"]["session_id"] == "sess-trace-1"
    assert "session:sess-trace-1" in config["tags"]
    assert "mission:fault_analysis" in config["tags"]
    # No checkpointer in this test, so no thread_id is configured.
    assert "configurable" not in config


def test_run_mission_without_session_omits_session_tag(monkeypatch):
    monkeypatch.setattr(tracing_module, "is_mlflow_tracing_enabled", lambda: False)
    monkeypatch.setattr(
        mission_graph_module, "_resolve_agent_version", lambda agent_id: (None, False)
    )
    monkeypatch.setattr(mission_graph_module, "get_checkpointer", _no_checkpointer)
    captured: dict = {}

    class _FakeGraph:
        def invoke(self, state, config=None):
            captured["config"] = config
            return dict(state, status="SUCCEEDED")

    monkeypatch.setattr(mission_graph_module, "mission_graph", _FakeGraph())
    run_mission({"mission_id": "fault_analysis"}, {"asset_id": "O21"}, "exec-trace-2")
    config = captured["config"]
    assert "session_id" not in config["metadata"]
    assert not [t for t in config["tags"] if t.startswith("session:")]


# --- LangSmith config on the Deep Agent invoke ---


def test_deep_agent_node_passes_trace_metadata(db_session, monkeypatch):
    """deep_agent_node forwards the session identity to run_deep_agent."""
    monkeypatch.setattr(tracing_module, "is_mlflow_tracing_enabled", lambda: False)
    load_missions.main()
    mission = mission_service.get_mission(db_session, "fault_analysis")
    mission_dict = mission_service.mission_to_dict(mission)
    # Autonomous mode guarantees deep_agent_node runs.
    mission_dict["execution"]["mode"] = "autonomous"
    seen: dict = {}

    def fake_build(tools, system_prompt=None, skills=None, model_config=None,
                   middleware=None):
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None,
                 skill_maps=None, trace_metadata=None, trace_tags=None):
        seen["trace_metadata"] = trace_metadata
        seen["trace_tags"] = trace_tags
        return {
            "summary": "Done.",
            "evidence": [
                {"source": "deep_agent", "kind": "summary", "summary": "Done.",
                 "stub": False}
            ],
            "actions_taken": [],
            "errors": [],
        }

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)
    final = run_mission(
        mission_dict,
        {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"},
        "exec-trace-3",
        session_id="sess-trace-3",
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert seen["trace_metadata"]["session_id"] == "sess-trace-3"
    assert seen["trace_metadata"]["execution_id"] == "exec-trace-3"
    assert "session:sess-trace-3" in seen["trace_tags"]
