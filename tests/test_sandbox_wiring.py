"""Sandbox wiring tests — deep_agent_node passes the session sandbox
to build_deep_agent, and falls back cleanly without one.

No LLM, no server: the Deep Agent is faked (same pattern as
test_mission_graph._fake_deep_agent) and the sandbox resolver is
patched to return a fake backend, so these tests pin the wiring
(mission flag + session_id -> backend kwarg + execution note) only.
"""

from types import SimpleNamespace

from runtime import deep_agent as deep_agent_module
from runtime.mission_graph import run_mission

MISSION = {
    "mission_id": "fault_analysis",
    "name": "Fault Analysis",
    "version": 1,
    "objective": "Analyze equipment faults and identify root causes.",
    "input_schema": {
        "type": "object",
        "required": ["asset_id", "start_date", "end_date"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "root_causes": {"type": "array"},
            "evidence": {"type": "array"},
            "recommendations": {"type": "array"},
        },
    },
    "execution": {
        "mode": "autonomous",
        "agent_id": "fault_investigator",
        "sandbox": True,
    },
    "capabilities": ["fault_history", "maintenance_analysis", "knowledge_retrieval"],
    "agent_id": "fault_investigator",
}

INPUT = {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"}


def _install_fake_agent(monkeypatch, captured):
    """Stand-in Deep Agent returning a canned outcome (no LLM)."""
    outcome = {
        "summary": "The faults point at the door sensor.",
        "evidence": [
            {
                "source": "deep_agent", "kind": "fault",
                "summary": "F-101: Door sensor fault", "stub": False,
            }
        ],
        "actions_taken": [],
        "errors": [],
    }

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        return object()

    def fake_run(agent, mission_arg, input_data, **kwargs):
        return dict(outcome)

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)


def _install_fake_resolver(monkeypatch, calls, backend):
    import app.services.sandbox_service as sandbox_service

    def fake_resolve(db, *, session_id, agent_id, mission_id, **kwargs):
        calls.append(session_id)
        return backend

    monkeypatch.setattr(
        sandbox_service, "resolve_session_sandbox", fake_resolve
    )


def test_sandbox_backend_passed_to_agent(monkeypatch):
    captured = {}
    calls = []
    _install_fake_agent(monkeypatch, captured)
    _install_fake_resolver(monkeypatch, calls, SimpleNamespace(id="sbx-1"))
    final = run_mission(dict(MISSION), INPUT, "exec_sbx_1", session_id="sess-1")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert calls == ["sess-1"]
    assert captured["backend"] is not None
    assert captured["backend"].id == "sbx-1"
    assert any("Sandbox backend: sbx-1" in n for n in final.get("notes") or [])


def test_no_session_means_no_sandbox(monkeypatch):
    captured = {}
    calls = []
    _install_fake_agent(monkeypatch, captured)
    _install_fake_resolver(monkeypatch, calls, SimpleNamespace(id="sbx-1"))
    final = run_mission(dict(MISSION), INPUT, "exec_sbx_2")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert calls == []
    assert captured["backend"] is None


def test_sandbox_flag_off_means_no_sandbox(monkeypatch):
    captured = {}
    calls = []
    _install_fake_agent(monkeypatch, captured)
    _install_fake_resolver(monkeypatch, calls, SimpleNamespace(id="sbx-1"))
    mission = dict(MISSION)
    mission["execution"] = {"mode": "autonomous", "agent_id": "fault_investigator"}
    final = run_mission(mission, INPUT, "exec_sbx_3", session_id="sess-1")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert calls == []
    assert captured["backend"] is None


def test_sandbox_unavailable_falls_back_to_memory(monkeypatch):
    captured = {}
    calls = []
    _install_fake_agent(monkeypatch, captured)
    _install_fake_resolver(monkeypatch, calls, None)
    final = run_mission(dict(MISSION), INPUT, "exec_sbx_4", session_id="sess-1")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert calls == ["sess-1"]
    assert captured["backend"] is None
    assert any("Sandbox unavailable" in n for n in final.get("notes") or [])
