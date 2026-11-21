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


# The analytics agent uses the same per-session sandbox backend for its
# report-file work (runtime/mission_graph.py reads the mission flag
# generically — no mission is hardcoded in the sandbox path).
ANALYTIS_MISSION = {
    "mission_id": "agent_analytis_mission",
    "name": "Agent Analytis Mission",
    "version": 1,
    "objective": "Analyze data and research the web, then export an HTML report.",
    "input_schema": {"type": "object", "required": ["message"]},
    "output_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "actions_taken": {"type": "array"},
            "files": {"type": "array"},
        },
    },
    "execution": {
        "mode": "autonomous",
        "agent_id": "agent_analytis",
        "sandbox": True,
        # These tests pin the free-form path: they verify the sandbox /
        # export wiring (mission flag + session_id -> backend kwarg),
        # which is identical code before the flow/free-form branch in
        # deep_agent_node. The flow path itself is covered by the
        # router tests (tests/test_flow_router_structured.py) and the
        # live --self-test; the fake agent here cannot drive a router.
        "harness": {"flow_subgraph": {"enabled": False}},
    },
    "capabilities": ["agent_analytis"],
    "agent_id": "agent_analytis",
}


def _fake_sandbox_backend(files):
    """OpenSandboxBackend stand-in: ls lists output/, download_files
    returns the canned bytes (SimpleNamespace mirrors
    FileDownloadResponse's path/content/error fields).

    Implements the export marker contract (runtime/sandbox/export.py):
    ``execute`` answers ``touch`` (run marker), ``find … -newer`` (all
    canned files are newer than the marker), ``stat`` (byte size) and
    ``readlink -f``/``realpath`` (echo the path back absolute).
    """
    entries = [{"path": f"output/{name}", "is_dir": False} for name in files]

    def ls(directory):
        assert directory == "output"
        return SimpleNamespace(entries=list(entries), error=None)

    def _key(path):
        return path.lstrip("/").removeprefix("output/")

    def download_files(paths):
        return [
            SimpleNamespace(path=p, content=files[_key(p)], error=None) for p in paths
        ]

    def execute(command):
        import shlex

        if command.startswith("touch "):
            return SimpleNamespace(output="", exit_code=0)
        if " -newer " in command:
            out = "\n".join(f"/output/{name}" for name in files)
            return SimpleNamespace(output=out, exit_code=0)
        if command.startswith("stat -L -c %s"):
            # shlex.quote only quotes when needed, so split instead.
            path = shlex.split(command)[4]
            return SimpleNamespace(output=str(len(files[_key(path)])), exit_code=0)
        if "readlink -f" in command or "realpath" in command:
            tokens = shlex.split(command)
            path = tokens[tokens.index("-f") + 1]
            path = path if path.startswith("/") else f"/{path}"
            return SimpleNamespace(output=path, exit_code=0)
        return SimpleNamespace(output="", exit_code=0)

    return SimpleNamespace(
        id="sbx-9", ls=ls, download_files=download_files, execute=execute
    )


def test_analytis_mission_uses_session_sandbox(monkeypatch):
    captured = {}
    calls = []
    _install_fake_agent(monkeypatch, captured)
    _install_fake_resolver(monkeypatch, calls, SimpleNamespace(id="sbx-9"))
    final = run_mission(
        dict(ANALYTIS_MISSION),
        {"message": "phân tích dữ liệu bán hàng"},
        "exec_sbx_9",
        session_id="sess-9",
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert calls == ["sess-9"]
    assert captured["backend"] is not None
    assert captured["backend"].id == "sbx-9"
    assert any("Sandbox backend: sbx-9" in n for n in final.get("notes") or [])


def test_analytis_mission_exports_report_files(monkeypatch):
    captured = {}
    calls = []
    _install_fake_agent(monkeypatch, captured)
    _install_fake_resolver(
        monkeypatch,
        calls,
        _fake_sandbox_backend(
            {"report.html": b"<html>bao cao</html>", "data.csv": b"a,b\n1,2\n"}
        ),
    )
    final = run_mission(
        dict(ANALYTIS_MISSION),
        {"message": "phân tích dữ liệu bán hàng"},
        "exec_sbx_10",
        session_id="sess-10",
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    exported = final["result"]["files"]
    # Export sorts paths (runtime/sandbox/export.py), so data.csv first.
    assert [(f["name"], f["content"]) for f in exported] == [
        ("data.csv", "a,b\n1,2\n"),
        ("report.html", "<html>bao cao</html>"),
    ]


def test_analytis_mission_sandbox_fallback_exports_nothing(monkeypatch):
    captured = {}
    calls = []
    _install_fake_agent(monkeypatch, captured)
    _install_fake_resolver(monkeypatch, calls, None)
    final = run_mission(
        dict(ANALYTIS_MISSION),
        {"message": "phân tích dữ liệu bán hàng"},
        "exec_sbx_11",
        session_id="sess-11",
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert final["result"]["files"] == []
    assert any("Sandbox unavailable" in n for n in final.get("notes") or [])
