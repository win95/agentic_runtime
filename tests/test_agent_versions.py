"""Agent versioning (Option C) tests — no live LLM, DB, or MLflow needed.

The Deep Agent node is exercised with the same doubles the middleware /
skills tests use (patched build_deep_agent / run_deep_agent), so the
version path is tested without a model.
"""

import pytest

import infra.repositories as repositories
import runtime.mission_graph as mission_graph_module
from app import models
from app.config import settings
from app.core.db import SessionLocal
from app.services import agent_version_service as avs
from app.services import mission_service
from runtime import deep_agent as deep_agent_module
from runtime.deep_agent import ADMIN_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
from runtime.mission_graph import deep_agent_node, run_mission
from seeds import load_missions

ADMIN_TOOLS = {
    "list_executions", "get_execution", "list_missions", "get_mission",
    "list_agents", "system_health", "create_mission", "update_mission",
    "delete_mission", "dispatch_execution", "cancel_execution",
}


def _seed_agents(db_session):
    load_missions.main()
    return db_session


def _patch_capability_repositories(monkeypatch):
    """Repository-shaped rows (same as tests/test_mission_graph.py) —
    enough substantive evidence that a hybrid run never escalates."""
    monkeypatch.setattr(
        repositories,
        "search_fault_history",
        lambda *a, **k: (
            [
                {"id": "fh-1", "code": "F-101", "title": "Door sensor fault"},
                {"id": "fh-2", "code": "F-204", "title": "Motor overheating"},
            ],
            False,
        ),
    )
    monkeypatch.setattr(
        repositories,
        "get_maintenance_records",
        lambda *a, **k: ([{"id": "mr-1", "action": "Sensor cleaned"}], False),
    )
    monkeypatch.setattr(
        repositories,
        "search_documents",
        lambda *a, **k: ([{"doc_id": "d1", "title": "Door sensor doc"}], False),
    )
    monkeypatch.setattr(
        repositories,
        "query_asset_relationships",
        lambda *a, **k: ([{"from": "O21", "type": "HAS_COMPONENT", "to": "motor-M2"}], False),
    )


def test_manifest_covers_all_built_in_tools():
    manifest, digest = avs.compute_tool_manifest()
    assert len(manifest) == 17
    assert {e["name"] for e in manifest} == ADMIN_TOOLS | {
        "calculate_failure_rate", "get_maintenance_records", "get_sensor_data",
        "query_asset_relationships", "search_fault_history",
        "search_technical_documents",
    }
    entry = next(e for e in manifest if e["name"] == "get_sensor_data")
    assert entry["capability"] == "sensor_data"
    assert entry["read_only"] is True
    assert entry["implementation"].endswith("get_sensor_data")
    _, digest2 = avs.compute_tool_manifest()
    assert digest == digest2


def test_create_version_idempotent_then_new_on_change(db_session):
    _seed_agents(db_session)  # seed already created v1 for both agents
    db = SessionLocal()
    try:
        same, created = avs.create_version(db, "admin_bot", source="test")
        assert created is False and same.version == 1 and same.is_active
        # The seed YAML writes model: ${LLM_MODEL}, so the snapshot
        # follows settings.llm_model (item 9, 2026-11-11).
        assert same.model == settings.llm_model and same.temperature == 0.0
        assert set(same.allowed_tools) == ADMIN_TOOLS
        assert len(same.tool_manifest) == 17
        assert len(same.manifest_sha256) == 64
        assert len(avs.list_versions(db, "admin_bot")) == 1

        agent = db.get(models.Agent, "admin_bot")
        agent.temperature = 0.5
        db.commit()
        v2, created3 = avs.create_version(db, "admin_bot", source="test")
        assert created3 is True and v2.version == 2 and v2.is_active
        assert v2.temperature == 0.5
        assert db.get(models.AgentVersion, same.id).is_active is False
        assert avs.get_active_version(db, "admin_bot").version == 2

        # Rollback: activate v1 again.
        back = avs.activate_version(db, "admin_bot", 1)
        assert back.version == 1 and back.is_active
        assert avs.get_active_version(db, "admin_bot").version == 1
        assert avs.get_version(db, "admin_bot", 99) is None
        with pytest.raises(avs.AgentVersionNotFoundError):
            avs.activate_version(db, "admin_bot", 99)
        with pytest.raises(avs.AgentNotFoundError):
            avs.create_version(db, "no_such_agent", source="test")
    finally:
        db.close()


def test_seed_loader_creates_versions_once(db_session):
    load_missions.main()
    db = SessionLocal()
    try:
        for agent_id in ("admin_bot", "fault_investigator"):
            versions = avs.list_versions(db, agent_id)
            assert len(versions) == 1 and versions[0].version == 1
            assert versions[0].is_active
        load_missions.main()  # re-seed unchanged: no new versions
        for agent_id in ("admin_bot", "fault_investigator"):
            assert len(avs.list_versions(db, agent_id)) == 1
        # A config change in the agent row creates v2 via ensure
        # (the seed loader re-upserts from YAML, so a row-level change
        # is the way to simulate an edited YAML here).
        agent = db.get(models.Agent, "fault_investigator")
        agent.temperature = 0.3
        db.commit()
        version, created = avs.ensure_version_for_agent(
            db, "fault_investigator", source="seed"
        )
        assert created is True and version.version == 2
        versions = avs.list_versions(db, "fault_investigator")
        assert [v.version for v in versions] == [2, 1]
        assert avs.get_active_version(db, "fault_investigator").temperature == 0.3
    finally:
        db.close()


def _fake_agent(monkeypatch, seen):
    def fake_build(tools, system_prompt=None, skills=None, model_config=None,
                   middleware=None):
        seen["tools"] = [t.name for t in tools]
        seen["system_prompt"] = system_prompt
        seen["model_config"] = model_config
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None):
        return {
            "summary": "Done.",
            "evidence": [{
                "source": "deep_agent", "kind": "summary",
                "summary": "Done.", "stub": False,
            }],
            "actions_taken": [],
            "errors": [],
        }

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)


def test_node_uses_active_version(db_session, monkeypatch):
    _seed_agents(db_session)
    mission = mission_service.get_mission(db_session, "system_admin")
    mission_dict = mission_service.mission_to_dict(mission)
    seen: dict = {}
    _fake_agent(monkeypatch, seen)

    state = {
        "execution_id": "exec-av-1", "mission_id": "system_admin",
        "mission": mission_dict, "input": {"request": "kiểm tra hệ thống"},
        "mode": "autonomous", "plan": [], "evidence": [], "result": None,
        "status": "RUNNING", "errors": [], "notes": [],
    }
    out = deep_agent_node(state)
    assert out.get("agent_version") == 1
    assert out.get("agent_version_drift") is False
    assert set(seen["tools"]) == ADMIN_TOOLS
    assert seen["model_config"] is not None
    assert seen["model_config"].llm_model == settings.llm_model
    assert seen["model_config"].llm_temperature == 0.0
    # The pinned prompt text (from the agent row), not the registry lookup.
    assert seen["system_prompt"] is not None
    assert "admin bot" in seen["system_prompt"]


def test_node_records_drift_when_tools_change(db_session, monkeypatch):
    _seed_agents(db_session)
    from runtime.tools.registry import registry

    entry = registry.get("list_executions")
    assert entry is not None
    original = entry.description
    entry.description = "CHANGED description"
    try:
        mission = mission_service.get_mission(db_session, "system_admin")
        mission_dict = mission_service.mission_to_dict(mission)
        seen: dict = {}
        _fake_agent(monkeypatch, seen)
        state = {
            "execution_id": "exec-av-2", "mission_id": "system_admin",
            "mission": mission_dict, "input": {"request": "x"},
            "mode": "autonomous", "plan": [], "evidence": [], "result": None,
            "status": "RUNNING", "errors": [], "notes": [],
        }
        out = deep_agent_node(state)
        assert out.get("agent_version") == 1
        assert out.get("agent_version_drift") is True
    finally:
        entry.description = original


def test_node_enforces_version_allowed_tools(db_session, monkeypatch):
    _seed_agents(db_session)
    db = SessionLocal()
    try:
        agent = db.get(models.Agent, "fault_investigator")
        agent.allowed_tools = ["search_fault_history"]
        db.commit()
        avs.create_version(db, "fault_investigator", source="test")
    finally:
        db.close()
    mission = mission_service.get_mission(db_session, "fault_analysis")
    mission_dict = mission_service.mission_to_dict(mission)
    seen: dict = {}
    _fake_agent(monkeypatch, seen)
    state = {
        "execution_id": "exec-av-3", "mission_id": "fault_analysis",
        "mission": mission_dict,
        "input": {"asset_id": "O21", "start_date": "2026-10-01",
                  "end_date": "2026-10-31"},
        "mode": "autonomous", "plan": [], "evidence": [], "result": None,
        "status": "RUNNING", "errors": [], "notes": [],
    }
    out = deep_agent_node(state)
    assert out.get("agent_version") == 2
    assert seen["tools"] == ["search_fault_history"]


FAULT_TOOLS = {
    "calculate_failure_rate",
    "get_maintenance_records",
    "get_sensor_data",
    "query_asset_relationships",
    "search_fault_history",
    "search_technical_documents",
}


def test_fault_agent_receives_all_six_tools(db_session, monkeypatch):
    """F1 fix (2026-11-11): the fault agent lists six tools in
    allowed_tools, and the seeded mission now declares all five
    capabilities (incl. sensor_data + asset_relations), so the
    capability gate must let all six through to the Deep Agent.
    Before the fix only four arrived."""
    _seed_agents(db_session)
    mission = mission_service.get_mission(db_session, "fault_analysis")
    mission_dict = mission_service.mission_to_dict(mission)
    assert set(mission_dict["capabilities"]) >= {
        "fault_history", "maintenance_analysis", "knowledge_retrieval",
        "sensor_data", "asset_relations",
    }
    seen: dict = {}
    _fake_agent(monkeypatch, seen)
    state = {
        "execution_id": "exec-av-f1", "mission_id": "fault_analysis",
        "mission": mission_dict,
        "input": {"asset_id": "O21", "start_date": "2026-10-01",
                  "end_date": "2026-10-31"},
        "mode": "autonomous", "plan": [], "evidence": [], "result": None,
        "status": "RUNNING", "errors": [], "notes": [],
    }
    out = deep_agent_node(state)
    assert out.get("agent_version") == 1
    assert set(seen["tools"]) == FAULT_TOOLS


def test_run_mission_carries_agent_version(db_session, monkeypatch):
    _seed_agents(db_session)
    mission = mission_service.get_mission(db_session, "system_admin")
    mission_dict = mission_service.mission_to_dict(mission)
    seen: dict = {}
    _fake_agent(monkeypatch, seen)
    final = run_mission(
        mission_dict, {"request": "execution nào lỗi hôm nay?"}, "exec-av-run"
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert final.get("agent_version") == 1
    assert final.get("agent_version_drift") is False


def test_capability_only_run_records_agent_version(db_session, monkeypatch):
    """Option C semantics, revised (item 4, 2026-11-11): the version is
    resolved once in run_mission before branching, so a hybrid run
    that never escalates RECORDS the active version — NULL now means
    "no version active / no agent configured", not "no Deep Agent
    ran". The guard below fails loudly if the run ever reaches the
    Deep Agent."""
    _seed_agents(db_session)
    _patch_capability_repositories(monkeypatch)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("Deep Agent must not run in a capability-only test")

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", _must_not_run)
    mission = mission_service.get_mission(db_session, "fault_analysis")
    mission_dict = mission_service.mission_to_dict(mission)
    final = run_mission(
        mission_dict,
        {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"},
        "exec-av-capability-only",
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert len(final["evidence"]) >= 3  # the capability path really ran
    assert final.get("agent_version") == 1
    assert not final.get("agent_version_drift")


def test_versions_router(db_session, client):
    _seed_agents(db_session)
    db = SessionLocal()
    try:
        agent = db.get(models.Agent, "admin_bot")
        agent.temperature = 0.7
        db.commit()
        avs.create_version(db, "admin_bot", source="test")
    finally:
        db.close()

    resp = client.get("/agents/admin_bot/versions")
    assert resp.status_code == 200
    versions = resp.json()
    assert [v["version"] for v in versions] == [2, 1]
    assert versions[0]["is_active"] is True
    assert versions[0]["temperature"] == 0.7
    assert len(versions[0]["tool_manifest"]) == 17

    resp = client.get("/agents/admin_bot/versions/1")
    assert resp.status_code == 200
    assert resp.json()["version"] == 1

    resp = client.post("/agents/admin_bot/versions/1/activate")
    assert resp.status_code == 200
    assert resp.json()["is_active"] is True
    assert client.get("/agents/admin_bot/versions/2").json()["is_active"] is False

    assert client.get("/agents/nope/versions").status_code == 404
    assert client.get("/agents/admin_bot/versions/99").status_code == 404
    assert client.post("/agents/admin_bot/versions/99/activate").status_code == 404


# --- No-version prompt resolution (Part 2 fix) ---
#
# The runtime never looks up MLflow for prompts: with no active version
# the node runs the code constant (a copy of the YAML seed text), even
# when the registry is reachable and holds a different prompt. These
# tests fail on the old code (the registry text would win) and pass on
# the new code.


def _no_version_mission(mission_id, agent_id, capabilities):
    return {
        "mission_id": mission_id,
        "agent_id": agent_id,
        "objective": "Test mission.",
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object", "properties": {}},
        "execution": {"mode": "autonomous"},
        "capabilities": capabilities,
    }


def _fake_reachable_registry_with_other_text(monkeypatch):
    """Pretend the MLflow registry is up and holds a DIFFERENT prompt."""
    import mlflow.genai as genai

    class _Version:
        template = "REGISTRY TEXT — must never run"
        version = 99

    monkeypatch.setattr("infra.mlflow_setup.is_mlflow_tracing_enabled", lambda: True)
    monkeypatch.setattr(genai, "load_prompt", lambda *a, **k: _Version())


def _no_version_state(mission):
    return {
        "execution_id": "exec-nv-1",
        "mission_id": mission["mission_id"],
        "mission": mission,
        "input": {"asset_id": "O21"},
        "mode": "autonomous",
        "plan": [],
        "evidence": [],
        "result": None,
        "status": "RUNNING",
        "errors": [],
        "notes": [],
    }


def test_no_version_fault_branch_uses_code_constant(monkeypatch):
    _fake_reachable_registry_with_other_text(monkeypatch)
    monkeypatch.setattr(
        mission_graph_module, "_resolve_agent_version", lambda agent_id: (None, False)
    )
    seen: dict = {}
    _fake_agent(monkeypatch, seen)
    out = deep_agent_node(
        _no_version_state(
            _no_version_mission(
                "fault_analysis", "fault_investigator", ["fault_history"]
            )
        )
    )
    assert out.get("agent_version") is None
    assert seen["system_prompt"] == DEFAULT_SYSTEM_PROMPT


def test_no_version_admin_branch_uses_code_constant(monkeypatch):
    _fake_reachable_registry_with_other_text(monkeypatch)
    monkeypatch.setattr(
        mission_graph_module, "_resolve_agent_version", lambda agent_id: (None, False)
    )
    seen: dict = {}
    _fake_agent(monkeypatch, seen)
    out = deep_agent_node(
        _no_version_state(
            _no_version_mission("system_admin", "admin_bot", ["system_admin"])
        )
    )
    assert out.get("agent_version") is None
    assert seen["system_prompt"] == ADMIN_SYSTEM_PROMPT


def test_build_deep_agent_defaults_to_code_constant(monkeypatch):
    """build_deep_agent() with no prompt uses DEFAULT_SYSTEM_PROMPT."""
    import deepagents

    captured: dict = {}

    def fake_create_deep_agent(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create_deep_agent)
    # No live LLM in tests: the model itself is irrelevant here.
    monkeypatch.setattr(deep_agent_module, "get_chat_model", lambda: object())
    deep_agent_module.build_deep_agent([])
    assert captured["system_prompt"] == DEFAULT_SYSTEM_PROMPT
