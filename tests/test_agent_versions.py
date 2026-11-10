"""Agent versioning (Option C) tests — no live LLM, DB, or MLflow needed.

The Deep Agent node is exercised with the same doubles the middleware /
skills tests use (patched build_deep_agent / run_deep_agent), so the
version path is tested without a model.
"""

import pytest

from app import models
from app.core.db import SessionLocal
from app.services import agent_version_service as avs
from app.services import mission_service
from runtime import deep_agent as deep_agent_module
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
        assert same.model == "gpt-4o-mini" and same.temperature == 0.0
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
    def fake_build(tools, system_prompt=None, skills=None, model_config=None):
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
    assert seen["model_config"].llm_model == "gpt-4o-mini"
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
