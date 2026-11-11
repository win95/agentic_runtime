"""Admin bot tests — registry permission, admin tools, seed, autonomous run.

No LLM, no live infra: admin tool functions run against the in-memory
SQLite DB (tools open their own session on the same StaticPool DB); the
Deep Agent is monkeypatched for the autonomous-run test; system_health
reports the real (down) services without raising.
"""

import pathlib

import runtime.deep_agent as deep_agent_module
import runtime.tools.admin_tools as admin_tools  # noqa: F401 — importing registers them
from app import models
from app.config import settings
from app.services import execution_service, mission_service
from runtime.mission_graph import run_mission
from runtime.tools.registry import registry
from seeds import load_missions

ADMIN_TOOL_NAMES = {
    "list_executions",
    "get_execution",
    "list_missions",
    "get_mission",
    "list_agents",
    "system_health",
    "create_mission",
    "update_mission",
    "delete_mission",
    "dispatch_execution",
    "cancel_execution",
}

ADMIN_CONTEXT = {
    "mission_id": "system_admin",
    "tenant": "default",
    "execution_id": "exec_admin_test",
    "capabilities": ["system_admin"],
}

FAULT_CONTEXT = {
    "mission_id": "fault_analysis",
    "tenant": "default",
    "execution_id": "exec_test",
    "capabilities": ["fault_history", "maintenance_analysis", "knowledge_retrieval"],
}

BASE = pathlib.Path(__file__).resolve().parents[1] / "seeds"


def _invoke(tool_name: str, args: dict | None = None):
    """Invoke a registered admin tool by name (the realistic agent path)."""
    entry = registry.get(tool_name)
    assert entry is not None and entry.tool is not None
    return entry.tool.invoke(args or {})


def _make_mission(db_session, mission_id="m1"):
    mission_service.create_mission(db_session, {
        "id": mission_id,
        "name": mission_id,
        "version": 1,
        "objective": "test mission",
        "input_schema": {},
        "output_schema": {},
        "execution_config": {"mode": "hybrid"},
        "capabilities": [],
        "agent_id": None,
    })


# --- Registry ---
def test_admin_tools_registered_with_admin_permission():
    names = {e.name for e in registry.list()}
    assert ADMIN_TOOL_NAMES <= names
    for name in ADMIN_TOOL_NAMES:
        entry = registry.get(name)
        assert entry.permission == "admin"
        assert entry.capability == "system_admin"
        assert entry.tool is not None


def test_permission_check_admin_context_allows_admin_tools():
    for name in ADMIN_TOOL_NAMES:
        assert registry.check(name, ADMIN_CONTEXT) is True
    names = {t.name for t in registry.get_langchain_tools(ADMIN_CONTEXT)}
    assert names == ADMIN_TOOL_NAMES


def test_permission_check_non_admin_context_denies_admin_tools():
    for name in ADMIN_TOOL_NAMES:
        assert registry.check(name, FAULT_CONTEXT) is False
    names = {t.name for t in registry.get_langchain_tools(FAULT_CONTEXT)}
    assert not (names & ADMIN_TOOL_NAMES)


def test_permission_gate_active_when_permissions_listed():
    """check() semantics: the permission gate only applies when the context
    lists permissions at all; an empty list means 'no restriction'."""
    admin_with_perm = dict(ADMIN_CONTEXT, permissions=["admin"])
    assert registry.check("list_missions", admin_with_perm) is True
    admin_wrong_perm = dict(ADMIN_CONTEXT, permissions=["other"])
    assert registry.check("list_missions", admin_wrong_perm) is False
    # Fault tools need their own permission when one is listed.
    assert registry.check("search_fault_history", admin_with_perm) is False


# --- Read tools ---
def test_list_and_get_mission_tools(db_session):
    _make_mission(db_session, "m1")
    listed = _invoke("list_missions")
    assert listed["count"] == 1
    assert listed["missions"][0]["mission_id"] == "m1"
    got = _invoke("get_mission", {"mission_id": "m1"})
    assert got["name"] == "m1"
    assert _invoke("get_mission", {"mission_id": "nope"})["error"] == "Mission not found"


def test_list_executions_filters(db_session):
    _make_mission(db_session, "m1")
    _make_mission(db_session, "m2")
    for i, (mid, status) in enumerate([("m1", "SUCCEEDED"), ("m1", "FAILED"), ("m2", "QUEUED")]):
        db_session.add(models.Execution(
            id=f"exec_f{i}", mission_id=mid, mission_version=1,
            input={}, status=status,
        ))
    db_session.commit()
    assert _invoke("list_executions")["count"] == 3
    assert _invoke("list_executions", {"status": "FAILED"})["count"] == 1
    assert _invoke("list_executions", {"mission_id": "m2"})["count"] == 1
    assert _invoke("list_executions", {"limit": 2})["count"] == 2
    got = _invoke("get_execution", {"execution_id": "exec_f0"})
    assert got["status"] == "SUCCEEDED"
    assert got["result"] is None
    assert _invoke("get_execution", {"execution_id": "nope"})["error"] == "Execution not found"


def test_list_agents_tool(db_session):
    db_session.add(models.Agent(id="a1", name="A1", provider="openai", model="gpt-4o-mini"))
    db_session.commit()
    listed = _invoke("list_agents")
    assert listed["count"] == 1
    assert listed["agents"][0]["agent_id"] == "a1"


def test_system_health_never_raises():
    health = _invoke("system_health")
    assert set(health["checks"]) == {"database", "redis", "qdrant", "mlflow", "minio"}
    for check in health["checks"].values():
        assert check["status"] in ("ok", "fail", "skipped")
        assert check["detail"]
    assert health["overall"] in ("ok", "degraded")
    # In the test env the SQLite DB answers but nothing else is up.
    assert health["checks"]["database"]["status"] == "ok"
    assert health["checks"]["redis"]["status"] == "skipped"  # memory:// broker


# --- Write tools ---
def test_create_update_delete_mission_tools(db_session):
    created = _invoke("create_mission", {
        "mission_id": "m1", "name": "M One", "objective": "test",
        "capabilities": ["system_admin"],
    })
    assert created["created"] is True
    assert created["mission_id"] == "m1"
    duplicate = _invoke("create_mission", {"mission_id": "m1", "name": "M One"})
    assert "already exists" in duplicate["error"]

    updated = _invoke("update_mission", {"mission_id": "m1", "name": "M Uno"})
    assert updated["updated"] is True
    assert updated["name"] == "M Uno"
    assert updated["objective"] == "test"  # unchanged fields are kept
    # Upsert: updating a missing mission creates it.
    assert _invoke("update_mission", {"mission_id": "m2", "name": "M Two"})["updated"] is True

    deleted = _invoke("delete_mission", {"mission_id": "m2"})
    assert deleted == {"deleted": True, "mission_id": "m2"}
    assert _invoke("delete_mission", {"mission_id": "m2"})["error"] == "Mission not found"


def test_delete_mission_with_executions_refused(db_session):
    _make_mission(db_session, "m1")
    db_session.add(models.Execution(
        id="exec_x", mission_id="m1", mission_version=1, input={}, status="SUCCEEDED",
    ))
    db_session.commit()
    result = _invoke("delete_mission", {"mission_id": "m1"})
    assert result["deleted"] is False
    assert result["refused"] is True
    assert result["execution_count"] == 1
    # Nothing was deleted: mission and execution are intact.
    assert mission_service.get_mission(db_session, "m1") is not None
    assert execution_service.get_execution(db_session, "exec_x") is not None


def test_dispatch_execution_tool(db_session, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        execution_service, "dispatch_execution", lambda execution_id: calls.append(execution_id)
    )
    _make_mission(db_session, "m1")
    result = _invoke("dispatch_execution", {"mission_id": "m1", "input_data": {}})
    assert result["status"] == "QUEUED"
    assert result["execution_id"].startswith("exec_")
    assert calls == [result["execution_id"]]
    assert execution_service.get_execution(db_session, result["execution_id"]) is not None

    assert _invoke("dispatch_execution", {"mission_id": "nope"})["error"] == "Mission not found"


def test_dispatch_execution_missing_input(db_session, monkeypatch):
    monkeypatch.setattr(execution_service, "dispatch_execution", lambda execution_id: None)
    mission_service.create_mission(db_session, {
        "id": "m1", "name": "m1", "version": 1, "objective": "",
        "input_schema": {"type": "object", "required": ["asset_id"]},
        "output_schema": {},
        "execution_config": {"mode": "hybrid"},
        "capabilities": [], "agent_id": None,
    })
    result = _invoke("dispatch_execution", {"mission_id": "m1", "input_data": {}})
    assert "asset_id" in result["error"]
    assert result["missing_fields"] == ["asset_id"]


def test_cancel_execution_tool(db_session, monkeypatch):
    monkeypatch.setattr(execution_service, "dispatch_execution", lambda execution_id: None)
    _make_mission(db_session, "m1")
    dispatched = _invoke("dispatch_execution", {"mission_id": "m1", "input_data": {}})
    execution_id = dispatched["execution_id"]
    cancelled = _invoke("cancel_execution", {"execution_id": execution_id})
    assert cancelled == {"execution_id": execution_id, "status": "CANCELLED", "cancelled": True}
    # Cancelling again keeps the status; unknown ids error.
    assert _invoke("cancel_execution", {"execution_id": execution_id})["cancelled"] is True
    assert _invoke("cancel_execution", {"execution_id": "nope"})["error"] == "Execution not found"
    # A finished execution is left alone.
    db_session.add(models.Execution(
        id="exec_done", mission_id="m1", mission_version=1, input={}, status="SUCCEEDED",
    ))
    db_session.commit()
    done = _invoke("cancel_execution", {"execution_id": "exec_done"})
    assert done == {"execution_id": "exec_done", "status": "SUCCEEDED", "cancelled": False}


# --- Seed ---
def test_seed_loads_system_admin_mission_and_agent(db_session):
    load_missions.main()
    mission = mission_service.get_mission(db_session, "system_admin")
    assert mission is not None
    assert mission.agent_id == "admin_bot"
    assert mission.capabilities == ["system_admin"]
    assert (mission.execution_config or {}).get("mode") == "autonomous"
    assert (mission.input_schema or {}).get("required") == ["request"]
    assert set((mission.output_schema or {}).get("properties", {})) == {
        "answer", "actions_taken", "evidence",
    }
    agent = db_session.get(models.Agent, "admin_bot")
    assert agent is not None
    assert set(agent.allowed_tools or []) == ADMIN_TOOL_NAMES
    # The fault_investigator agent is seeded from its own YAML as well.
    investigator = db_session.get(models.Agent, "fault_investigator")
    assert investigator is not None
    assert investigator.provider == "openai"
    # The seed YAML writes model: ${LLM_MODEL}, so the seeded model
    # follows settings.llm_model (item 9, 2026-11-11).
    assert investigator.model == settings.llm_model
    assert set(investigator.allowed_tools or []) == {
        "calculate_failure_rate",
        "get_maintenance_records",
        "get_sensor_data",
        "query_asset_relationships",
        "search_fault_history",
        "search_technical_documents",
    }
    # The existing seed is untouched.
    assert mission_service.get_mission(db_session, "fault_analysis") is not None


def test_seed_files_parse():
    mission = load_missions.load_mission_file(BASE / "missions" / "system_admin.yaml")
    assert mission["id"] == "system_admin"
    assert mission["agent_id"] == "admin_bot"
    agent = load_missions.load_agent_file(BASE / "agents" / "admin_bot.yaml")
    assert agent["id"] == "admin_bot"
    assert set(agent["allowed_tools"]) == ADMIN_TOOL_NAMES
    investigator = load_missions.load_agent_file(
        BASE / "agents" / "fault_investigator.yaml"
    )
    assert investigator["id"] == "fault_investigator"
    assert investigator["model"] == settings.llm_model
    assert investigator["temperature"] == 0.0
    assert set(investigator["allowed_tools"]) == {
        "calculate_failure_rate",
        "get_maintenance_records",
        "get_sensor_data",
        "query_asset_relationships",
        "search_fault_history",
        "search_technical_documents",
    }


# --- Autonomous run (mocked Deep Agent — no LLM) ---
def test_autonomous_system_admin_run_succeeds(db_session, monkeypatch):
    load_missions.main()
    mission = mission_service.get_mission(db_session, "system_admin")
    mission_dict = mission_service.mission_to_dict(mission)

    seen_prompts: list[str | None] = []

    def fake_build_deep_agent(tools, system_prompt=None, skills=None, model_config=None,
                              middleware=None, backend=None):
        seen_prompts.append(system_prompt)
        return object()

    def fake_run_deep_agent(agent, mission_arg, input_data, callbacks=None, files=None):
        return {
            "summary": "Có 1 execution lỗi hôm nay.",
            "evidence": [{
                "source": "deep_agent", "kind": "summary",
                "summary": "Có 1 execution lỗi hôm nay.", "stub": False,
            }],
            "actions_taken": ["list_executions", "get_execution"],
            "errors": [],
        }

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build_deep_agent)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run_deep_agent)

    final = run_mission(mission_dict, {"request": "execution nào lỗi hôm nay?"}, "exec_admin_run")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert final["result"]["answer"] == "Có 1 execution lỗi hôm nay."
    assert final["result"]["actions_taken"] == ["list_executions", "get_execution"]
    assert len(final["result"]["evidence"]) >= 1
    # The admin system prompt (not the fault_investigator one) was used.
    assert seen_prompts and seen_prompts[0] is not None
    assert "admin bot" in seen_prompts[0].lower()
