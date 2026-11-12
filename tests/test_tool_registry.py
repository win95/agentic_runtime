"""Tool Registry tests — registration, schemas, permission checks."""

from pathlib import Path

import yaml

import runtime.tools  # noqa: F401 — importing registers the built-in tools
from runtime.tools.registry import registry

EXPECTED_TOOLS = {
    "search_fault_history",
    "get_maintenance_records",
    "search_technical_documents",
    "query_asset_relationships",
    "get_sensor_data",
    "calculate_failure_rate",
    # Admin tools (runtime/tools/admin_tools.py).
    "get_executions",
    "get_missions",
    "list_agents",
    "system_health",
    "save_mission",
    "delete_mission",
    "dispatch_execution",
    "cancel_execution",
}

FAULT_CONTEXT = {
    "mission_id": "fault_analysis",
    "tenant": "default",
    "execution_id": "exec_test",
    "capabilities": ["fault_history", "maintenance_analysis", "knowledge_retrieval"],
}


def test_all_tools_registered():
    names = {entry.name for entry in registry.list()}
    assert names == EXPECTED_TOOLS


def test_tool_schemas_valid():
    for entry in registry.list():
        assert entry.name
        assert entry.description
        assert entry.capability
        assert entry.tool is not None
        # LangChain infers the args schema from the typed signature; it is
        # legitimately empty for parameterless tools (system_health).
        assert isinstance(entry.input_schema, dict)


def test_permission_check_allows_listed_capability():
    assert registry.check("search_fault_history", FAULT_CONTEXT) is True
    assert registry.check("get_maintenance_records", FAULT_CONTEXT) is True
    assert registry.check("search_technical_documents", FAULT_CONTEXT) is True
    assert registry.check("calculate_failure_rate", FAULT_CONTEXT) is True


def test_permission_check_denies_unlisted_capability():
    # sensor_data and asset_relations are NOT in the mission's capabilities.
    assert registry.check("get_sensor_data", FAULT_CONTEXT) is False
    assert registry.check("query_asset_relationships", FAULT_CONTEXT) is False
    assert registry.check("nonexistent_tool", FAULT_CONTEXT) is False


def test_seeded_fault_mission_declares_all_tool_capabilities():
    """F1 fix (2026-11-11): the seeded fault_analysis mission must
    declare every capability its agent's tools need, or the gate
    silently drops tools (get_sensor_data, query_asset_relationships)
    from the agent. Reads the real YAML so the test tracks the config
    instead of a hardcoded copy."""
    mission_yaml = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "seeds" / "missions"
         / "fault_analysis.yaml").read_text(encoding="utf-8")
    )
    context = dict(FAULT_CONTEXT)
    context["capabilities"] = mission_yaml["capabilities"]
    names = {tool.name for tool in registry.get_langchain_tools(context)}
    assert names == {
        "search_fault_history",
        "get_maintenance_records",
        "search_technical_documents",
        "query_asset_relationships",
        "get_sensor_data",
        "calculate_failure_rate",
    }


def test_permission_check_denies_unlisted_tool():
    context = dict(FAULT_CONTEXT)
    context["allowed_tools"] = ["search_fault_history"]
    assert registry.check("search_fault_history", context) is True
    assert registry.check("get_maintenance_records", context) is False


def test_list_for_context_returns_only_allowed():
    tools = registry.get_langchain_tools(FAULT_CONTEXT)
    names = {tool.name for tool in tools}
    assert names == {
        "search_fault_history",
        "get_maintenance_records",
        "search_technical_documents",
        "calculate_failure_rate",
    }
