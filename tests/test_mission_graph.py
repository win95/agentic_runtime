"""Mission Graph tests — compile + deterministic run.

No LLM, no live DB, no API keys: the repository boundary is monkeypatched
to return rows in the shape of real FaultHistory / MaintenanceRecord /
Qdrant document / FalkorDB relation rows, so the graph runs end-to-end
with no infra. Repositories themselves never return fake data in
production.
"""

import pytest

import infra.repositories as repositories
from runtime.mission_graph import build_mission_graph, run_mission

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
    "execution": {"mode": "deterministic", "agent_id": "fault_investigator"},
    "capabilities": ["fault_history", "maintenance_analysis", "knowledge_retrieval"],
    "agent_id": "fault_investigator",
}

INPUT = {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"}


@pytest.fixture(autouse=True)
def _seeded_repositories(monkeypatch):
    """Repository-shaped demo rows (same shape as real store rows)."""
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


def test_graph_compiles():
    graph = build_mission_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")


def test_deterministic_run():
    final = run_mission(MISSION, INPUT, "exec_test_deterministic")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    result = final["result"]
    assert result is not None
    assert set(result.keys()) >= {"root_causes", "evidence", "recommendations"}
    assert len(result["root_causes"]) >= 1
    assert len(result["evidence"]) >= 1
    # Evidence comes from all four capability nodes.
    sources = {item.get("source") for item in final["evidence"]}
    assert sources >= {"fault_history", "maintenance", "knowledge_retrieval", "asset_relations"}
    assert final["errors"] == []


def test_hybrid_run_succeeds_without_llm():
    """Hybrid mode: capability evidence is rich enough (>=3 items) that the
    Deep Agent is not needed, so the run succeeds with no LLM key."""
    mission = dict(MISSION)
    mission["execution"] = {"mode": "hybrid", "agent_id": "fault_investigator"}
    final = run_mission(mission, INPUT, "exec_test_hybrid")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert len(final["evidence"]) >= 3
