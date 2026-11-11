"""Mission Graph tests — compile + deterministic run.

No LLM, no live DB, no API keys: the repository boundary is monkeypatched
to return rows in the shape of real FaultHistory / MaintenanceRecord /
Qdrant document / FalkorDB relation rows, so the graph runs end-to-end
with no infra. Repositories themselves never return fake data in
production.
"""

import pytest

import infra.repositories as repositories
from runtime import deep_agent as deep_agent_module
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


# --- Item 3 (2026-11-11, Option A): agent outcome maps to schema keys ---
def _fake_deep_agent(monkeypatch, *, summary, evidence=None):
    """Stand-in Deep Agent returning a canned outcome (no LLM)."""
    outcome = {
        "summary": summary,
        "evidence": evidence
        if evidence is not None
        else [
            {
                "source": "deep_agent", "kind": "summary",
                "summary": summary, "stub": False,
            }
        ],
        "actions_taken": [],
        "errors": [],
    }

    def fake_build(*args, **kwargs):
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None):
        return dict(outcome)

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)


def _autonomous_mission():
    mission = dict(MISSION)
    mission["execution"] = {"mode": "autonomous", "agent_id": "fault_investigator"}
    return mission


def test_autonomous_with_evidence_maps_root_causes(monkeypatch):
    """Option A: substantive evidence items become root causes (one
    line each, citing the item's own summary) — nothing invented."""
    _fake_deep_agent(
        monkeypatch,
        summary="The faults point at the door sensor and the motor.",
        evidence=[
            {
                "source": "deep_agent", "kind": "fault",
                "summary": "F-101: Door sensor fault", "stub": False,
            },
            {
                "source": "deep_agent", "kind": "document",
                "summary": "Document: Door sensor manual", "stub": False,
            },
        ],
    )
    final = run_mission(_autonomous_mission(), INPUT, "exec_test_auto_evidence")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert final["result"]["root_causes"] == [
        "F-101: Door sensor fault",
        "Document: Door sensor manual",
    ]
    # No actionable sentence in the summary -> honest generic item.
    assert final["result"]["recommendations"] == [
        "Investigate the listed causes with the available evidence"
    ]


def test_autonomous_without_evidence_uses_unknown_fallback(monkeypatch):
    """Option A: with no substantive evidence the run still SUCCEEDS
    with the same honest fallback the capability path uses — the agent
    cannot invent root causes from nothing, and validation must not
    punish that honesty with FAILED."""
    _fake_deep_agent(
        monkeypatch,
        summary="No faults found in the available data.",
    )
    final = run_mission(_autonomous_mission(), INPUT, "exec_test_auto_no_evidence")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert final["result"]["root_causes"] == ["unknown — insufficient evidence"]


def test_autonomous_recommendations_quote_agent_summary(monkeypatch):
    """Option A: recommendations quote actionable sentences from the
    agent's own summary verbatim — no specific action is invented."""
    _fake_deep_agent(
        monkeypatch,
        summary="The motor overheated. Recommend replacing the traction "
        "motor soon. The door sensor looks fine.",
        evidence=[
            {
                "source": "deep_agent", "kind": "fault",
                "summary": "F-204: Motor overheating", "stub": False,
            },
        ],
    )
    final = run_mission(_autonomous_mission(), INPUT, "exec_test_auto_reco")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert final["result"]["recommendations"] == [
        "Recommend replacing the traction motor soon."
    ]


def test_hybrid_escalation_keeps_capability_root_causes(monkeypatch):
    """Option A merge: on hybrid escalation the Deep Agent node must
    not overwrite the capability subgraph's result — its
    root_causes/recommendations survive when the agent provides none."""
    # Empty stores: capability evidence is "empty" only, so the hybrid
    # run escalates to the Deep Agent.
    monkeypatch.setattr(
        repositories, "search_fault_history", lambda *a, **k: ([], False)
    )
    monkeypatch.setattr(
        repositories, "get_maintenance_records", lambda *a, **k: ([], False)
    )
    monkeypatch.setattr(
        repositories, "search_documents", lambda *a, **k: ([], False)
    )
    monkeypatch.setattr(
        repositories, "query_asset_relationships", lambda *a, **k: ([], False)
    )
    _fake_deep_agent(
        monkeypatch,
        summary="Nothing more to add.",
    )
    mission = dict(MISSION)
    mission["execution"] = {"mode": "hybrid", "agent_id": "fault_investigator"}
    final = run_mission(mission, INPUT, "exec_test_hybrid_merge")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    # The capability aggregation's fallback + its two recommendations.
    assert final["result"]["root_causes"] == ["unknown — insufficient evidence"]
    assert final["result"]["recommendations"] == [
        "Review the collected evidence and confirm the suspected root cause on site.",
        "Schedule preventive maintenance for the affected component.",
    ]
