"""Skill-map tests — map parsing, plan rendering, deterministic tracing.

No LLM, no live services: the plan trace is computed from recorded
tool calls, and the Mission Graph runs are driven with mocked agent
factories like the other graph tests.
"""

from types import SimpleNamespace

from langchain_core.messages import ToolMessage

import runtime.deep_agent as deep_agent_module
import seeds.load_missions as load_missions
from app.services import mission_service
from runtime.deep_agent import run_deep_agent
from runtime.mission_graph import run_mission
from runtime.skills import load_skills

KNOWN = {"search_fault_history", "get_sensor_data", "query_asset_relationships"}

BASE_MAP = {
    "start": "identify",
    "nodes": [
        {"id": "identify", "goal": "Parse the input.", "tools": []},
        {
            "id": "postgres-faults",
            "goal": "Search fault history.",
            "tools": ["search_fault_history"],
            "branches": [{"when": "no faults found", "goto": "postgres-maintenance"}],
        },
        {"id": "postgres-maintenance", "goal": "Get maintenance records.", "tools": []},
        {
            "id": "postgres-sensors",
            "goal": "Get sensor readings.",
            "tools": ["get_sensor_data", "no-such-tool"],
            "branches": [{"when": "nothing to correlate", "goto": "graph-relations"}],
        },
        {
            "id": "graph-relations",
            "goal": "Query relations.",
            "tools": ["query_asset_relationships"],
        },
        {
            "id": "synthesize",
            "goal": "Draft findings.",
            "tools": [],
            "stop_when": "enough evidence",
        },
    ],
}


def _frontmatter(map_value):
    return {"name": "x", "description": "X skill", "map": map_value}


# --- Parsing ---


def test_parse_skill_map_reads_terminal_flag():
    """A node may declare `terminal: true` (the router honors END only
    there); anything else is not terminal, including a non-boolean."""
    from runtime.skills.map_plan import parse_skill_map

    parsed = parse_skill_map(_frontmatter({
        "start": "a",
        "nodes": [
            {"id": "a", "goal": "start"},
            {"id": "b", "goal": "end", "terminal": True},
            {"id": "c", "goal": "not a bool", "terminal": "yes"},
        ],
    }), "x")
    assert parsed is not None
    assert parsed.node_by_id("a").terminal is False
    assert parsed.node_by_id("b").terminal is True
    assert parsed.node_by_id("c").terminal is False


# --- Loader: pilot skill + map-less skills unchanged ---
def test_loader_parses_data_access_map():
    skills = {s.name: s for s in load_skills()}
    assert "data-access" in skills
    skill_map = skills["data-access"].map
    assert skill_map is not None
    assert skill_map.start == "identify"
    assert skill_map.node_ids == [
        "identify",
        "postgres-faults",
        "postgres-maintenance",
        "postgres-sensors",
        "graph-relations",
        "qdrant-documents",
        "synthesize",
    ]
    tools = {t for n in skill_map.nodes for t in n.tools}
    assert tools == {
        "search_fault_history",
        "get_maintenance_records",
        "get_sensor_data",
        "query_asset_relationships",
        "search_technical_documents",
    }


def test_loader_mapless_skills_have_no_map():
    skills = {s.name: s for s in load_skills()}
    assert skills["fault-investigation"].map is None
    assert skills["admin-operations"].map is None
    assert skills["report-writing"].map is None


# --- Rendering ---


# --- Trace computation ---


# --- run_deep_agent computes the trace from real tool calls ---
class _ToolCallAgent:
    """Recording agent returning canned tool calls + ToolMessages."""

    def __init__(self, calls):
        # calls: list of (tool name, ok: bool)
        self.calls = calls

    def invoke(self, state, config=None):
        messages = []
        for i, (name, ok) in enumerate(self.calls):
            messages.append(
                SimpleNamespace(
                    content="",
                    tool_calls=[{"name": name, "id": f"c{i}", "args": {}}],
                )
            )
            content = "{}" if ok else '{"error": "boom"}'
            messages.append(ToolMessage(content=content, tool_call_id=f"c{i}"))
        messages.append(SimpleNamespace(content="done"))
        return {"messages": messages}


def test_run_deep_agent_plan_trace_from_tool_calls():
    skill_map = next(s for s in load_skills() if s.name == "data-access").map
    agent = _ToolCallAgent(
        [
            ("search_fault_history", True),
            ("get_sensor_data", False),  # failed call still visits its node
            ("calculate_failure_rate", True),  # not in the map: deviation
        ]
    )
    outcome = run_deep_agent(
        agent,
        {"mission_id": "fault_analysis", "objective": "Analyze faults."},
        {"asset_id": "O21"},
        skill_maps=[skill_map],
    )
    trace = outcome["plan_trace"]["data-access"]
    assert trace["visited"] == ["postgres-faults", "postgres-sensors"]
    assert trace["deviations"] == [{"tool": "calculate_failure_rate"}]
    # Grounding is separate from the trace: the failed sensor call is
    # NOT an action taken, but it still visited its node.
    assert outcome["actions_taken"] == ["search_fault_history", "calculate_failure_rate"]
    assert any("get_sensor_data" in e for e in outcome["errors"])


def test_run_deep_agent_no_maps_no_trace_key():
    agent = _ToolCallAgent([("search_fault_history", True)])
    outcome = run_deep_agent(
        agent,
        {"mission_id": "fault_analysis", "objective": "Analyze faults."},
        {"asset_id": "O21"},
    )
    assert "plan_trace" not in outcome


# --- deep_agent_node: plan injection + trace merge + old-double fallback ---
def _fake_outcome_with_trace():
    return {
        "summary": "Done.",
        "evidence": [
            {"source": "deep_agent", "kind": "summary", "summary": "Done.", "stub": False}
        ],
        "actions_taken": ["search_fault_history", "query_asset_relationships"],
        "errors": [],
        "plan_trace": {
            "data-access": {
                "visited": ["postgres-faults", "graph-relations"],
                "tools_by_node": {
                    "postgres-faults": ["search_fault_history"],
                    "graph-relations": ["query_asset_relationships"],
                },
                "skipped": ["identify", "postgres-maintenance", "postgres-sensors",
                            "qdrant-documents", "synthesize"],
                "deviations": [],
            }
        },
    }


def test_deep_agent_node_injects_plan_and_merges_trace(db_session, monkeypatch):
    load_missions.main()
    mission = mission_service.get_mission(db_session, "fault_analysis")
    mission_dict = mission_service.mission_to_dict(mission)
    # Pin the autonomous path: a hybrid run may satisfy the capability
    # graph and route straight to validate (evidence-dependent), which
    # would skip deep_agent_node and this test's subject entirely.
    mission_dict["execution"]["mode"] = "autonomous"
    seen: dict = {}

    def fake_build(tools, system_prompt=None, skills=None, model_config=None,
                   middleware=None, backend=None):
        seen["system_prompt"] = system_prompt
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None,
                 skill_maps=None):
        seen["skill_maps"] = skill_maps
        return _fake_outcome_with_trace()

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)

    final = run_mission(
        mission_dict,
        {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"},
        "exec-plan-trace",
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert seen["skill_maps"] is not None and len(seen["skill_maps"]) == 1
    assert seen["skill_maps"][0].skill_name == "data-access"
    assert "Plan — skill 'data-access'" in (seen["system_prompt"] or "")
    assert final["result"]["plan_trace"]["data-access"]["visited"] == [
        "postgres-faults",
        "graph-relations",
    ]
    assert any("Plan (data-access):" in n for n in final["notes"])


def test_deep_agent_node_old_double_without_skill_maps_still_works(db_session, monkeypatch):
    """Doubles predating skill_maps (fixed signature) keep working: the
    node retries without the kwarg and the run proceeds, minus the trace."""
    load_missions.main()
    mission = mission_service.get_mission(db_session, "fault_analysis")
    mission_dict = mission_service.mission_to_dict(mission)
    # Autonomous mode guarantees deep_agent_node runs (see the test above).
    mission_dict["execution"]["mode"] = "autonomous"

    def fake_build(tools, system_prompt=None, skills=None, model_config=None,
                   middleware=None, backend=None):
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None):
        return {
            "summary": "Done.",
            "evidence": [
                {"source": "deep_agent", "kind": "summary", "summary": "Done.", "stub": False}
            ],
            "actions_taken": [],
            "errors": [],
        }

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)

    final = run_mission(
        mission_dict,
        {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"},
        "exec-plan-old-double",
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert "plan_trace" not in final["result"]


# --- Mission-level branch traces for the edge-case assets ---
#
# The mock eval harness's fake model makes NO tool calls, so plan_trace
# is empty there and asserting visited nodes in eval is meaningless —
# these tests assert the branches with the recording fake model
# instead. The tool sequences below are the ones the data-access map
# prescribes for each asset (see skills/data-access/SKILL.md):
# O23 (healthy): faults empty -> maintenance, then faults+maintenance
# both empty -> skip sensors, go to graph-relations.
# O24 (maintenance-only): faults empty -> maintenance (non-empty),
# so sensors stay on the path.
def _run_with_tool_sequence(db_session, monkeypatch, asset_id, calls):
    load_missions.main()
    mission = mission_service.get_mission(db_session, "fault_analysis")
    mission_dict = mission_service.mission_to_dict(mission)
    # Autonomous mode guarantees deep_agent_node runs (see the test above).
    mission_dict["execution"]["mode"] = "autonomous"
    monkeypatch.setattr(
        deep_agent_module,
        "build_deep_agent",
        lambda tools, system_prompt=None, skills=None, model_config=None,
        middleware=None, backend=None: _ToolCallAgent(calls),
    )
    return run_mission(
        mission_dict,
        {"asset_id": asset_id, "start_date": "2026-10-01", "end_date": "2026-10-31"},
        f"exec-plan-{asset_id.lower()}",
    )


def test_o23_healthy_branch_skips_sensors(db_session, monkeypatch):
    final = _run_with_tool_sequence(db_session, monkeypatch, "O23", [
        ("search_fault_history", True),
        ("get_maintenance_records", True),
        ("query_asset_relationships", True),
        ("search_technical_documents", True),
    ])
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    trace = final["result"]["plan_trace"]["data-access"]
    assert trace["visited"] == [
        "postgres-faults",
        "postgres-maintenance",
        "graph-relations",
        "qdrant-documents",
    ]
    assert trace["skipped"] == ["identify", "postgres-sensors", "synthesize"]
    assert trace["deviations"] == []


def test_o24_maintenance_only_branch_keeps_sensors(db_session, monkeypatch):
    final = _run_with_tool_sequence(db_session, monkeypatch, "O24", [
        ("search_fault_history", True),
        ("get_maintenance_records", True),
        ("get_sensor_data", True),
        ("query_asset_relationships", True),
        ("search_technical_documents", True),
    ])
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    trace = final["result"]["plan_trace"]["data-access"]
    assert trace["visited"] == [
        "postgres-faults",
        "postgres-maintenance",
        "postgres-sensors",
        "graph-relations",
        "qdrant-documents",
    ]
    assert trace["skipped"] == ["identify", "synthesize"]
    assert trace["deviations"] == []
