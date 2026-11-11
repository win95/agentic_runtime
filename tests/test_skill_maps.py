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
from runtime.context_budget import estimate_tokens
from runtime.deep_agent import run_deep_agent
from runtime.mission_graph import run_mission
from runtime.skills import load_skills
from runtime.skills.map_plan import (
    compute_plan_trace,
    parse_skill_map,
    plan_trace_note,
    render_plan,
)

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
def test_parse_valid_map():
    skill_map = parse_skill_map(_frontmatter(BASE_MAP), "x", known_tools=KNOWN)
    assert skill_map is not None
    assert skill_map.start == "identify"
    assert skill_map.node_ids == [
        "identify",
        "postgres-faults",
        "postgres-maintenance",
        "postgres-sensors",
        "graph-relations",
        "synthesize",
    ]
    assert skill_map.node_by_id("postgres-faults").branches[0].goto == "postgres-maintenance"
    assert skill_map.node_by_id("synthesize").stop_when == "enough evidence"


def test_parse_no_map_returns_none():
    assert parse_skill_map({"name": "x", "description": "X"}, "x") is None
    assert parse_skill_map(None, "x") is None


def test_parse_malformed_map_returns_none():
    assert parse_skill_map(_frontmatter("not-a-map"), "x") is None
    assert parse_skill_map(_frontmatter({"nodes": []}), "x") is None
    assert parse_skill_map(_frontmatter({"nodes": [{"goal": "no id"}]}), "x") is None


def test_parse_unknown_tool_dropped_with_others_kept():
    skill_map = parse_skill_map(_frontmatter(BASE_MAP), "x", known_tools=KNOWN)
    assert skill_map is not None
    assert skill_map.node_by_id("postgres-sensors").tools == ["get_sensor_data"]


def test_parse_unknown_goto_dropped():
    bad = {
        "start": "a",
        "nodes": [
            {"id": "a", "goal": "A", "branches": [{"when": "x", "goto": "ghost"}]},
            {"id": "b", "goal": "B"},
        ],
    }
    skill_map = parse_skill_map(_frontmatter(bad), "x", known_tools=set())
    assert skill_map is not None
    assert skill_map.node_by_id("a").branches == []


def test_parse_duplicate_node_keeps_first_and_bad_start_falls_back():
    dup = {
        "start": "ghost",
        "nodes": [
            {"id": "a", "goal": "first"},
            {"id": "a", "goal": "second"},
            {"id": "b", "goal": "B"},
        ],
    }
    skill_map = parse_skill_map(_frontmatter(dup), "x", known_tools=set())
    assert skill_map is not None
    assert skill_map.start == "a"
    assert skill_map.node_by_id("a").goal == "first"
    assert skill_map.node_ids == ["a", "b"]


def test_parse_without_known_tools_accepts_all():
    skill_map = parse_skill_map(_frontmatter(BASE_MAP), "x")
    assert skill_map is not None
    assert skill_map.node_by_id("postgres-sensors").tools == [
        "get_sensor_data",
        "no-such-tool",
    ]


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
def test_render_plan_compact_and_complete():
    skill_map = next(s for s in load_skills() if s.name == "data-access").map
    plan = render_plan(skill_map)
    for node_id in skill_map.node_ids:
        assert node_id in plan
    for tool in ("search_fault_history", "query_asset_relationships", "search_technical_documents"):
        assert tool in plan
    assert "deviate" in plan  # the Option C contract is stated
    assert estimate_tokens(plan) <= 600, f"plan is {estimate_tokens(plan)} tokens"


# --- Trace computation ---
def test_compute_plan_trace_visited_skipped_deviations():
    skill_map = parse_skill_map(_frontmatter(BASE_MAP), "x", known_tools=KNOWN)
    trace = compute_plan_trace(
        skill_map,
        ["search_fault_history", "get_sensor_data", "calculate_failure_rate"],
    )
    assert trace["visited"] == ["postgres-faults", "postgres-sensors"]
    assert trace["tools_by_node"] == {
        "postgres-faults": ["search_fault_history"],
        "postgres-sensors": ["get_sensor_data"],
    }
    assert trace["skipped"] == [
        "identify",
        "postgres-maintenance",
        "graph-relations",
        "synthesize",
    ]
    assert trace["deviations"] == [{"tool": "calculate_failure_rate"}]


def test_compute_plan_trace_empty_calls():
    skill_map = parse_skill_map(_frontmatter(BASE_MAP), "x", known_tools=KNOWN)
    trace = compute_plan_trace(skill_map, [])
    assert trace["visited"] == []
    assert trace["skipped"] == skill_map.node_ids
    assert trace["deviations"] == []


def test_plan_trace_note_format():
    trace = {
        "visited": ["identify", "postgres-faults", "qdrant-documents"],
        "tools_by_node": {},
        "skipped": ["postgres-sensors"],
        "deviations": [{"tool": "calculate_failure_rate"}],
    }
    note = plan_trace_note("data-access", trace)
    assert note == (
        "Plan (data-access): visited identify -> postgres-faults -> qdrant-documents; "
        "skipped postgres-sensors; deviation: calculate_failure_rate (not in map)"
    )


def test_plan_trace_note_empty_calls():
    skill_map = parse_skill_map(_frontmatter(BASE_MAP), "x", known_tools=KNOWN)
    note = plan_trace_note("data-access", compute_plan_trace(skill_map, []))
    assert note == (
        "Plan (data-access): skipped identify, postgres-faults, "
        "postgres-maintenance, postgres-sensors, graph-relations, synthesize"
    )


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
                   middleware=None):
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
                   middleware=None):
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
