"""Hotel receptionist tests — YAML seed, session limits, reception skill map.

No LLM, no live infra: the MCP tools named here do not exist yet (the
PMS / staff / docs servers are not connected), so these tests pin the
CONTRACT — the agent + mission + skill map those servers must match —
and the seed behaviour. A live booking run needs the MCP servers plus
the Slack adapter (both still missing, see docs/hotel-reception.md).
"""

import pathlib

import yaml
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool as langchain_tool
from pydantic import Field

from app import models
from app.config import settings
from app.services import agent_version_service as avs
from app.services import mission_service
from runtime import deep_agent as deep_agent_module
from runtime import skill_subgraph as skill_subgraph_module
from runtime.mission_graph import run_mission
from runtime.skills import load_skills
from runtime.skills.map_plan import parse_skill_map, render_plan
from seeds import load_missions

BASE = pathlib.Path(__file__).resolve().parents[1] / "seeds"
SKILL_PATH = pathlib.Path(__file__).resolve().parents[1] / "skills" / "reception" / "SKILL.md"

# The MCP tool contract (docs/hotel-reception.md): these names must be
# what the PMS / staff / docs MCP servers expose.
HOTEL_TOOLS = {
    "check_room_availability",
    "get_room_rates",
    "find_guest",
    "register_guest",
    "create_booking",
    "get_booking",
    "cancel_booking",
    "change_booking_dates",
    "list_staff",
    "create_task",
    "assign_task",
    "get_task",
    "search_policy_documents",
    "read_policy_document",
}

MAP_NODE_IDS = [
    "route",
    "booking-dates",
    "check-availability",
    "quote-price",
    "confirm-booking",
    "create-booking",
    "report-booking",
    "task-details",
    "find-staff",
    "create-task",
    "assign-task",
    "report-task",
    "policy-lookup",
    "booking-lookup",
    "booking-change",
    "task-status",
]


def _frontmatter() -> dict:
    text = SKILL_PATH.read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---\n", 2)[1])


# --- Seed files ---
def test_seed_files_parse():
    mission = load_missions.load_mission_file(BASE / "missions" / "hotel_reception.yaml")
    assert mission["id"] == "hotel_reception"
    assert mission["agent_id"] == "hotel_receptionist"
    execution = mission["execution_config"] or {}
    assert execution.get("mode") == "autonomous"
    assert execution.get("skills") == ["reception", "session-memory"]
    assert "guard_destructive_target" in (execution.get("hooks") or [])
    assert mission["capabilities"] == ["hotel_reception"]
    assert (mission["input_schema"] or {}).get("required") == ["message"]
    assert set((mission["output_schema"] or {}).get("properties", {})) == {
        "answer", "actions_taken", "evidence",
    }
    agent = load_missions.load_agent_file(BASE / "agents" / "hotel_receptionist.yaml")
    assert agent["id"] == "hotel_receptionist"
    assert agent["model"] == settings.llm_model
    assert agent["temperature"] == 0.5
    assert agent["session_history"] == {"max_turns": 10, "max_tokens": 4096}
    assert set(agent["allowed_tools"]) == HOTEL_TOOLS
    # The confirmation rule is the load-bearing sentence of the prompt.
    prompt = agent["system_prompt"]
    assert "xác nhận" in prompt
    assert "create_booking" in prompt
    assert "lễ tân" in prompt


def test_seed_loads_hotel_reception_mission_and_agent(db_session):
    load_missions.main()
    mission = mission_service.get_mission(db_session, "hotel_reception")
    assert mission is not None
    assert mission.agent_id == "hotel_receptionist"
    assert mission.capabilities == ["hotel_reception"]
    assert (mission.execution_config or {}).get("mode") == "autonomous"
    assert (mission.input_schema or {}).get("required") == ["message"]
    agent = db_session.get(models.Agent, "hotel_receptionist")
    assert agent is not None
    assert set(agent.allowed_tools or []) == HOTEL_TOOLS
    # The seed snapshots the agent config (Option C): the pinned
    # version carries the same tools and session limits.
    version = avs.get_active_version(db_session, "hotel_receptionist")
    assert version is not None
    assert set(version.allowed_tools or []) == HOTEL_TOOLS
    assert version.session_history == {"max_turns": 10, "max_tokens": 4096}


# --- Reception skill map ---
def test_reception_skill_bans_builtin_tools():
    """The skill names the deepagents built-ins as forbidden: the map
    alone cannot stop the 3B model from reaching for them (they stay
    visible in its tool list), so the ban must be explicit text the
    model reads — the hook only denies at call time."""
    body = SKILL_PATH.read_text(encoding="utf-8").split("---\n", 2)[2]
    for name in ("ls", "read_file", "write_file", "edit_file", "delete",
                 "glob", "grep", "task", "execute"):
        assert f"`{name}`" in body, f"skill must name forbidden tool {name}"
    assert "không có quyền truy cập file" in body


def test_reception_skill_map_contract():
    """The declared map (no tool validation): the full contract."""
    skill_map = parse_skill_map(_frontmatter(), "reception")
    assert skill_map is not None
    assert skill_map.start == "route"
    assert skill_map.node_ids == MAP_NODE_IDS
    tools = {t for n in skill_map.nodes for t in n.tools}
    assert tools == HOTEL_TOOLS
    route = skill_map.node_by_id("route")
    assert {b.goto for b in route.branches} == {
        "booking-dates", "task-details", "policy-lookup", "booking-lookup",
        "booking-change", "task-status",
    }
    confirm = skill_map.node_by_id("confirm-booking")
    assert "chưa xác nhận" in (confirm.stop_when or "")
    assert skill_map.node_by_id("create-booking").tools == ["create_booking"]
    assert skill_map.node_by_id("assign-task").tools == ["assign_task"]
    # Terminal nodes (2026-11-13): END is honored only here. The
    # mid-flow nodes (booking-dates, task-details, find-staff, ...) are
    # not terminal — a premature END there falls back to the next node.
    assert {n.id for n in skill_map.nodes if n.terminal} == {
        "route", "report-booking", "report-task", "policy-lookup",
        "booking-lookup", "booking-change", "task-status",
    }
    rendered = render_plan(skill_map)
    assert "confirm-booking" in rendered
    assert "create_booking" in rendered


def test_reception_skill_loads_with_structure_intact():
    """load_skills() drops the map's tools until the MCP servers connect
    (unknown tools are dropped with a warning by design) — but the map
    structure itself must survive, so the plan still renders."""
    skills = {s.name: s for s in load_skills()}
    assert "reception" in skills
    skill_map = skills["reception"].map
    assert skill_map is not None
    assert skill_map.start == "route"
    assert skill_map.node_ids == MAP_NODE_IDS
    assert {t for n in skill_map.nodes for t in n.tools} == set()


class _ToolRecordingModel(GenericFakeChatModel):
    """Records the tool list bound at each model call — the model's
    actual view, which is what Part 1 (b) must change."""
    seen_tools: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.seen_tools.append([
            getattr(t, "name", None) or str(t) for t in tools
        ])
        return self


@langchain_tool
def _dummy_mcp_tool() -> str:
    """Stand-in for an MCP tool."""
    return "ok"


def test_hotel_agent_model_sees_no_builtin_tools(monkeypatch):
    """Part 1 (b): for hotel_reception the model is never offered the
    deepagents built-ins; for other missions they stay (fault needs
    `execute` in its sandbox). A recording model captures the bound
    tools — the model's actual view, not the registry's."""
    def _bound_tool_names(middleware):
        model = _ToolRecordingModel(messages=iter([AIMessage(content="done")] * 8))
        monkeypatch.setattr(deep_agent_module, "get_chat_model", lambda: model)
        agent = deep_agent_module.build_deep_agent(
            [_dummy_mcp_tool], system_prompt="test prompt", middleware=middleware
        )
        agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
        assert model.seen_tools, "model was never called with tools"
        return model.seen_tools[0]

    hotel_tools = _bound_tool_names(
        [deep_agent_module.builtin_tools_exclusion("hotel_reception")]
    )
    assert hotel_tools == ["_dummy_mcp_tool"]
    assert deep_agent_module.builtin_tools_exclusion("fault_analysis") is None
    fault_tools = _bound_tool_names(None)
    assert "_dummy_mcp_tool" in fault_tools
    assert {"ls", "read_file", "write_file", "edit_file", "delete",
            "glob", "grep", "task"} <= set(fault_tools)


def test_deep_agent_node_adds_builtin_exclusion_for_hotel(db_session, monkeypatch):
    """The Mission Graph passes the built-in exclusion middleware to the
    hotel subgraph's node agents (and only there) — the per-run
    middleware list is where the decision lives, not in
    build_deep_agent's signature."""
    load_missions.main()
    mission = mission_service.get_mission(db_session, "hotel_reception")
    mission_dict = mission_service.mission_to_dict(mission)

    seen: dict = {}

    def fake_build_subgraph(skill_map, tools, system_prompt=None, **kwargs):
        seen["middleware"] = list(kwargs.get("middleware") or [])
        return object()

    def fake_run_subgraph(compiled, mission_arg, input_data, **kwargs):
        return {
            "summary": "ok",
            "evidence": [{
                "source": "deep_agent", "kind": "summary",
                "summary": "ok", "stub": False,
            }],
            "actions_taken": [],
            "errors": [],
        }

    monkeypatch.setattr(
        skill_subgraph_module, "build_skill_subgraph", fake_build_subgraph)
    monkeypatch.setattr(
        skill_subgraph_module, "run_skill_subgraph", fake_run_subgraph)

    final = run_mission(mission_dict, {"message": "hi"}, "exec_hotel_wiring")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    middleware = seen.get("middleware") or []
    exclusions = [m for m in middleware
                  if type(m).__name__ == "_ToolExclusionMiddleware"]
    assert len(exclusions) == 1
    assert exclusions[0]._excluded == deep_agent_module.BUILTIN_TOOL_NAMES
