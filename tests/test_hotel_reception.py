"""Hotel receptionist tests — YAML seed, session limits, reception skill map.

No LLM, no live infra: the MCP tools named here do not exist yet (the
PMS / staff / docs servers are not connected), so these tests pin the
CONTRACT — the agent + mission + skill map those servers must match —
and the seed behaviour. A live booking run needs the MCP servers plus
the Slack adapter (both still missing, see docs/hotel-reception.md).
"""

import pathlib

import yaml

from app import models
from app.config import settings
from app.services import agent_version_service as avs
from app.services import mission_service
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
    assert agent["temperature"] == 0.0
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
