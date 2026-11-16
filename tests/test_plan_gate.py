"""Skill map coverage + flow subgraph wiring (PlanGate removed — flow compiler enforces)."""

import pathlib

import yaml

from runtime.skills.map_plan import parse_skill_map, render_plan
from runtime.subgraph.flow import use_flow_subgraph
from seeds import load_missions

SKILL_PATH = pathlib.Path(__file__).resolve().parents[1] / "skills" / "reception" / "SKILL.md"

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
    "record_payment",
    "list_payments",
    "confirm_payment",
}


def _reception_map():
    text = SKILL_PATH.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(text.split("---\n", 2)[1])
    skill_map = parse_skill_map(frontmatter, "reception", known_tools=None)
    assert skill_map is not None
    return skill_map


def test_reception_map_covers_all_14_contract_tools():
    skill_map = _reception_map()
    covered = {tool for node in skill_map.nodes for tool in node.tools}
    assert HOTEL_TOOLS <= covered


def test_render_plan_advisory_header():
    skill_map = _reception_map()
    plan = render_plan(skill_map)
    assert "may deviate" in plan
    assert "Start at" in plan


def test_use_flow_subgraph_scope():
    skill_map = _reception_map()
    assert use_flow_subgraph("fault_analysis", [skill_map]) is False
    assert use_flow_subgraph("hotel_reception", [skill_map]) is True

