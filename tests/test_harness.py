"""Declarative mission harness — validation and policy equivalence."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from runtime.deep_agent import ADMIN_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
from runtime.harness.schema import GuestChannelHarness, MissionHarness
from runtime.harness.loader import (
    load_mission_harness_from_file,
    parse_agent_harness,
    parse_mission_harness,
)
from runtime.harness.registry import (
    builtin_deny_message,
    effective_harness,
    guest_chat_mission_id,
    guest_output_capabilities,
    mission_denies_builtin_tools,
    reset_harness_cache_for_tests,
    use_flow_subgraph_harness,
    user_runnable_mission_ids,
)
from runtime.harness.validation import format_deny_message, validate_cross_mission_harness
from runtime.skills.loader import load_skills, select_skills
from seeds import load_missions

BASE = Path(__file__).resolve().parents[1] / "seeds"
HOTEL_DENY_SUFFIX = (
    "the hotel receptionist has no file or subagent tools — all room, "
    "guest, booking, staff and policy actions go through the MCP tools "
    "(check_room_availability, get_booking, create_booking, ...). Do not "
    "retry this tool; use the MCP tool for the task instead."
)


@pytest.fixture(autouse=True)
def _fresh_harness_cache():
    reset_harness_cache_for_tests()
    yield
    reset_harness_cache_for_tests()


def test_defaults_validate():
    h = parse_mission_harness(None)
    assert h.security.user_runnable is False
    assert h.flow_subgraph.enabled is False
    assert h.builtin_tools.deny is False
    assert h.context_budget.reclaim_retrieval_for_history is False


def test_deny_requires_message_template():
    with pytest.raises(ValueError, match="deny_message_template"):
        parse_mission_harness({"builtin_tools": {"deny": True}})


def test_harness_rejects_unknown_builtin_keys():
    with pytest.raises(ValueError):
        parse_mission_harness({"builtin_tools": {"deni": True}})


def test_harness_tool_calls_rejects_extra_fields():
    with pytest.raises(ValueError):
        parse_mission_harness({"tool_calls": {"sequential": ["a"], "extra": True}})


def test_harness_output_files_rejects_extra_fields():
    with pytest.raises(ValueError):
        parse_mission_harness({"output_files": {"required": True, "bogus": True}})


def test_harness_output_artifacts_merge_defaults():
    h = parse_mission_harness({"output_artifacts": {"sink": "minio"}})
    assert h.output_artifacts.sink == "minio"
    assert h.output_artifacts.strip_inline is True


def test_harness_output_artifacts_rejects_extra_fields():
    with pytest.raises(ValueError):
        parse_mission_harness({"output_artifacts": {"sink": "none", "bogus": True}})


def test_harness_context_budget_rejects_extra_fields():
    with pytest.raises(ValueError):
        parse_mission_harness(
            {"context_budget": {"reclaim_retrieval_for_history": True, "extra": True}}
        )


def test_deny_template_rejects_doubled_braces():
    with pytest.raises(ValueError, match="escaped braces"):
        parse_mission_harness(
            {
                "builtin_tools": {
                    "deny": True,
                    "deny_message_template": "Tool '{{tool_name}}' denied",
                }
            }
        )


def test_deny_template_rejects_invalid_fields():
    with pytest.raises(ValueError, match="only"):
        parse_mission_harness(
            {
                "builtin_tools": {
                    "deny": True,
                    "deny_message_template": "Tool '{other}' denied",
                }
            }
        )


def test_hotel_equivalence():
    path = BASE / "missions" / "hotel_reception.yaml"
    h = load_mission_harness_from_file(path)
    assert h.security.user_runnable is True
    assert h.guest_channel.exposes_public_chat is True
    assert h.builtin_tools.deny is True
    assert h.flow_subgraph.enabled is True
    assert h.guest_output.register_reply_tools is True
    assert h.guest_output.capability == "hotel_reception"


def test_fault_and_admin_equivalence():
    fault = load_mission_harness_from_file(BASE / "missions" / "fault_analysis.yaml")
    admin = load_mission_harness_from_file(BASE / "missions" / "system_admin.yaml")
    for h in (fault, admin):
        assert h.security.user_runnable is False
        assert h.guest_channel.exposes_public_chat is False
        assert h.builtin_tools.deny is False
        assert h.flow_subgraph.enabled is False
        assert h.guest_output.register_reply_tools is False


def test_registry_matches_old_constants():
    assert user_runnable_mission_ids() == frozenset({"hotel_reception"})
    assert guest_chat_mission_id() == "hotel_reception"
    assert mission_denies_builtin_tools("hotel_reception") is True
    assert mission_denies_builtin_tools("fault_analysis") is False
    assert use_flow_subgraph_harness("hotel_reception", ["reception"]) is True
    assert use_flow_subgraph_harness("fault_analysis", ["reception"]) is False
    assert guest_output_capabilities() == ["hotel_reception"]


def test_builtin_deny_message_exact_hotel_text():
    tpl = load_mission_harness_from_file(
        BASE / "missions" / "hotel_reception.yaml"
    ).builtin_tools.deny_message_template
    assert tpl is not None
    expected = format_deny_message(tpl, "read_file")
    msg = builtin_deny_message("hotel_reception", "read_file")
    assert msg == expected
    assert msg.startswith(
        "Tool 'read_file' denied by hook 'deny_builtin_tools': "
    )
    assert HOTEL_DENY_SUFFIX in msg


def test_execution_config_harness_overrides_seed_for_builtin_deny():
    override = {
        "harness": {
            "builtin_tools": {
                "deny": False,
                "deny_message_template": None,
            }
        }
    }
    assert mission_denies_builtin_tools("hotel_reception") is True
    assert mission_denies_builtin_tools("hotel_reception", override) is False
    assert use_flow_subgraph_harness(
        "hotel_reception", ["reception"], {"harness": {"flow_subgraph": {"enabled": False}}}
    ) is False


def test_duplicate_guest_chat_mission_rejected_at_validation():
    hotel = load_mission_harness_from_file(BASE / "missions" / "hotel_reception.yaml")
    other = MissionHarness(
        guest_channel=GuestChannelHarness(exposes_public_chat=True),
    )
    with pytest.raises(ValueError, match="exposes_public_chat"):
        validate_cross_mission_harness(
            [("hotel_reception", hotel), ("other_mission", other)]
        )


def test_load_mission_file_embeds_harness():
    row = load_missions.load_mission_file(BASE / "missions" / "hotel_reception.yaml")
    harness = row["execution_config"]["harness"]
    assert harness["security"]["user_runnable"] is True


def test_agent_harness_prompt_fallbacks():
    fault = parse_agent_harness(
        yaml.safe_load(
            (BASE / "agents" / "fault_investigator.yaml").read_text()
        ).get("harness")
    )
    admin = parse_agent_harness(
        yaml.safe_load((BASE / "agents" / "admin_bot.yaml").read_text()).get("harness")
    )
    assert fault.system_prompt_fallback == "default"
    assert admin.system_prompt_fallback == "admin"
    inv = load_missions.load_agent_file(BASE / "agents" / "fault_investigator.yaml")
    assert inv["system_prompt"] == DEFAULT_SYSTEM_PROMPT
    adm = load_missions.load_agent_file(BASE / "agents" / "admin_bot.yaml")
    assert adm["system_prompt"] == ADMIN_SYSTEM_PROMPT


def test_new_agent_skill_slug_uses_removesuffix(tmp_path):
    from scripts.new_agent import scaffold

    paths = scaffold("my_bottle_bot", repo_root=tmp_path)
    assert paths["skill"].parent.name == "my-bottle"


def test_scaffolded_agent_resolves_without_python(tmp_path):
    from scripts.new_agent import scaffold

    paths = scaffold(
        "pytest_demo_bot",
        capability="pytest_demo",
        repo_root=tmp_path,
    )
    reset_harness_cache_for_tests()
    mission = load_missions.load_mission_file(paths["mission"])
    agent = load_missions.load_agent_file(paths["agent"])
    assert agent["id"] == "pytest_demo_bot"
    assert mission["id"] == "pytest_demo_bot_mission"
    h = effective_harness(mission["id"], mission["execution_config"])
    assert h.security.user_runnable is False
    skill_slug = "pytest-demo"
    from runtime.skills import loader as skills_loader

    original = skills_loader.DEFAULT_SKILLS_DIR
    try:
        skills_loader.DEFAULT_SKILLS_DIR = tmp_path / "skills"
        loaded = load_skills()
        names = {s.name for s in loaded}
        assert skill_slug in names
        mission_dict = {
            "mission_id": mission["id"],
            "objective": mission["objective"],
            "execution": mission["execution_config"],
        }
        selected = select_skills(
            mission_dict,
            {"message": "pytest demo request"},
            skills=loaded,
        )
        assert any(s.name == skill_slug for s in selected)
    finally:
        skills_loader.DEFAULT_SKILLS_DIR = original