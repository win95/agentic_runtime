"""Declarative mission harness — validation and policy equivalence."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from runtime.deep_agent import ADMIN_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
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
    user_runnable_mission_ids,
    use_flow_subgraph_harness,
)
from seeds import load_missions

BASE = Path(__file__).resolve().parents[1] / "seeds"


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


def test_deny_requires_message_template():
    with pytest.raises(ValueError, match="deny_message_template"):
        parse_mission_harness({"builtin_tools": {"deny": True}})


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


def test_builtin_deny_message_uses_template():
    msg = builtin_deny_message("hotel_reception", "read_file")
    assert msg is not None
    assert "read_file" in msg
    assert "MCP tools" in msg


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


def test_scaffolded_agent_resolves_without_python(tmp_path):
    from scripts.new_agent import scaffold

    paths = scaffold("pytest_demo_bot", capability="pytest_demo")
    try:
        reset_harness_cache_for_tests()
        mission = load_missions.load_mission_file(paths["mission"])
        agent = load_missions.load_agent_file(paths["agent"])
        assert agent["id"] == "pytest_demo_bot"
        assert mission["id"] == "pytest_demo_bot_mission"
        h = effective_harness(mission["id"], mission["execution_config"])
        assert h.security.user_runnable is False
        parse_mission_harness(None)  # defaults still valid
    finally:
        for p in paths.values():
            p.unlink(missing_ok=True)
        paths["skill"].parent.rmdir()