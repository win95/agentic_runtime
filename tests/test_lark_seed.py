"""Seed load for agent_lark_me + mission — YAML parses and seeds cleanly."""

import pathlib

from app.config import settings
from app.core.db import SessionLocal
from app.models import Agent, Mission
from seeds import load_missions

BASE = pathlib.Path(load_missions.__file__).resolve().parent


def test_lark_yaml_files_parse():
    agent = load_missions.load_agent_file(BASE / "agents" / "agent_lark_me.yaml")
    assert agent["id"] == "agent_lark_me"
    assert "lark_calendar_freebusy" in agent["allowed_tools"]
    assert "lark_list_tasks" in agent["allowed_tools"]
    assert agent["model"] == settings.llm_model  # ${LLM_MODEL} default

    mission = load_missions.load_mission_file(
        BASE / "missions" / "agent_lark_me_mission.yaml"
    )
    assert mission["id"] == "agent_lark_me_mission"
    assert mission["agent_id"] == "agent_lark_me"
    assert mission["execution_config"]["sandbox"] is False
    assert mission["execution_config"]["skills"] == ["agent-lark", "session-memory"]
    assert mission["capabilities"] == [
        "lark_messaging",
        "lark_tasks",
        "lark_calendar",
    ]
    tools = mission["execution_config"]["harness"]["tool_budget"]["tools"]
    assert "lark_calendar_freebusy" in tools
    assert "anyOf" in mission["input_schema"]


def test_lark_agent_and_mission_seed(db_session):
    failures = load_missions.main(
        agent="agent_lark_me", mission="agent_lark_me_mission"
    )
    assert failures == []
    db = SessionLocal()
    agent = db.get(Agent, "agent_lark_me")
    assert agent is not None
    mission = db.get(Mission, "agent_lark_me_mission")
    assert mission is not None
    assert mission.agent_id == "agent_lark_me"
    harness = mission.execution_config["harness"]
    assert harness["flow_subgraph"]["enabled"] is True
    assert harness["tool_budget"]["max_calls"]["lark_send_message"] == 5
    assert harness["security"]["user_runnable"] is False
