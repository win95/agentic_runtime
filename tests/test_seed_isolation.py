"""Seed isolation: one broken YAML must not stop the rest, and one
agent/mission can be seeded by name.

`load_missions.main()` reads from the module-level BASE, so the tests
point it at a tmp dir with a mix of good and broken files.
"""

import pathlib

import pytest

from app import models
from seeds import load_missions

GOOD_MISSION = """\
mission_id: test_mission
version: 1
name: Test Mission
objective: Test mission.
input_schema:
  type: object
  properties:
    message:
      type: string
  required: [message]
output_schema:
  type: object
  properties:
    answer:
      type: string
execution:
  mode: autonomous
  agent_id: test_agent
capabilities: [test_cap]
"""

GOOD_AGENT = """\
id: test_agent
name: Test Agent
provider: openai
model: ${LLM_MODEL}
temperature: 0.0
session_history: {max_turns: 3, max_tokens: 2048}
allowed_tools: []
"""

OTHER_AGENT = """\
id: other_agent
name: Other Agent
provider: openai
model: ${LLM_MODEL}
temperature: 0.0
allowed_tools: []
"""

GOOD_SCHEDULE = """\
id: sched_test
mission_id: test_mission
cron: "0 7 * * *"
enabled: false
"""


def _write(base: pathlib.Path, group: str, name: str, content: str) -> None:
    directory = base / group
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(content, encoding="utf-8")


@pytest.fixture
def tmp_base(tmp_path, monkeypatch):
    monkeypatch.setattr(load_missions, "BASE", tmp_path)
    return tmp_path


def test_bad_files_do_not_stop_the_rest(db_session, tmp_base, capsys):
    _write(tmp_base, "missions", "test_mission.yaml", GOOD_MISSION)
    _write(tmp_base, "missions", "bad_mission.yaml", "version: 1\n")
    _write(tmp_base, "agents", "test_agent.yaml", GOOD_AGENT)
    _write(tmp_base, "agents", "broken.yaml", "id: [unclosed\n")
    _write(tmp_base, "agents", "bad_agent.yaml",
           "id: bad_agent\nsession_history: {max_turns: 0}\n")
    _write(tmp_base, "schedules", "good_sched.yaml", GOOD_SCHEDULE)
    _write(tmp_base, "schedules", "bad_sched.yaml",
           "id: sched_bad\nmission_id: test_mission\n")

    failures = load_missions.main()

    assert len(failures) == 4
    failed_names = {pathlib.Path(p).name for p, _ in failures}
    assert failed_names == {
        "bad_mission.yaml", "broken.yaml", "bad_agent.yaml", "bad_sched.yaml",
    }
    # The good files still seeded.
    assert db_session.get(models.Mission, "test_mission") is not None
    assert db_session.get(models.Agent, "test_agent") is not None
    assert db_session.get(models.Schedule, "sched_test") is not None
    assert db_session.get(models.Agent, "bad_agent") is None
    out = capsys.readouterr().out
    assert "Seed done: 1 missions, 1 agents, 1 schedules, 4 failed" in out


def test_seed_single_agent_only(db_session, tmp_base):
    _write(tmp_base, "missions", "test_mission.yaml", GOOD_MISSION)
    _write(tmp_base, "agents", "test_agent.yaml", GOOD_AGENT)
    _write(tmp_base, "agents", "other_agent.yaml", OTHER_AGENT)
    _write(tmp_base, "schedules", "good_sched.yaml", GOOD_SCHEDULE)

    failures = load_missions.main(agent="test_agent")

    assert failures == []
    assert db_session.get(models.Agent, "test_agent") is not None
    assert db_session.get(models.Agent, "other_agent") is None
    assert db_session.get(models.Mission, "test_mission") is None
    assert db_session.get(models.Schedule, "sched_test") is None


def test_seed_single_mission_only(db_session, tmp_base):
    _write(tmp_base, "missions", "test_mission.yaml", GOOD_MISSION)
    _write(tmp_base, "agents", "test_agent.yaml", GOOD_AGENT)

    failures = load_missions.main(mission="test_mission")

    assert failures == []
    assert db_session.get(models.Mission, "test_mission") is not None
    assert db_session.get(models.Agent, "test_agent") is None


def test_seed_unknown_filter_lists_available(tmp_base):
    _write(tmp_base, "agents", "test_agent.yaml", GOOD_AGENT)

    with pytest.raises(ValueError, match="available: test_agent"):
        load_missions.main(agent="nope")
