"""Skills tests — loader, deterministic selection, and Deep Agent wiring.

No LLM, no live services: the native deepagents skills mechanism is
exercised with a recording fake model (``bind_tools`` returns self), and
the Mission Graph runs are driven with mocked agent factories like the
other graph tests.
"""

from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

import runtime.deep_agent as deep_agent_module
import seeds.load_missions as load_missions
from app.services import mission_service
from runtime.deep_agent import build_deep_agent, run_deep_agent
from runtime.mission_graph import run_mission
from runtime.skills import load_skills, select_skills, skill_files_for_run

BASE = load_missions.BASE

FAULT_MISSION = {
    "mission_id": "fault_analysis",
    "name": "Fault Analysis",
    "version": 1,
    "objective": "Analyze equipment faults and identify root causes.",
    "input_schema": {"type": "object", "properties": {}},
    "output_schema": {"type": "object", "properties": {}},
    "execution": {"mode": "hybrid", "agent_id": "fault_investigator"},
    "capabilities": ["fault_history", "maintenance_analysis", "knowledge_retrieval"],
    "agent_id": "fault_investigator",
}

ADMIN_MISSION = {
    "mission_id": "system_admin",
    "name": "System Admin",
    "version": 1,
    "objective": (
        "Administer the agentic-runtime platform: inspect executions, "
        "missions and agents, check the health of platform dependencies, "
        "and manage missions and executions."
    ),
    "input_schema": {"type": "object", "properties": {}},
    "output_schema": {"type": "object", "properties": {}},
    "execution": {"mode": "autonomous", "agent_id": "admin_bot"},
    "capabilities": ["system_admin"],
    "agent_id": "admin_bot",
}

NEUTRAL_MISSION = {
    "mission_id": "general",
    "name": "General",
    "version": 1,
    "objective": "Do general tasks.",
    "input_schema": {"type": "object", "properties": {}},
    "output_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}, "evidence": {"type": "array"}},
    },
    "execution": {"mode": "autonomous", "agent_id": "general"},
    "capabilities": [],
    "agent_id": "general",
}

ALL_SKILL_NAMES = {"fault-investigation", "admin-operations", "report-writing", "data-access"}


# --- Loader ---
def test_loader_lists_seeded_skills():
    skills = load_skills()
    assert {s.name for s in skills} == ALL_SKILL_NAMES
    for skill in skills:
        assert len(skill.description) > 20
        assert skill.keywords, f"{skill.name} has no keywords"
        assert "When to Use" in skill.body
        assert skill.raw.startswith("---\n")


def test_loader_skips_invalid_skill_files(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "SKILL.md").write_text(
        "---\nname: good\ndescription: Good skill\nkeywords: good\n---\n\n# Good\n",
        encoding="utf-8",
    )
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text("# No frontmatter here\n", encoding="utf-8")
    (tmp_path / "empty").mkdir()
    assert [s.name for s in load_skills(tmp_path)] == ["good"]


def test_loader_missing_directory_loads_nothing(tmp_path):
    assert load_skills(tmp_path / "does-not-exist") == []


# --- Selection ---
def test_select_explicit_declaration_wins():
    mission = dict(FAULT_MISSION)
    mission["execution"] = {**FAULT_MISSION["execution"], "skills": ["report-writing"]}
    selected = select_skills(
        mission, {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"}
    )
    assert [s.name for s in selected] == ["report-writing"]


def test_select_explicit_unknown_name_skipped():
    mission = dict(FAULT_MISSION)
    mission["execution"] = {**FAULT_MISSION["execution"], "skills": ["nope", "report-writing"]}
    assert [s.name for s in select_skills(mission, {})] == ["report-writing"]


def test_select_fault_request_matches_fault_investigation():
    selected = select_skills(
        FAULT_MISSION,
        {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"},
    )
    assert [s.name for s in selected] == ["fault-investigation"]


def test_select_admin_request_matches_admin_operations():
    selected = select_skills(ADMIN_MISSION, {"request": "execution nào lỗi hôm nay?"})
    assert [s.name for s in selected] == ["admin-operations"]


def test_select_unrelated_request_matches_nothing():
    assert select_skills(NEUTRAL_MISSION, {"request": "what is the weather today?"}) == []


def test_skill_files_shape():
    files = skill_files_for_run(load_skills())
    assert set(files) == {f"/skills/{name}/SKILL.md" for name in ALL_SKILL_NAMES}
    for path, data in files.items():
        assert data["encoding"] == "utf-8"
        assert data["content"].startswith("---\n")
        assert f"name: {path.split('/')[2]}" in data["content"]


def test_seed_yaml_declares_skills():
    fault = load_missions.load_mission_file(BASE / "missions" / "fault_analysis.yaml")
    assert fault["execution_config"]["skills"] == [
        "fault-investigation",
        "data-access",
        "report-writing",
    ]
    admin = load_missions.load_mission_file(BASE / "missions" / "system_admin.yaml")
    assert admin["execution_config"]["skills"] == ["admin-operations", "report-writing"]


# --- Wiring: run_deep_agent passes files through ---
class _RecordingAgent:
    def __init__(self):
        self.seen_inputs: list = []
        self.seen_configs: list = []

    def invoke(self, state, config=None):
        self.seen_inputs.append(state)
        self.seen_configs.append(config)
        return {"messages": [SimpleNamespace(content="ok")]}


def test_run_deep_agent_passes_skill_files():
    agent = _RecordingAgent()
    files = skill_files_for_run(load_skills())
    run_deep_agent(agent, FAULT_MISSION, {"asset_id": "O21"}, files=files)
    assert agent.seen_inputs[0]["files"] == files
    agent2 = _RecordingAgent()
    run_deep_agent(agent2, FAULT_MISSION, {"asset_id": "O21"})
    assert "files" not in agent2.seen_inputs[0]


# --- Wiring: deep_agent_node selects + hands over skills ---
def _fake_outcome():
    return {
        "summary": "Done.",
        "evidence": [
            {"source": "deep_agent", "kind": "summary", "summary": "Done.", "stub": False}
        ],
        "actions_taken": [],
        "errors": [],
    }


def test_deep_agent_node_injects_declared_skills(db_session, monkeypatch):
    load_missions.main()
    mission = mission_service.get_mission(db_session, "system_admin")
    mission_dict = mission_service.mission_to_dict(mission)
    seen: dict = {}

    def fake_build(tools, system_prompt=None, skills=None, model_config=None,
                    middleware=None):
        seen["skills"] = skills
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None):
        seen["files"] = files
        return _fake_outcome()

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)

    final = run_mission(mission_dict, {"request": "execution nào lỗi hôm nay?"}, "exec-skill-admin")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    # system_admin's skills have no maps: no plan trace is produced.
    assert "plan_trace" not in final["result"]
    assert seen["skills"] == ["/skills/"]
    assert set(seen["files"]) == {
        "/skills/admin-operations/SKILL.md",
        "/skills/report-writing/SKILL.md",
    }
    assert "admin-operations" in seen["files"]["/skills/admin-operations/SKILL.md"]["content"]


def test_deep_agent_node_no_skill_when_no_match(monkeypatch):
    seen: dict = {}

    def fake_build(tools, system_prompt=None, skills=None, model_config=None,
                    middleware=None):
        seen["skills"] = skills
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None):
        seen["files"] = files
        return _fake_outcome()

    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)

    final = run_mission(
        NEUTRAL_MISSION, {"request": "what is the weather today?"}, "exec-skill-none"
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert seen["skills"] is None
    assert seen["files"] is None


# --- Native mechanism: the real deepagents SkillsMiddleware discloses ---
class _RecordingFakeModel(GenericFakeChatModel):
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.extend(messages)
        return super()._generate(messages, stop, run_manager, **kwargs)


def _message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


def test_native_skills_disclosed_to_model(monkeypatch):
    """End-to-end proof of the native mechanism: a real create_deep_agent
    with skills=["/skills/"] and the skill file handed over via invoke
    shows the skill index to the model in the system prompt."""
    model = _RecordingFakeModel(messages=iter([AIMessage(content="done")] * 8))
    monkeypatch.setattr(deep_agent_module, "get_chat_model", lambda: model)
    skill = next(s for s in load_skills() if s.name == "fault-investigation")
    agent = build_deep_agent([], system_prompt="test prompt", skills=["/skills/"])
    agent.invoke(
        {"messages": [{"role": "user", "content": "analyze faults in O21"}],
         "files": skill_files_for_run([skill])}
    )
    assert any("fault-investigation" in _message_text(m) for m in model.seen), (
        "skill index never reached the model"
    )
