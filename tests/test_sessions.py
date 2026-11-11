"""Session tests — Option B (docs/sessions.md).

Covers: auto-generated vs client-supplied session ids, format
validation (422), agent/mission binding (422), session grouping +
GET /sessions, history loading + caps, and history injection into
the Deep Agent's system prompt (recording fake model, following
tests/test_skill_maps.py). No LLM, no live services.
"""

from datetime import datetime, timedelta, timezone

import pytest

import runtime.deep_agent as deep_agent_module
import seeds.load_missions as load_missions
from app import models
from app.services import execution_service, mission_service, session_service
from runtime.context_budget import estimate_tokens
from runtime.mission_graph import run_mission
from runtime.session_history import (
    DEFAULT_LIMITS,
    MAX_SESSION_TURNS,
    SESSION_HISTORY_TOKENS,
    SessionHistoryLimits,
    cap_turns,
    parse_session_history_config,
    render_session_history,
    resolve_session_limits,
)

FAULT_INPUT = {
    "asset_id": "O21",
    "start_date": "2026-10-01",
    "end_date": "2026-10-31",
}
ADMIN_INPUT = {"request": "List the missions"}


def _seed():
    load_missions.main()


def _mock_dispatch(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        execution_service, "dispatch_execution", lambda execution_id: calls.append(execution_id)
    )
    return calls


def _post_execution(client, mission_id, input_data, session_id=None):
    payload = {"mission_id": mission_id, "input": input_data}
    if session_id is not None:
        payload["session_id"] = session_id
    return client.post("/executions", json=payload)


# --- API: id generation / validation ---
def test_create_auto_generates_session_id(client, monkeypatch):
    _seed()
    _mock_dispatch(monkeypatch)
    response = _post_execution(client, "fault_analysis", FAULT_INPUT)
    assert response.status_code == 202
    body = response.json()
    assert body["session_id"].startswith("sess_")
    assert len(body["session_id"]) == len("sess_") + 12


def test_create_client_supplied_session_id_accepted(client, monkeypatch):
    _seed()
    _mock_dispatch(monkeypatch)
    response = _post_execution(
        client, "fault_analysis", FAULT_INPUT, session_id="sess-client-1"
    )
    assert response.status_code == 202
    assert response.json()["session_id"] == "sess-client-1"


def test_create_invalid_session_id_422(client, monkeypatch):
    _seed()
    _mock_dispatch(monkeypatch)
    # Schema-level: space + bang fail the ^[A-Za-z0-9_-]{1,64}$ pattern.
    response = _post_execution(client, "fault_analysis", FAULT_INPUT, session_id="bad id!")
    assert response.status_code == 422
    # Service-level (bypasses the schema): same rule — covered by
    # test_create_invalid_session_id_service_level below.


def test_create_invalid_session_id_service_level(db_session):
    _seed()
    mission = mission_service.get_mission(db_session, "fault_analysis")
    assert mission is not None
    execution, problems = execution_service.create_execution(
        db_session, "fault_analysis", dict(FAULT_INPUT), session_id="bad id!"
    )
    assert execution is None
    assert problems == ["field 'session_id' must match ^[A-Za-z0-9_-]{1,64}$"]


# --- API: binding ---
def test_same_session_different_agent_or_mission_422(client, monkeypatch):
    _seed()
    _mock_dispatch(monkeypatch)
    first = _post_execution(
        client, "fault_analysis", FAULT_INPUT, session_id="sess-bind-1"
    )
    assert first.status_code == 202
    # Same session, different mission (and agent): rejected.
    response = _post_execution(
        client, "system_admin", ADMIN_INPUT, session_id="sess-bind-1"
    )
    assert response.status_code == 422
    assert "session 'sess-bind-1' belongs to mission 'fault_analysis'" in response.text
    # Same session, same mission: accepted.
    second = _post_execution(
        client, "fault_analysis", FAULT_INPUT, session_id="sess-bind-1"
    )
    assert second.status_code == 202


# --- API: session grouping + GET /sessions ---
def test_get_session_summary_and_executions(client, monkeypatch):
    _seed()
    _mock_dispatch(monkeypatch)
    for _ in range(2):
        assert (
            _post_execution(
                client, "fault_analysis", FAULT_INPUT, session_id="sess-group-1"
            ).status_code
            == 202
        )
    response = client.get("/sessions/sess-group-1")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "sess-group-1"
    assert body["agent_id"] == "fault_investigator"
    assert body["mission_id"] == "fault_analysis"
    assert body["execution_count"] == 2
    assert len(body["executions"]) == 2
    # Oldest first (conversation order).
    created = [e["created_at"] for e in body["executions"]]
    assert created == sorted(created)
    assert all(e["session_id"] == "sess-group-1" for e in body["executions"])


def test_get_session_unknown_404_and_malformed_422(client):
    assert client.get("/sessions/sess-no-such").status_code == 404
    assert client.get("/sessions/bad%20id!").status_code == 422


def test_list_executions_session_filter(client, monkeypatch):
    _seed()
    _mock_dispatch(monkeypatch)
    _post_execution(client, "fault_analysis", FAULT_INPUT, session_id="sess-f-1")
    _post_execution(client, "fault_analysis", FAULT_INPUT, session_id="sess-f-2")
    _post_execution(client, "fault_analysis", FAULT_INPUT, session_id="sess-f-1")
    response = client.get("/executions?session_id=sess-f-1")
    assert response.status_code == 200
    ids = [e["session_id"] for e in response.json()]
    assert ids == ["sess-f-1", "sess-f-1"]
    assert client.get("/executions?session_id=sess-nope").json() == []


# --- Service: history loading ---
def _terminal_execution(db_session, execution_id, session_id, answer=None,
                        status="SUCCEEDED", created_at=None, mission_id="fault_analysis",
                        agent_id="fault_investigator", error=None, output=None):
    execution = models.Execution(
        id=execution_id,
        mission_id=mission_id,
        mission_version=1,
        agent_id=agent_id,
        session_id=session_id,
        input=dict(FAULT_INPUT),
        status=status,
        error=error,
        created_at=created_at or datetime.now(timezone.utc).replace(tzinfo=None),
    )
    db_session.add(execution)
    if output is None:
        output = {}
        if answer is not None:
            output = {"root_causes": [answer], "recommendations": ["REC-" + answer]}
    db_session.add(
        models.ExecutionResult(
            id=f"res-{execution_id}",
            execution_id=execution_id,
            output=output,
            evidence=[{"kind": "fault", "summary": "F-204 motor overheating"}],
        )
    )
    db_session.commit()
    return execution


def test_load_session_history_only_terminal_and_excludes_current(db_session):
    _seed()
    base = datetime(2026, 11, 11, 5, 0, 0)
    _terminal_execution(db_session, "exec-h1", "sess-h", answer="CAUSE-ONE",
                        created_at=base)
    _terminal_execution(db_session, "exec-h2", "sess-h", answer="CAUSE-TWO",
                        status="FAILED", error="boom", created_at=base + timedelta(seconds=1))
    # QUEUED run: no result yet, must not become a turn.
    db_session.add(models.Execution(
        id="exec-h3", mission_id="fault_analysis", mission_version=1,
        agent_id="fault_investigator", session_id="sess-h",
        input=dict(FAULT_INPUT), status="QUEUED",
        created_at=base + timedelta(seconds=2),
    ))
    db_session.commit()
    history = session_service.load_session_history(db_session, "sess-h", "exec-current")
    assert history["total_prior"] == 2
    assert history["omitted"] == 0
    turns = history["turns"]
    assert [t["execution_id"] for t in turns] == ["exec-h1", "exec-h2"]
    assert "CAUSE-ONE" in turns[0]["summary"]
    assert turns[0]["evidence_count"] == 1
    assert turns[1]["status"] == "FAILED"
    assert turns[1]["error"] == "boom"
    # The current execution itself is excluded even if terminal.
    history = session_service.load_session_history(db_session, "sess-h", "exec-h1")
    assert [t["execution_id"] for t in history["turns"]] == ["exec-h2"]


def test_load_session_history_empty_session(db_session):
    history = session_service.load_session_history(db_session, "sess-empty", "exec-x")
    assert history == {"turns": [], "omitted": 0, "total_prior": 0}


# --- Caps ---
def test_cap_turns_drops_oldest_and_marks_omitted():
    turns = [
        {"execution_id": f"exec-{i}", "status": "SUCCEEDED", "created_at": None,
         "input": "{}", "summary": f"turn {i}", "evidence_count": 0, "error": None}
        for i in range(MAX_SESSION_TURNS + 2)
    ]
    kept, omitted = cap_turns(turns)
    assert len(kept) == MAX_SESSION_TURNS
    assert omitted == 2
    assert kept[0]["execution_id"] == "exec-2"  # newest kept
    assert kept[-1]["execution_id"] == f"exec-{MAX_SESSION_TURNS + 1}"
    rendered = render_session_history(kept, omitted=omitted, session_id="sess-cap")
    assert "[Earlier turns omitted: 2]" in rendered
    assert estimate_tokens(rendered) <= SESSION_HISTORY_TOKENS


def test_cap_turns_token_budget_trims_further():
    # Near-max turns: 1,000-char input + 1,500-char answer each, so
    # ~640 tokens per turn — 10 turns would be ~6,400 > 4,096 and the
    # token cap must drop more after the turn cap already dropped 2.
    long_answer = "x" * 1500
    long_input = "y" * 1000
    turns = [
        {"execution_id": f"exec-{i}", "status": "SUCCEEDED", "created_at": None,
         "input": long_input, "summary": long_answer, "evidence_count": 0, "error": None}
        for i in range(MAX_SESSION_TURNS + 2)
    ]
    kept, omitted = cap_turns(turns)
    assert len(kept) < MAX_SESSION_TURNS  # token cap bites after the turn cap
    assert omitted > 2
    rendered = render_session_history(kept, omitted=omitted)
    assert estimate_tokens(rendered) <= SESSION_HISTORY_TOKENS
    assert f"[Earlier turns omitted: {omitted}]" in rendered
    # The newest turn always survives.
    assert kept[-1]["execution_id"] == f"exec-{MAX_SESSION_TURNS + 1}"


def test_render_empty_history_is_empty():
    assert render_session_history([], omitted=0) == ""


# --- Graph: injection into the Deep Agent's system prompt ---
def _run_with_recording_fake(db_session, monkeypatch, mission_dict, history, session_id,
                             input_data=None):
    seen: dict = {}

    def fake_build(tools, system_prompt=None, skills=None, model_config=None,
                   middleware=None):
        seen["system_prompt"] = system_prompt
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None,
                 skill_maps=None):
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
        dict(input_data) if input_data is not None else dict(FAULT_INPUT),
        "exec-second-1",
        session_history=history,
        session_id=session_id,
    )
    return final, seen


def test_history_injection_prior_answer_reaches_system_prompt(db_session, monkeypatch):
    _seed()
    _terminal_execution(db_session, "exec-first-1", "sess-inject",
                        answer="MOTOR-OVERHEAT-XYZ",
                        created_at=datetime(2026, 11, 11, 5, 0, 0))
    history = session_service.load_session_history(
        db_session, "sess-inject", "exec-second-1"
    )
    assert len(history["turns"]) == 1
    mission = mission_service.get_mission(db_session, "fault_analysis")
    mission_dict = mission_service.mission_to_dict(mission)
    # Autonomous mode guarantees deep_agent_node runs.
    mission_dict["execution"]["mode"] = "autonomous"
    final, seen = _run_with_recording_fake(
        db_session, monkeypatch, mission_dict, history, "sess-inject"
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    system_prompt = seen["system_prompt"] or ""
    assert "MOTOR-OVERHEAT-XYZ" in system_prompt
    assert "REC-MOTOR-OVERHEAT-XYZ" in system_prompt
    assert "Session history: 1 prior turn(s)" in system_prompt
    assert "<data source=\"session_history\"" in system_prompt
    assert any("Session history: 1 prior turn(s)" in n for n in final["notes"])


def test_fresh_session_no_history_block(db_session, monkeypatch):
    _seed()
    mission = mission_service.get_mission(db_session, "fault_analysis")
    mission_dict = mission_service.mission_to_dict(mission)
    mission_dict["execution"]["mode"] = "autonomous"
    final, seen = _run_with_recording_fake(
        db_session, monkeypatch, mission_dict, None, None
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    # No history block was rendered: the prompt paragraph mentions
    # session history in general, but no prior-turn block is present.
    assert "<data source=\"session_history\"" not in (seen["system_prompt"] or "")
    assert "prior turn(s)" not in (seen["system_prompt"] or "")
    assert not any("Session history:" in n for n in final["notes"])


def test_history_injection_pinned_version_prompt(db_session, monkeypatch):
    """The history block also reaches the agent when an AgentVersion
    pins the prompt (the version branch, not just the default one)."""
    _seed()
    _terminal_execution(db_session, "exec-first-2", "sess-inject-v",
                        answer="MOTOR-OVERHEAT-XYZ",
                        created_at=datetime(2026, 11, 11, 5, 0, 0))
    history = session_service.load_session_history(
        db_session, "sess-inject-v", "exec-second-2"
    )
    mission = mission_service.get_mission(db_session, "fault_analysis")
    mission_dict = mission_service.mission_to_dict(mission)
    mission_dict["execution"]["mode"] = "autonomous"
    # Seed an active agent version so deep_agent_node takes the
    # pinned-prompt branch.
    from app.services import agent_version_service

    agent_version_service.create_version(
        db_session,
        "fault_investigator",
        source="test",
        prompt_text="PINNED-PROMPT-MARKER",
    )
    final, seen = _run_with_recording_fake(
        db_session, monkeypatch, mission_dict, history, "sess-inject-v"
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    system_prompt = seen["system_prompt"] or ""
    assert "PINNED-PROMPT-MARKER" in system_prompt
    assert "MOTOR-OVERHEAT-XYZ" in system_prompt


# --- Per-agent limits (docs/sessions.md) ---
def test_parse_session_history_config():
    assert parse_session_history_config(None) is None
    assert parse_session_history_config({}) == {
        "max_turns": MAX_SESSION_TURNS,
        "max_tokens": SESSION_HISTORY_TOKENS,
    }
    assert parse_session_history_config({"max_turns": 3}) == {
        "max_turns": 3,
        "max_tokens": SESSION_HISTORY_TOKENS,
    }
    assert parse_session_history_config({"max_turns": 3, "max_tokens": 2048}) == {
        "max_turns": 3,
        "max_tokens": 2048,
    }
    for bad in (
        {"max_turns": 0},
        {"max_turns": -1},
        {"max_tokens": 100},
        {"max_turns": "3"},
        {"max_tokens": True},
        {"max_turns": 3, "max_context": 100},
        {"max_turns": 3.5},
        "10",
        [3],
    ):
        with pytest.raises(ValueError):
            parse_session_history_config(bad, source="agent test")


def test_seed_agent_session_history_and_prompt():
    base = load_missions.BASE / "agents"
    fault = load_missions.load_agent_file(base / "fault_investigator.yaml")
    assert fault["session_history"] == {"max_turns": 3, "max_tokens": 2048}
    assert "Session history:" in fault["system_prompt"]
    assert "Do not re-derive or repeat prior work without need" in fault["system_prompt"]
    admin = load_missions.load_agent_file(base / "admin_bot.yaml")
    assert admin["session_history"] == {"max_turns": 10, "max_tokens": 4096}
    assert "Session history:" in admin["system_prompt"]
    assert "Do not repeat completed work without need" in admin["system_prompt"]


def _short_turns(count):
    return [
        {"execution_id": f"exec-{i}", "status": "SUCCEEDED", "created_at": None,
         "input": "{}", "summary": f"turn {i}", "evidence_count": 0, "error": None}
        for i in range(count)
    ]


def test_cap_turns_per_agent_limits():
    turns = _short_turns(12)
    fault_limits = SessionHistoryLimits(max_turns=3, max_tokens=2048)
    kept, omitted = cap_turns(turns, limits=fault_limits)
    assert len(kept) == 3
    assert omitted == 9
    assert [t["execution_id"] for t in kept] == ["exec-9", "exec-10", "exec-11"]
    rendered = render_session_history(
        kept, omitted=omitted, session_id="sess-lim",
        limits=fault_limits, agent_id="fault_investigator",
    )
    assert "limited to 3 turn(s) / 2,048 tokens for agent fault_investigator" in rendered
    assert "[Earlier turns omitted: 9]" in rendered
    # Same history under the admin limits: 10 kept, 2 omitted.
    admin_limits = SessionHistoryLimits(max_turns=10, max_tokens=4096)
    kept, omitted = cap_turns(turns, limits=admin_limits)
    assert len(kept) == 10
    assert omitted == 2
    rendered = render_session_history(
        kept, omitted=omitted, session_id="sess-lim",
        limits=admin_limits, agent_id="admin_bot",
    )
    assert "limited to 10 turn(s) / 4,096 tokens for agent admin_bot" in rendered
    # No limits given: the global defaults apply (same as before).
    kept, omitted = cap_turns(turns)
    assert len(kept) == MAX_SESSION_TURNS
    assert omitted == 2


def test_resolve_session_limits_precedence():
    version = {"max_turns": 2, "max_tokens": 1024}
    agent = {"max_turns": 5, "max_tokens": 3000}
    assert resolve_session_limits(version_limits=version, agent_limits=agent) == (
        SessionHistoryLimits(max_turns=2, max_tokens=1024)
    )
    assert resolve_session_limits(agent_limits=agent) == (
        SessionHistoryLimits(max_turns=5, max_tokens=3000)
    )
    assert resolve_session_limits() == DEFAULT_LIMITS
    # An invalid stored value falls back instead of breaking the run.
    assert resolve_session_limits(version_limits={"max_turns": 0}) == DEFAULT_LIMITS
    assert resolve_session_limits(version_limits={"max_turns": "3"}) == DEFAULT_LIMITS
    assert resolve_session_limits(
        version_limits={"max_turns": 0}, agent_limits=agent
    ) == SessionHistoryLimits(max_turns=5, max_tokens=3000)


def _five_prior_turns(db_session, session_id, mission_id, agent_id, base, id_prefix="exec-prior"):
    for i in range(5):
        output = (
            {"answer": f"ADMIN-ANSWER-{i}"}
            if mission_id == "system_admin"
            else None
        )
        answer = None if output is not None else f"CAUSE-{i}"
        _terminal_execution(
            db_session, f"{id_prefix}-{i}", session_id,
            answer=answer, output=output,
            created_at=base + timedelta(seconds=i),
            mission_id=mission_id, agent_id=agent_id,
        )


def test_history_injection_uses_agent_limits(db_session, monkeypatch):
    _seed()
    base = datetime(2026, 11, 11, 5, 0, 0)
    # fault_investigator: 3 turns / 2,048 tokens — 5 priors, newest 3 shown.
    _five_prior_turns(db_session, "sess-lim-f", "fault_analysis", "fault_investigator", base,
                      id_prefix="exec-prior-f")
    history = session_service.load_session_history(db_session, "sess-lim-f", "exec-second-1")
    assert len(history["turns"]) == 5  # load-time cap is the global one
    mission = mission_service.get_mission(db_session, "fault_analysis")
    mission_dict = mission_service.mission_to_dict(mission)
    mission_dict["execution"]["mode"] = "autonomous"
    final, seen = _run_with_recording_fake(
        db_session, monkeypatch, mission_dict, history, "sess-lim-f"
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    system_prompt = seen["system_prompt"] or ""
    for i in (2, 3, 4):
        assert f"exec-prior-f-{i}" in system_prompt
    for i in (0, 1):
        assert f"exec-prior-f-{i}" not in system_prompt
    assert "limited to 3 turn(s) / 2,048 tokens for agent fault_investigator" in system_prompt
    assert any("Session history: 3 prior turn(s) (2 earlier omitted)" in n for n in final["notes"])
    # admin_bot: 10 turns / 4,096 tokens — all 5 priors shown.
    _five_prior_turns(db_session, "sess-lim-a", "system_admin", "admin_bot", base,
                      id_prefix="exec-prior-a")
    history = session_service.load_session_history(db_session, "sess-lim-a", "exec-second-1")
    mission = mission_service.get_mission(db_session, "system_admin")
    mission_dict = mission_service.mission_to_dict(mission)
    final, seen = _run_with_recording_fake(
        db_session, monkeypatch, mission_dict, history, "sess-lim-a",
        input_data=dict(ADMIN_INPUT),
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    system_prompt = seen["system_prompt"] or ""
    for i in range(5):
        assert f"exec-prior-a-{i}" in system_prompt
    assert "limited to 10 turn(s) / 4,096 tokens for agent admin_bot" in system_prompt
    assert any("Session history: 5 prior turn(s)" in n for n in final["notes"])


def test_version_snapshot_carries_session_limits(db_session):
    _seed()
    from app.services import agent_version_service

    fault_version, created = agent_version_service.create_version(
        db_session, "fault_investigator", source="test"
    )
    assert created is False  # seed already snapshotted the same config
    assert fault_version.session_history == {"max_turns": 3, "max_tokens": 2048}
    admin_version, _ = agent_version_service.create_version(
        db_session, "admin_bot", source="test"
    )
    assert admin_version.session_history == {"max_turns": 10, "max_tokens": 4096}
    # Changing the limits on the live row creates a new version on
    # re-seed (limits are part of the snapshot equality check).
    agent = db_session.get(models.Agent, "fault_investigator")
    agent.session_history = {"max_turns": 5, "max_tokens": 3000}
    db_session.commit()
    new_version, created = agent_version_service.ensure_version_for_agent(
        db_session, "fault_investigator", source="test"
    )
    assert created is True
    assert new_version.version == fault_version.version + 1
    assert new_version.session_history == {"max_turns": 5, "max_tokens": 3000}
    # Re-seeding the unchanged config creates nothing.
    same_version, created = agent_version_service.ensure_version_for_agent(
        db_session, "fault_investigator", source="test"
    )
    assert created is False
    assert same_version.version == new_version.version
