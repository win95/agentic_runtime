"""Human approval for destructive tools — Option C (docs/hooks.md).

Hook tests call ``request_permission`` directly; router tests go
through the API. The hook opens its own SessionLocal, so tests read
rows it created through the ``db_session`` fixture (``expire_all``
first — different session).
"""

from __future__ import annotations

from datetime import datetime

from app import models
from app.services import execution_service, permission_service
from runtime.hooks import HookContext
from runtime.hooks.builtin import request_permission
from runtime.notifications import slack as slack_notify

ADMIN_CTX = {
    "execution_id": "exec-test",
    "mission_id": "system_admin",
    "agent_id": "admin_bot",
}


def _ctx(tool_name: str, tool_input: dict) -> HookContext:
    return HookContext(tool_name=tool_name, tool_input=tool_input, **ADMIN_CTX)


def _pending_rows(db_session):
    db_session.expire_all()
    return (
        db_session.query(models.PermissionRequest)
        .order_by(models.PermissionRequest.created_at.asc())
        .all()
    )


def _seed_mission(db_session, mission_id: str = "system_admin"):
    mission = models.Mission(
        id=mission_id,
        name="System Admin",
        objective="test",
        input_schema={
            "type": "object",
            "properties": {"request": {"type": "string"}},
            "required": ["request"],
        },
        output_schema={"type": "object", "properties": {}},
        execution_config={"mode": "autonomous", "agent_id": "admin_bot"},
        capabilities=["system_admin"],
        agent_id="admin_bot",
    )
    db_session.add(mission)
    db_session.commit()
    return mission


# --- Hook: deny + create -------------------------------------------------


def test_first_call_denies_and_creates_pending(db_session):
    decision = request_permission(
        _ctx("delete_mission", {"mission_id": "fault_analysis"})
    )
    assert decision is not None
    assert decision.action == "deny"
    assert "request_permission" in decision.message
    rows = _pending_rows(db_session)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "pending"
    assert row.tool_name == "delete_mission"
    assert row.target == "fault_analysis"
    assert row.tool_input == {"mission_id": "fault_analysis"}
    assert row.id in decision.message


def test_second_call_reuses_pending_request(db_session):
    first = request_permission(_ctx("delete_mission", {"mission_id": "m1"}))
    second = request_permission(_ctx("delete_mission", {"mission_id": "m1"}))
    rows = _pending_rows(db_session)
    assert len(rows) == 1
    assert rows[0].id in first.message
    assert rows[0].id in second.message
    assert "already pending approval" in second.message


def test_approve_then_allow_single_use(db_session):
    request_permission(_ctx("delete_mission", {"mission_id": "m1"}))
    row = _pending_rows(db_session)[0]
    assert permission_service.approve_request(db_session, row.id) is not None

    assert request_permission(_ctx("delete_mission", {"mission_id": "m1"})) is None
    db_session.expire_all()
    assert db_session.get(models.PermissionRequest, row.id).status == "used"

    # Approval is single-use: the next call asks again.
    decision = request_permission(_ctx("delete_mission", {"mission_id": "m1"}))
    assert decision is not None and decision.action == "deny"
    assert len(_pending_rows(db_session)) == 2


def test_cancel_execution_same_flow(db_session):
    assert request_permission(
        _ctx("cancel_execution", {"execution_id": "exec-9"})
    ) is not None
    row = _pending_rows(db_session)[0]
    assert row.tool_name == "cancel_execution"
    assert row.target == "exec-9"
    permission_service.approve_request(db_session, row.id)
    assert request_permission(_ctx("cancel_execution", {"execution_id": "exec-9"})) is None


def test_non_destructive_tool_ignored(db_session):
    assert request_permission(_ctx("get_missions", {})) is None
    assert request_permission(_ctx("system_health", {})) is None
    assert _pending_rows(db_session) == []


def test_missing_target_ignored(db_session):
    assert request_permission(_ctx("delete_mission", {})) is None
    assert request_permission(_ctx("delete_mission", {"mission_id": ""})) is None
    assert _pending_rows(db_session) == []


def test_expired_pending_creates_fresh_request(db_session):
    request_permission(_ctx("delete_mission", {"mission_id": "m1"}))
    old = _pending_rows(db_session)[0]
    old.expires_at = datetime(2000, 1, 1)
    db_session.commit()

    decision = request_permission(_ctx("delete_mission", {"mission_id": "m1"}))
    rows = _pending_rows(db_session)
    assert len(rows) == 2
    assert rows[0].status == "expired"
    assert rows[1].status == "pending"
    assert rows[1].id in decision.message


def test_expired_approval_denies_with_fresh_request(db_session):
    request_permission(_ctx("delete_mission", {"mission_id": "m1"}))
    row = _pending_rows(db_session)[0]
    permission_service.approve_request(db_session, row.id)
    row.expires_at = datetime(2000, 1, 1)
    db_session.commit()

    decision = request_permission(_ctx("delete_mission", {"mission_id": "m1"}))
    assert decision is not None and decision.action == "deny"
    rows = _pending_rows(db_session)
    assert len(rows) == 2
    assert rows[0].status == "expired"
    assert rows[1].status == "pending"


def test_db_error_denies_fail_closed(db_session, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(permission_service, "check_permission", _boom)
    decision = request_permission(_ctx("delete_mission", {"mission_id": "m1"}))
    assert decision is not None
    assert decision.action == "deny"
    assert "could not verify human approval" in decision.message
    assert _pending_rows(db_session) == []


# --- Slack ping ----------------------------------------------------------


def test_permission_slack_message_has_approve_path():
    payload = slack_notify.build_permission_message(
        request_id="perm_abc",
        tool_name="delete_mission",
        target="fault_analysis",
        execution_id="exec-1",
        mission_id="system_admin",
    )
    assert "perm_abc" in payload["text"]
    assert "POST /permissions/perm_abc/approve" in payload["text"]
    assert "delete_mission(fault_analysis)" in payload["text"]


def test_permission_slack_notify_no_webhook_is_noop():
    assert slack_notify.notify_permission_request(
        request_id="perm_abc",
        tool_name="delete_mission",
        target="m1",
        webhook_url="",
    ) is False


# --- Router: list / get / approve / deny --------------------------------


def _make_pending(db_session, monkeypatch, tool_input=None):
    """A pending request tied to a real QUEUED execution of system_admin."""
    _seed_mission(db_session)
    calls: list[str] = []
    monkeypatch.setattr(
        execution_service, "dispatch_execution", lambda eid: calls.append(eid)
    )
    execution, problems = execution_service.create_execution(
        db_session,
        "system_admin",
        {"request": "delete mission fault_analysis"},
        session_id="sess_perm1",
    )
    assert execution is not None, problems
    decision = request_permission(
        HookContext(
            execution_id=execution.id,
            mission_id="system_admin",
            agent_id="admin_bot",
            tool_name="delete_mission",
            tool_input=tool_input or {"mission_id": "fault_analysis"},
        )
    )
    assert decision is not None
    return execution, _pending_rows(db_session)[0]


def test_list_and_get_pending(client, db_session, monkeypatch):
    _, row = _make_pending(db_session, monkeypatch)
    response = client.get("/permissions")
    assert response.status_code == 200
    assert [item["request_id"] for item in response.json()] == [row.id]

    response = client.get(f"/permissions/{row.id}")
    assert response.status_code == 200
    assert response.json()["target"] == "fault_analysis"
    assert response.json()["status"] == "pending"

    assert client.get("/permissions/perm_missing").status_code == 404


def test_approve_redispatches_original_execution(client, db_session, monkeypatch):
    execution, row = _make_pending(db_session, monkeypatch)
    response = client.post(f"/permissions/{row.id}/approve", json={"note": "ok"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert body["decision_note"] == "ok"
    redispatched_id = body["redispatched_execution_id"]
    assert redispatched_id is not None and redispatched_id != execution.id

    db_session.expire_all()
    new_execution = execution_service.get_execution(db_session, redispatched_id)
    assert new_execution is not None
    assert new_execution.mission_id == "system_admin"
    assert new_execution.input == {"request": "delete mission fault_analysis"}
    assert new_execution.session_id == "sess_perm1"
    assert new_execution.status == "QUEUED"

    # The re-run now claims the approval: the hook lets the call through.
    assert request_permission(
        HookContext(
            execution_id=redispatched_id,
            mission_id="system_admin",
            agent_id="admin_bot",
            tool_name="delete_mission",
            tool_input={"mission_id": "fault_analysis"},
        )
    ) is None


def test_approve_twice_is_conflict(client, db_session, monkeypatch):
    _, row = _make_pending(db_session, monkeypatch)
    assert client.post(f"/permissions/{row.id}/approve", json={}).status_code == 200
    assert client.post(f"/permissions/{row.id}/approve", json={}).status_code == 409
    assert client.post("/permissions/perm_missing/approve", json={}).status_code == 404


def test_deny_no_redispatch_then_approve_conflicts(client, db_session, monkeypatch):
    _, row = _make_pending(db_session, monkeypatch)
    response = client.post(f"/permissions/{row.id}/deny", json={"note": "no"})
    assert response.status_code == 200
    assert response.json()["status"] == "denied"
    assert response.json()["redispatched_execution_id"] is None
    assert client.post(f"/permissions/{row.id}/approve", json={}).status_code == 409
    assert client.post("/permissions/perm_missing/deny", json={}).status_code == 404
