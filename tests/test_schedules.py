"""Schedule tests — cron validation, dispatcher, router, seed, delete guard.

Tests run against in-memory SQLite (see conftest); Celery dispatch is
monkeypatched per test, so no broker is needed.
"""

import pathlib
from datetime import datetime, timezone

import pytest

from app import models
from app.services import execution_service, mission_service, schedule_service
from seeds.load_missions import load_schedule_file, seed_schedules

MISSION_PAYLOAD = {
    "id": "fault_analysis",
    "name": "Fault Analysis",
    "version": 1,
    "objective": "Analyze equipment faults.",
    "input_schema": {"type": "object", "required": ["asset_id"]},
    "output_schema": {"type": "object", "properties": {"root_causes": {"type": "array"}}},
    "execution_config": {"mode": "deterministic", "agent_id": "fault_investigator"},
    "capabilities": ["fault_history"],
    "agent_id": "fault_investigator",
}

SEED_YAML = (
    pathlib.Path(__file__).resolve().parent.parent
    / "seeds"
    / "schedules"
    / "system_admin_daily_health.yaml"
)

NOW = datetime(2026, 11, 9, 7, 0)  # a Monday 07:00, naive UTC


def _make_mission(db_session, payload=None):
    return mission_service.create_mission(db_session, dict(payload or MISSION_PAYLOAD))


def _make_schedule(db_session, **overrides):
    data = {
        "mission_id": "fault_analysis",
        "cron": "0 7 * * *",
        "input_data": {"asset_id": "O21"},
    }
    data.update(overrides)
    return schedule_service.create_schedule(db_session, data)


def _patch_dispatch(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        execution_service, "dispatch_execution", lambda eid: calls.append(eid)
    )
    return calls


# --- Cron validation ---


@pytest.mark.parametrize(
    "bad",
    ["not a cron", "0 25 * * *", "*/5 * *", "0 7 * *", "", "@daily", "0 7 * * * *"],
)
def test_validate_cron_rejects_invalid(bad):
    with pytest.raises(schedule_service.InvalidCronError):
        schedule_service.validate_cron(bad)


@pytest.mark.parametrize("good", ["0 7 * * *", "*/5 * * * *", "0 0 1 * *", "30 2 * * 1-5"])
def test_validate_cron_accepts_valid(good):
    schedule_service.validate_cron(good)  # no raise


def test_compute_next_run():
    assert schedule_service.compute_next_run(
        "0 7 * * *", datetime(2026, 11, 9, 6, 0)
    ) == datetime(2026, 11, 9, 7, 0)
    assert schedule_service.compute_next_run(
        "0 7 * * *", datetime(2026, 11, 9, 8, 0)
    ) == datetime(2026, 11, 10, 7, 0)


def test_create_schedule_computes_next_run(db_session):
    _make_mission(db_session)
    schedule = _make_schedule(db_session)
    assert schedule.id.startswith("sched_")
    assert schedule.next_run_at is not None
    assert schedule.next_run_at.hour == 7 and schedule.next_run_at.minute == 0
    assert schedule.next_run_at > schedule.created_at
    assert schedule.enabled is True
    assert schedule.tenant == "default"


def test_create_schedule_invalid_cron(db_session):
    _make_mission(db_session)
    with pytest.raises(schedule_service.InvalidCronError):
        _make_schedule(db_session, cron="0 25 * * *")


def test_create_schedule_unknown_mission(db_session):
    with pytest.raises(schedule_service.ScheduleMissionNotFoundError):
        _make_schedule(db_session, mission_id="nope")


# --- Dispatcher ---


def test_dispatcher_dispatches_due_schedule(db_session, monkeypatch):
    calls = _patch_dispatch(monkeypatch)
    _make_mission(db_session)
    schedule = _make_schedule(db_session)
    schedule.next_run_at = datetime(2026, 11, 9, 6, 30)
    db_session.commit()

    summary = schedule_service.run_due_schedules(db_session, now=NOW)

    assert summary["due"] == 1
    assert len(summary["dispatched"]) == 1
    assert calls == summary["dispatched"]
    assert summary["failed"] == []
    executions = db_session.query(models.Execution).all()
    assert len(executions) == 1
    assert executions[0].status == "QUEUED"
    assert executions[0].mission_id == "fault_analysis"
    assert executions[0].input == {"asset_id": "O21"}
    db_session.refresh(schedule)
    assert schedule.next_run_at == datetime(2026, 11, 10, 7, 0)
    assert schedule.last_run_at == NOW
    assert schedule.last_error is None


def test_dispatcher_no_double_dispatch(db_session, monkeypatch):
    calls = _patch_dispatch(monkeypatch)
    _make_mission(db_session)
    schedule = _make_schedule(db_session)
    schedule.next_run_at = datetime(2026, 11, 9, 6, 30)
    db_session.commit()

    first = schedule_service.run_due_schedules(db_session, now=NOW)
    second = schedule_service.run_due_schedules(db_session, now=NOW)

    assert len(first["dispatched"]) == 1
    assert second == {"due": 0, "dispatched": [], "failed": []}
    assert len(calls) == 1
    assert db_session.query(models.Execution).count() == 1


def test_dispatcher_skips_disabled(db_session, monkeypatch):
    _patch_dispatch(monkeypatch)
    _make_mission(db_session)
    schedule = _make_schedule(db_session, enabled=False)
    schedule.next_run_at = datetime(2026, 11, 9, 6, 30)
    db_session.commit()

    summary = schedule_service.run_due_schedules(db_session, now=NOW)

    assert summary == {"due": 0, "dispatched": [], "failed": []}
    assert db_session.query(models.Execution).count() == 0
    db_session.refresh(schedule)
    assert schedule.next_run_at == datetime(2026, 11, 9, 6, 30)


def test_dispatcher_skips_not_yet_due(db_session, monkeypatch):
    _patch_dispatch(monkeypatch)
    _make_mission(db_session)
    schedule = _make_schedule(db_session)
    schedule.next_run_at = datetime(2026, 11, 9, 8, 0)
    db_session.commit()

    summary = schedule_service.run_due_schedules(db_session, now=NOW)

    assert summary == {"due": 0, "dispatched": [], "failed": []}
    assert db_session.query(models.Execution).count() == 0


def test_dispatcher_dispatch_failure_keeps_due_and_records(db_session, monkeypatch):
    monkeypatch.setattr(
        execution_service,
        "dispatch_execution",
        lambda eid: (_ for _ in ()).throw(RuntimeError("broker down")),
    )
    _make_mission(db_session)
    schedule = _make_schedule(db_session)
    schedule.next_run_at = datetime(2026, 11, 9, 6, 30)
    db_session.commit()

    summary = schedule_service.run_due_schedules(db_session, now=NOW)

    assert summary["due"] == 1
    assert summary["dispatched"] == []
    assert len(summary["failed"]) == 1
    assert "broker down" in summary["failed"][0]["error"]
    # The execution row exists (marked FAILED by create_execution),
    # with finished_at set like every other terminal row ...
    executions = db_session.query(models.Execution).all()
    assert len(executions) == 1
    assert executions[0].status == "FAILED"
    assert executions[0].finished_at is not None
    # ... and the schedule stays due so the next beat tick retries.
    db_session.refresh(schedule)
    assert schedule.next_run_at == datetime(2026, 11, 9, 6, 30)
    assert schedule.last_error and "broker down" in schedule.last_error


def test_update_cron_recomputes_next_run(db_session):
    _make_mission(db_session)
    schedule = _make_schedule(db_session)
    updated = schedule_service.update_schedule(
        db_session, schedule.id, {"cron": "0 8 * * *"}
    )
    assert updated is not None
    assert updated.cron == "0 8 * * *"
    assert updated.next_run_at is not None
    assert updated.next_run_at.hour == 8 and updated.next_run_at.minute == 0
    # next_run_at is naive UTC (schedule_service._utcnow() convention),
    # so compare against naive UTC now, not naive local now — the
    # latter fails whenever the local zone is behind UTC (item 11).
    assert updated.next_run_at > datetime.now(timezone.utc).replace(tzinfo=None)


def test_enable_recomputes_stale_next_run(db_session):
    _make_mission(db_session)
    schedule = _make_schedule(db_session, enabled=False)
    schedule.next_run_at = datetime(2026, 1, 1, 7, 0)
    db_session.commit()
    updated = schedule_service.set_enabled(db_session, schedule.id, True)
    assert updated is not None
    assert updated.enabled is True
    assert updated.next_run_at is not None
    # next_run_at is naive UTC (schedule_service._utcnow() convention),
    # so compare against naive UTC now, not naive local now — the
    # latter fails whenever the local zone is behind UTC (item 11).
    assert updated.next_run_at > datetime.now(timezone.utc).replace(tzinfo=None)


# --- Router ---


def test_router_crud_and_enable_disable(client):
    assert client.post("/missions", json=MISSION_PAYLOAD).status_code == 201

    response = client.post(
        "/schedules",
        json={
            "mission_id": "fault_analysis",
            "cron": "0 7 * * *",
            "input_data": {"asset_id": "O21"},
        },
    )
    assert response.status_code == 201
    body = response.json()
    schedule_id = body["id"]
    assert schedule_id.startswith("sched_")
    assert body["next_run_at"] is not None
    assert body["enabled"] is True

    assert client.post(
        "/schedules",
        json={"mission_id": "fault_analysis", "cron": "0 25 * * *"},
    ).status_code == 400
    assert client.post(
        "/schedules",
        json={"mission_id": "nope", "cron": "0 7 * * *"},
    ).status_code == 400

    assert client.get("/schedules").status_code == 200
    assert len(client.get("/schedules").json()) == 1
    assert client.get(f"/schedules/{schedule_id}").status_code == 200
    assert client.get("/schedules/nope").status_code == 404

    response = client.patch(f"/schedules/{schedule_id}", json={"cron": "0 8 * * *"})
    assert response.status_code == 200
    assert response.json()["cron"] == "0 8 * * *"
    assert response.json()["next_run_at"].endswith("08:00:00")
    assert client.patch("/schedules/nope", json={"cron": "0 8 * * *"}).status_code == 404
    assert client.patch(
        f"/schedules/{schedule_id}", json={"cron": "bad"}
    ).status_code == 400

    response = client.post(f"/schedules/{schedule_id}/disable")
    assert response.status_code == 200
    assert response.json()["enabled"] is False
    response = client.post(f"/schedules/{schedule_id}/enable")
    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert client.post("/schedules/nope/enable").status_code == 404

    assert client.delete(f"/schedules/{schedule_id}").status_code == 204
    assert client.get(f"/schedules/{schedule_id}").status_code == 404
    assert client.delete("/schedules/nope").status_code == 404


def test_delete_mission_with_schedule_returns_409(client):
    assert client.post("/missions", json=MISSION_PAYLOAD).status_code == 201
    response = client.post(
        "/schedules",
        json={"mission_id": "fault_analysis", "cron": "0 7 * * *"},
    )
    assert response.status_code == 201
    schedule_id = response.json()["id"]

    response = client.delete("/missions/fault_analysis")
    assert response.status_code == 409
    assert "schedule(s)" in response.json()["detail"]

    assert client.delete(f"/schedules/{schedule_id}").status_code == 204
    assert client.delete("/missions/fault_analysis").status_code == 204


# --- Seed ---


def test_seed_loads_example_schedule_disabled(db_session):
    _make_mission(
        db_session,
        {
            "id": "system_admin",
            "name": "System Admin",
            "version": 1,
            "objective": "Administer the platform.",
            "input_schema": {"type": "object", "required": ["request"]},
            "output_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            "execution_config": {"mode": "autonomous", "agent_id": "admin_bot"},
            "capabilities": ["system_admin"],
            "agent_id": "admin_bot",
        },
    )
    data = load_schedule_file(SEED_YAML)
    assert data["id"] == "sched_system_admin_daily_health"
    assert data["mission_id"] == "system_admin"
    assert data["cron"] == "0 7 * * *"
    assert data["enabled"] is False
    assert "kiểm tra sức khỏe hệ thống" in data["input_data"]["request"]

    # The weekly fault_analysis schedule is seeded by the same glob —
    # its mission must exist too, or seed_schedules raises.
    _make_mission(db_session)
    seed_schedules(db_session)

    schedule = schedule_service.get_schedule(
        db_session, "sched_system_admin_daily_health"
    )
    assert schedule is not None
    assert schedule.enabled is False
    assert schedule.cron == "0 7 * * *"
    assert schedule.next_run_at is not None

    weekly = schedule_service.get_schedule(
        db_session, "sched_fault_analysis_weekly"
    )
    assert weekly is not None
    assert weekly.enabled is False
    assert weekly.cron == "0 6 * * 1"
    assert weekly.input_data["asset_id"] == "O21"


# --- Beat wiring ---


