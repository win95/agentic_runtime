"""Schedule service — recurring mission runs (cron).

A Schedule says "run this mission with this input on this cron". The
dispatcher (``run_due_schedules`` below, driven every minute by the
Celery beat task ``app.worker.tasks.run_due_schedules``) creates one
Execution per due schedule via ``execution_service.create_execution``.

Double-dispatch guard: for each due schedule the service advances
``next_run_at`` and commits that advance BEFORE dispatching, so a
concurrent or overlapping beat tick cannot dispatch the same schedule
twice.

A dispatch failure does NOT lose the run: ``next_run_at`` is restored
to the missed time and the failure recorded in ``last_error``, so the
next beat tick retries it. (Previously the schedule still advanced on
failure, silently dropping that run.) A permanently broken schedule
retries once a minute — disable it to stop the retries. There is no
catch-up: a successful dispatch always jumps ``next_run_at`` to the
next occurrence *after now*, so a long-disabled schedule fires once
when re-enabled instead of replaying every missed run.

Cron times are naive UTC, like every other datetime in this codebase
(07:00 UTC is 14:00 in Asia/Saigon).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from croniter import CroniterBadCronError, croniter
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import models
from app.services import execution_service, mission_service

logger = logging.getLogger(__name__)


class InvalidCronError(ValueError):
    """The cron expression is not a valid 5-field cron."""


class ScheduleMissionNotFoundError(ValueError):
    """The schedule's mission does not exist."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def validate_cron(cron: str) -> None:
    """Reject anything that is not a valid 5-field cron expression."""
    fields = (cron or "").split()
    if len(fields) != 5:
        raise InvalidCronError(
            f"Cron expression must have exactly 5 fields "
            f"(got {len(fields)}): {cron!r}"
        )
    try:
        croniter(cron)
    except (CroniterBadCronError, ValueError) as exc:
        raise InvalidCronError(f"Invalid cron expression {cron!r}: {exc}") from exc


def compute_next_run(cron: str, now: datetime) -> datetime:
    """Next cron occurrence strictly after ``now`` (naive UTC)."""
    return croniter(cron, now).get_next(datetime)


def new_schedule_id() -> str:
    return f"sched_{uuid.uuid4().hex[:12]}"


def schedule_to_dict(schedule: models.Schedule) -> dict:
    return {
        "id": schedule.id,
        "mission_id": schedule.mission_id,
        "cron": schedule.cron,
        "input_data": schedule.input_data or {},
        "enabled": schedule.enabled,
        "tenant": schedule.tenant,
        "next_run_at": schedule.next_run_at,
        "last_run_at": schedule.last_run_at,
        "last_error": schedule.last_error,
        "created_at": schedule.created_at,
        "updated_at": schedule.updated_at,
    }


def create_schedule(db: Session, data: dict) -> models.Schedule:
    """Create a schedule. Validates the cron and computes the first run.

    Raises InvalidCronError on a bad expression, and
    ScheduleMissionNotFoundError when the mission does not exist (the
    foreign key enforces this on PostgreSQL; on SQLite, where foreign
    keys are not enforced, the mission check below is what raises).
    """
    validate_cron(data["cron"])
    if mission_service.resolve_mission(db, data["mission_id"]) is None:
        raise ScheduleMissionNotFoundError(
            f"Mission {data['mission_id']} not found"
        )
    now = _utcnow()
    schedule = models.Schedule(
        id=data.get("id") or new_schedule_id(),
        mission_id=data["mission_id"],
        cron=data["cron"],
        input_data=data.get("input_data") or {},
        enabled=data.get("enabled", True),
        tenant=data.get("tenant", "default"),
        next_run_at=compute_next_run(data["cron"], now),
    )
    db.add(schedule)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        if mission_service.resolve_mission(db, data["mission_id"]) is None:
            raise ScheduleMissionNotFoundError(
                f"Mission {data['mission_id']} not found"
            ) from None
        raise
    db.refresh(schedule)
    return schedule


def upsert_schedule(db: Session, data: dict) -> models.Schedule:
    """Insert or replace a schedule (used by seed loading).

    ``next_run_at`` is recomputed only when the cron changed, so a
    re-seed does not disturb the current run cadence.
    """
    existing = db.get(models.Schedule, data["id"]) if "id" in data else None
    if existing is None:
        return create_schedule(db, data)
    cron_changed = data.get("cron") is not None and data["cron"] != existing.cron
    if cron_changed:
        validate_cron(data["cron"])
    for key, value in data.items():
        setattr(existing, key, value)
    if cron_changed:
        existing.next_run_at = compute_next_run(existing.cron, _utcnow())
    db.commit()
    db.refresh(existing)
    return existing


def list_schedules(db: Session) -> list[models.Schedule]:
    return db.query(models.Schedule).order_by(models.Schedule.id).all()


def get_schedule(db: Session, schedule_id: str) -> models.Schedule | None:
    return db.get(models.Schedule, schedule_id)


def update_schedule(db: Session, schedule_id: str, fields: dict) -> models.Schedule | None:
    """Partial update. A cron change recomputes ``next_run_at`` from now;
    (re-)enabling a stale schedule does the same. Returns None when the
    schedule does not exist."""
    schedule = db.get(models.Schedule, schedule_id)
    if schedule is None:
        return None
    was_enabled = schedule.enabled
    now = _utcnow()
    if fields.get("cron") is not None:
        validate_cron(fields["cron"])
    for key, value in fields.items():
        if value is not None:
            setattr(schedule, key, value)
    if fields.get("cron") is not None:
        schedule.next_run_at = compute_next_run(schedule.cron, now)
    elif fields.get("enabled") is True and not was_enabled:
        if schedule.next_run_at is None or schedule.next_run_at <= now:
            schedule.next_run_at = compute_next_run(schedule.cron, now)
    db.commit()
    db.refresh(schedule)
    return schedule


def set_enabled(db: Session, schedule_id: str, enabled: bool) -> models.Schedule | None:
    return update_schedule(db, schedule_id, {"enabled": enabled})


def delete_schedule(db: Session, schedule_id: str) -> bool:
    schedule = db.get(models.Schedule, schedule_id)
    if schedule is None:
        return False
    db.delete(schedule)
    db.commit()
    return True


def run_due_schedules(db: Session, now: datetime | None = None) -> dict:
    """Dispatch one execution per due schedule. Returns a summary dict.

    Due = enabled, ``next_run_at`` set and in the past. Each due schedule
    is advanced (and that advance committed) BEFORE its execution is
    created, so overlapping beat ticks cannot double-dispatch. A failure
    to dispatch restores ``next_run_at`` to the missed time — the run is
    retried on the next beat tick, not lost — and records ``last_error``.
    """
    now = now or _utcnow()
    due = (
        db.query(models.Schedule)
        .filter(
            models.Schedule.enabled.is_(True),
            models.Schedule.next_run_at.is_not(None),
            models.Schedule.next_run_at <= now,
        )
        .order_by(models.Schedule.next_run_at)
        .all()
    )
    dispatched: list[str] = []
    failed: list[dict] = []
    for schedule in due:
        previous_next_run = schedule.next_run_at
        try:
            schedule.next_run_at = compute_next_run(schedule.cron, now)
        except (CroniterBadCronError, ValueError) as exc:
            # The cron became invalid after creation (hand-edited row):
            # record it and move on; no execution is created.
            schedule.last_error = f"Invalid cron {schedule.cron!r}: {exc}"
            db.commit()
            failed.append({"schedule_id": schedule.id, "error": schedule.last_error})
            logger.warning("Schedule %s has invalid cron, skipping: %s", schedule.id, exc)
            continue
        db.commit()  # advance committed BEFORE dispatch (double-dispatch guard)
        try:
            execution, problems = execution_service.create_execution(
                db, schedule.mission_id, schedule.input_data or {}
            )
        except Exception as exc:  # noqa: BLE001 — dispatch failure is a result
            schedule.next_run_at = previous_next_run  # keep the run due: retry next tick
            schedule.last_error = f"dispatch failed: {exc}"[:2000]
            db.commit()
            failed.append({"schedule_id": schedule.id, "error": schedule.last_error})
            logger.warning("Schedule %s dispatch failed: %s", schedule.id, exc)
            continue
        if execution is None:
            schedule.next_run_at = previous_next_run  # keep the run due: retry next tick
            if problems == ["__mission_not_found__"]:
                schedule.last_error = f"Mission {schedule.mission_id} not found"
            else:
                schedule.last_error = f"Invalid execution input: {problems}"
            db.commit()
            failed.append({"schedule_id": schedule.id, "error": schedule.last_error})
            continue
        schedule.last_run_at = now
        schedule.last_error = None
        db.commit()
        dispatched.append(execution.id)
    return {"due": len(due), "dispatched": dispatched, "failed": failed}
