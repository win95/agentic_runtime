"""Mission service — CRUD over Mission config rows.

A Mission is a versioned configuration document. The service stores the
latest version per mission id; `resolve_mission` is what the execution path
uses (it snapshots the version into the Execution row).
"""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import models


class MissionAlreadyExistsError(ValueError):
    """A mission with this id already exists. The API maps this to 409
    Conflict; use update (upsert) to change it."""

    def __init__(self, mission_id: str):
        super().__init__(f"Mission {mission_id} already exists")
        self.mission_id = mission_id


class MissionHasExecutionsError(Exception):
    """A mission with executions cannot be deleted (would break the
    executions_mission_id_fkey foreign key and destroy run history).

    The API maps this to 409 Conflict; callers should delete or archive
    the executions first if the mission really must go.
    """

    def __init__(self, mission_id: str, execution_count: int):
        super().__init__(
            f"Mission {mission_id} has {execution_count} execution(s)"
        )
        self.mission_id = mission_id
        self.execution_count = execution_count


class MissionHasSchedulesError(Exception):
    """A mission with schedules cannot be deleted (would break the
    schedules_mission_id_fkey foreign key). The API maps this to 409
    Conflict; delete the schedules first if the mission really must go.
    """

    def __init__(self, mission_id: str, schedule_count: int):
        super().__init__(
            f"Mission {mission_id} has {schedule_count} schedule(s)"
        )
        self.mission_id = mission_id
        self.schedule_count = schedule_count


def create_mission(db: Session, data: dict) -> models.Mission:
    """Insert a mission. Raises MissionAlreadyExistsError (409) when the
    id is taken — the pre-check gives the friendly error and the
    IntegrityError catch below is the safety net for a concurrent
    insert between the check and the commit."""
    if db.get(models.Mission, data["id"]) is not None:
        raise MissionAlreadyExistsError(data["id"])
    mission = models.Mission(**data)
    db.add(mission)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise MissionAlreadyExistsError(data["id"]) from None
    db.refresh(mission)
    return mission


def upsert_mission(db: Session, data: dict) -> models.Mission:
    """Insert or replace the mission with the same id (used by seed loading)."""
    existing = db.get(models.Mission, data["id"])
    if existing is None:
        return create_mission(db, data)
    for key, value in data.items():
        setattr(existing, key, value)
    db.commit()
    db.refresh(existing)
    return existing


def resolve_mission(db: Session, mission_id: str) -> models.Mission | None:
    """Return the current version of a mission, or None if unknown."""
    return db.get(models.Mission, mission_id)


def list_missions(db: Session) -> list[models.Mission]:
    return db.query(models.Mission).order_by(models.Mission.id).all()


def get_mission(db: Session, mission_id: str) -> models.Mission | None:
    return db.get(models.Mission, mission_id)


def delete_mission(db: Session, mission_id: str) -> bool:
    """Delete a mission. Returns False when it does not exist.

    Raises MissionHasExecutionsError (mapped to 409 by the router) when
    the mission still has executions, and MissionHasSchedulesError when
    it still has schedules (same 409 mapping). The count checks run
    first so the failure is deterministic on every backend (SQLite does
    not enforce foreign keys by default); the IntegrityError catch below
    is the safety net for an execution or schedule created between the
    check and the commit.
    """
    mission = db.get(models.Mission, mission_id)
    if mission is None:
        return False
    execution_count = (
        db.query(models.Execution)
        .filter(models.Execution.mission_id == mission_id)
        .count()
    )
    if execution_count > 0:
        raise MissionHasExecutionsError(mission_id, execution_count)
    schedule_count = (
        db.query(models.Schedule)
        .filter(models.Schedule.mission_id == mission_id)
        .count()
    )
    if schedule_count > 0:
        raise MissionHasSchedulesError(mission_id, schedule_count)
    db.delete(mission)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        count = (
            db.query(models.Execution)
            .filter(models.Execution.mission_id == mission_id)
            .count()
        )
        if count > 0:
            raise MissionHasExecutionsError(mission_id, count) from None
        scount = (
            db.query(models.Schedule)
            .filter(models.Schedule.mission_id == mission_id)
            .count()
        )
        if scount > 0:
            raise MissionHasSchedulesError(mission_id, scount) from None
        raise
    return True


def mission_to_dict(mission: models.Mission) -> dict:
    """Plain-dict snapshot of a mission (stored on the Execution row and
    passed to the LangGraph runtime — the graph never reads the DB for the
    mission definition itself)."""
    return {
        "mission_id": mission.id,
        "name": mission.name,
        "version": mission.version,
        "objective": mission.objective,
        "input_schema": mission.input_schema or {},
        "output_schema": mission.output_schema or {},
        "execution": mission.execution_config or {},
        "capabilities": mission.capabilities or [],
        "agent_id": mission.agent_id,
    }
