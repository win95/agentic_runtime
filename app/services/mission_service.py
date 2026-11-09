"""Mission service — CRUD over Mission config rows.

A Mission is a versioned configuration document. The service stores the
latest version per mission id; `resolve_mission` is what the execution path
uses (it snapshots the version into the Execution row).
"""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import models


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


def create_mission(db: Session, data: dict) -> models.Mission:
    mission = models.Mission(**data)
    db.add(mission)
    db.commit()
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
    the mission still has executions. The count check runs first so the
    failure is deterministic on every backend (SQLite does not enforce
    foreign keys by default); the IntegrityError catch below is the
    safety net for an execution created between the check and the commit.
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
        raise MissionHasExecutionsError(mission_id, count) from None
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
