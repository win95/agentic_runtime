"""Mission service — CRUD over Mission config rows.

A Mission is a versioned configuration document. The service stores the
latest version per mission id; `resolve_mission` is what the execution path
uses (it snapshots the version into the Execution row).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app import models


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
    mission = db.get(models.Mission, mission_id)
    if mission is None:
        return False
    db.delete(mission)
    db.commit()
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
