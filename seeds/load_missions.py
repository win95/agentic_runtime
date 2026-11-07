"""Seed loader — upsert YAML missions from seeds/missions into PostgreSQL.

Usage:
    python -m seeds.load_missions
"""

from __future__ import annotations

import logging
import pathlib

import yaml

from app.core.db import SessionLocal, init_db
from app.services.mission_service import upsert_mission

logger = logging.getLogger(__name__)
BASE = pathlib.Path(__file__).resolve().parent


def load_mission_file(path: pathlib.Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    execution = data.get("execution") or {}
    return {
        "id": data["mission_id"],
        "name": data.get("name", data["mission_id"]),
        "version": int(data.get("version", 1)),
        "objective": data.get("objective", ""),
        "input_schema": data.get("input_schema") or {},
        "output_schema": data.get("output_schema") or {},
        "execution_config": execution,
        "capabilities": data.get("capabilities") or [],
        "agent_id": data.get("agent_id") or execution.get("agent_id"),
    }


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        for path in sorted((BASE / "missions").glob("*.yaml")):
            mission = upsert_mission(db, load_mission_file(path))
            logger.info("Seeded mission %s (v%s)", mission.id, mission.version)
            print(f"Seeded mission {mission.id} (v{mission.version})")
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
