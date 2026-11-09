"""Mission + agent loader — upsert YAML missions and agents into PostgreSQL.

Usage:
    python -m seeds.load_missions
"""

from __future__ import annotations

import logging
import pathlib

import yaml

from app import models
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


def load_agent_file(path: pathlib.Path) -> dict:
    """Agent config row (models.Agent). No agent service exists, so the
    loader upserts directly with db.merge (insert-or-update by id)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "id": data["id"],
        "name": data.get("name", data["id"]),
        "provider": data.get("provider", "openai"),
        "model": data.get("model", "gpt-4o-mini"),
        "system_prompt": data.get("system_prompt", ""),
        "temperature": float(data.get("temperature", 0.0)),
        "allowed_tools": data.get("allowed_tools") or [],
    }


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        for path in sorted((BASE / "missions").glob("*.yaml")):
            mission = upsert_mission(db, load_mission_file(path))
            logger.info("Seeded mission %s (v%s)", mission.id, mission.version)
            print(f"Seeded mission {mission.id} (v{mission.version})")
        agents_dir = BASE / "agents"
        if agents_dir.is_dir():
            for path in sorted(agents_dir.glob("*.yaml")):
                agent = db.merge(models.Agent(**load_agent_file(path)))
                db.commit()
                logger.info("Seeded agent %s", agent.id)
                print(f"Seeded agent {agent.id}")
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
