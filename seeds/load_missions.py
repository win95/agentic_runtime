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
from app.services import schedule_service
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


def load_schedule_file(path: pathlib.Path) -> dict:
    """Schedule row (models.Schedule)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "id": data["id"],
        "mission_id": data["mission_id"],
        "cron": data["cron"],
        "input_data": data.get("input_data") or {},
        "enabled": bool(data.get("enabled", True)),
        "tenant": data.get("tenant", "default"),
    }


def seed_schedules(db) -> None:
    """Upsert schedules from seeds/schedules/*.yaml.

    A re-seed never flips a schedule the user enabled: the stored
    ``enabled`` value wins over the YAML one.
    """
    schedules_dir = BASE / "schedules"
    if not schedules_dir.is_dir():
        return
    for path in sorted(schedules_dir.glob("*.yaml")):
        data = load_schedule_file(path)
        existing = schedule_service.get_schedule(db, data["id"])
        if existing is not None:
            data["enabled"] = existing.enabled
        schedule = schedule_service.upsert_schedule(db, data)
        logger.info(
            "Seeded schedule %s (cron=%s, enabled=%s)",
            schedule.id,
            schedule.cron,
            schedule.enabled,
        )
        print(
            f"Seeded schedule {schedule.id} "
            f"(cron={schedule.cron}, enabled={schedule.enabled})"
        )


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
            from app.services import agent_version_service

            for path in sorted(agents_dir.glob("*.yaml")):
                agent = db.merge(models.Agent(**load_agent_file(path)))
                db.commit()
                logger.info("Seeded agent %s", agent.id)
                print(f"Seeded agent {agent.id}")
                # Option C: snapshot the agent's config as a version
                # when it changed (or when no version exists yet).
                # Re-seeding an unchanged agent creates nothing.
                version, created = agent_version_service.ensure_version_for_agent(
                    db, agent.id, source="seed"
                )
                if created:
                    print(f"  agent version v{version.version} created (active)")
                else:
                    print(f"  agent version v{version.version} already active")
        seed_schedules(db)
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
