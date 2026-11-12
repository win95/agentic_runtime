"""Mission + agent loader — upsert YAML missions and agents into PostgreSQL.

Usage:
    python -m seeds.load_missions [--agent ID] [--mission ID]

Per-file isolation: a broken file is skipped (logged + printed) and
the rest still seed; the failures are returned, and the CLI exits 1
when any file failed. A filter narrows the run to the matching file
only (by file stem, e.g. --agent hotel_receptionist); schedules seed
only in an unfiltered run.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re

import yaml

from app import models
from app.config import settings
from app.core.db import SessionLocal, init_db
from app.services import schedule_service
from app.services.mission_service import upsert_mission

logger = logging.getLogger(__name__)
BASE = pathlib.Path(__file__).resolve().parent

_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env_placeholders(value, defaults=None):
    """Recursively expand ${VAR} / ${VAR:-fallback} in seed YAML values.

    Resolution order: os.environ, then the inline ``:-fallback``
    (shell semantics), then ``defaults`` (seed-time settings values
    such as LLM_MODEL). A placeholder with no value anywhere raises
    ValueError — a literal ${...} must never reach the database as a
    model name.
    """
    defaults = defaults or {}
    if isinstance(value, str):

        def _replace(match: re.Match) -> str:
            name, fallback = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if fallback is not None:
                return fallback
            if name in defaults:
                return defaults[name]
            raise ValueError(
                f"Seed placeholder ${{{name}}} has no value: "
                f"set {name} in .env or give it a default"
            )

        return _ENV_PLACEHOLDER_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: expand_env_placeholders(v, defaults) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_placeholders(v, defaults) for v in value]
    return value


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
    loader upserts directly with db.merge (insert-or-update by id).

    The model follows the environment at seed time: the YAML writes
    ``model: ${LLM_MODEL}`` and this expands it (falling back to
    settings.llm_model when the variable is unset). A model change in
    .env + re-seed creates a new AgentVersion (Option C).
    """
    data = expand_env_placeholders(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        defaults={"LLM_MODEL": settings.llm_model},
    )
    from runtime.session_history import parse_session_history_config

    return {
        "id": data["id"],
        "name": data.get("name", data["id"]),
        "provider": data.get("provider", "openai"),
        "model": data.get("model") or settings.llm_model,
        "system_prompt": data.get("system_prompt", ""),
        "temperature": float(data.get("temperature", 0.0)),
        "allowed_tools": data.get("allowed_tools") or [],
        # Per-agent session-history limits (docs/sessions.md): validated
        # here — a bad block fails the seed loudly, never silently.
        "session_history": parse_session_history_config(
            data.get("session_history"), source=f"agent {data['id']}"
        ),
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


def seed_schedules(db) -> tuple:
    """Upsert schedules from seeds/schedules/*.yaml.

    A re-seed never flips a schedule the user enabled: the stored
    ``enabled`` value wins over the YAML one. Per-file isolation like
    ``main()``: a broken file is skipped. Returns
    (seeded_count, [(path, error), ...]).
    """
    failures: list = []
    seeded = 0
    schedules_dir = BASE / "schedules"
    if not schedules_dir.is_dir():
        return seeded, failures
    for path in sorted(schedules_dir.glob("*.yaml")):
        try:
            data = load_schedule_file(path)
            existing = schedule_service.get_schedule(db, data["id"])
            if existing is not None:
                data["enabled"] = existing.enabled
            schedule = schedule_service.upsert_schedule(db, data)
            seeded += 1
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
        except Exception as exc:  # noqa: BLE001 — one bad file skips only itself
            db.rollback()
            failures.append((str(path), str(exc)))
            logger.warning("Skipping schedule file %s: %s", path, exc)
            print(f"FAILED schedule {path.name}: {exc}")
    return seeded, failures


def _select_files(directory: pathlib.Path, wanted: str | None,
                  kind: str) -> list:
    """Files to seed: all *.yaml, or the one whose stem matches `wanted`."""
    paths = sorted(directory.glob("*.yaml")) if directory.is_dir() else []
    if wanted is None:
        return paths
    for path in paths:
        if path.stem == wanted:
            return [path]
    available = ", ".join(p.stem for p in paths) or "(none)"
    raise ValueError(f"Unknown {kind} {wanted!r}; available: {available}")


def main(agent: str | None = None, mission: str | None = None) -> list:
    """Upsert missions, agents and schedules from YAML.

    Per-file isolation: a broken file (bad YAML, missing placeholder,
    invalid session_history, ...) is skipped and recorded; the rest
    still seed. Returns the failures as [(path, error), ...] — the
    caller decides whether to fail. The CLI exits 1 on any failure.
    A filter narrows the run to the matching file only (by file stem);
    schedules seed only in an unfiltered run.
    """
    init_db()
    db = SessionLocal()
    failures: list = []
    counts = {"missions": 0, "agents": 0, "schedules": 0}
    try:
        # A filter narrows the run to the matching file only: --agent
        # skips missions, --mission skips agents; both together seed
        # the two files.
        if agent is None:
            for path in _select_files(BASE / "missions", mission, "mission"):
                try:
                    row = upsert_mission(db, load_mission_file(path))
                    counts["missions"] += 1
                    logger.info("Seeded mission %s (v%s)", row.id, row.version)
                    print(f"Seeded mission {row.id} (v{row.version})")
                except Exception as exc:  # noqa: BLE001 — one bad file skips itself
                    db.rollback()
                    failures.append((str(path), str(exc)))
                    logger.warning("Skipping mission file %s: %s", path, exc)
                    print(f"FAILED mission {path.name}: {exc}")
        if mission is None:
            agents_dir = BASE / "agents"
            if agents_dir.is_dir():
                from app.services import agent_version_service

                for path in _select_files(agents_dir, agent, "agent"):
                    try:
                        row = db.merge(models.Agent(**load_agent_file(path)))
                        db.commit()
                        counts["agents"] += 1
                        logger.info("Seeded agent %s", row.id)
                        print(f"Seeded agent {row.id}")
                        # Option C: snapshot the agent's config as a version
                        # when it changed (or when no version exists yet).
                        # Re-seeding an unchanged agent creates nothing.
                        version, created = (
                            agent_version_service.ensure_version_for_agent(
                                db, row.id, source="seed"
                            )
                        )
                        if created:
                            print(
                                f"  agent version v{version.version} created (active)"
                            )
                        else:
                            print(
                                f"  agent version v{version.version} already active"
                            )
                    except Exception as exc:  # noqa: BLE001 — one bad file skips itself
                        db.rollback()
                        failures.append((str(path), str(exc)))
                        logger.warning("Skipping agent file %s: %s", path, exc)
                        print(f"FAILED agent {path.name}: {exc}")
        if agent is None and mission is None:
            seeded_schedules, schedule_failures = seed_schedules(db)
            counts["schedules"] += seeded_schedules
            failures.extend(schedule_failures)
    finally:
        db.close()
    print(
        f"Seed done: {counts['missions']} missions, {counts['agents']} agents, "
        f"{counts['schedules']} schedules, {len(failures)} failed"
    )
    for path, error in failures:
        print(f"  FAILED {path}: {error}")
    return failures


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    _parser = argparse.ArgumentParser(
        description="Seed missions/agents/schedules from YAML")
    _parser.add_argument(
        "--agent", default=None,
        help="seed only this agent (file stem, e.g. hotel_receptionist)")
    _parser.add_argument(
        "--mission", default=None,
        help="seed only this mission (file stem, e.g. hotel_reception)")
    _args = _parser.parse_args()
    _failures = main(agent=_args.agent, mission=_args.mission)
    if _failures:
        sys.exit(1)
