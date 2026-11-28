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
    from runtime.harness.loader import harness_dict_from_model, parse_mission_harness

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    execution = dict(data.get("execution") or {})
    harness_model = parse_mission_harness(
        data.get("harness"),
        source=str(path),
    )
    from runtime.harness.validation import validate_output_artifacts_policy

    validate_output_artifacts_policy(
        harness_model,
        data.get("output_schema") or {},
        source=str(path),
    )
    execution["harness"] = harness_dict_from_model(harness_model)
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


def _collect_file_schedule_ids() -> set[str]:
    """Ids claimed by seeds/schedules/*.yaml.

    Loops share one id namespace with file schedules (§3.2 of the
    loops design): a loop id colliding with a file schedule id must
    fail the seed loudly, never silently overwrite. A broken file is
    skipped here — seed_schedules reports it loudly on its own pass.
    """
    ids: set[str] = set()
    schedules_dir = BASE / "schedules"
    if not schedules_dir.is_dir():
        return ids
    for path in sorted(schedules_dir.glob("*.yaml")):
        try:
            ids.add(load_schedule_file(path)["id"])
        except Exception as exc:  # noqa: BLE001 — reported by seed_schedules
            logger.warning("Skipping schedule file for id scan %s: %s", path, exc)
    return ids


def _collect_loop_ids() -> set[str]:
    """Loop ids declared in seeds/missions/*.yaml harness blocks.

    The other half of the shared id namespace: a file schedule id
    colliding with a loop id must fail the seed loudly. A broken
    mission file is skipped here — its mission seed fails loudly on
    its own pass.
    """
    from runtime.harness.loader import parse_mission_harness

    ids: set[str] = set()
    missions_dir = BASE / "missions"
    if not missions_dir.is_dir():
        return ids
    for path in sorted(missions_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            harness = parse_mission_harness(data.get("harness"), source=str(path))
            ids.update(loop.id for loop in harness.loops)
        except Exception as exc:  # noqa: BLE001 — reported by the mission pass
            logger.warning("Skipping mission file for loop scan %s: %s", path, exc)
    return ids


def seed_mission_loops(
    db,
    mission_row,
    harness_dict,
    *,
    file_schedule_ids: frozenset[str] | set[str],
    source: str,
) -> tuple:
    """Materialize one mission's harness loops into Schedule rows.

    Each loop becomes one row: ``id`` = loop id, ``mission_id``,
    ``cron``, ``input_data`` = loop input, ``enabled`` (stored value
    wins on re-seed, same rule as ``seed_schedules``), ``tenant``
    ``default`` (missions carry no tenant). ``next_run_at`` is
    recomputed only when the cron changes — ``upsert_schedule`` owns
    that rule, it is not reimplemented here.

    Per-loop isolation like ``seed_schedules`` per file: a loop that
    fails (bad cron, input failing the mission's input validation,
    id collision) is recorded and skipped; the mission's other loops
    still seed. Returns (seeded_count, [(source, error), ...]).
    """
    from app.services import execution_service
    from runtime.harness.schema import MissionHarness
    from runtime.security import sanitize_input_value

    failures: list = []
    seeded = 0
    try:
        harness = MissionHarness.model_validate(harness_dict or {})
    except Exception as exc:  # noqa: BLE001 — seed already validated it
        failures.append((source, f"invalid harness loops: {exc}"))
        return seeded, failures
    for loop in harness.loops:
        loop_source = f"{source} loop {loop.id}"
        try:
            if loop.id in file_schedule_ids:
                raise ValueError(
                    f"loop id {loop.id!r} collides with a file schedule id "
                    "in seeds/schedules/*.yaml (shared id namespace)"
                )
            schedule_service.validate_cron(loop.cron)
            # Same validator create_execution uses, on the sanitized
            # input exactly as create_execution sees it — a loop whose
            # input would fail must fail the seed, not every beat tick.
            problems = execution_service.validate_execution_input(
                mission_row, sanitize_input_value(loop.input)
            )
            if problems:
                raise ValueError(
                    f"loop {loop.id!r} input fails mission input "
                    f"validation: {problems}"
                )
            existing = schedule_service.get_schedule(db, loop.id)
            if existing is not None and existing.mission_id != mission_row.id:
                raise ValueError(
                    f"loop id {loop.id!r} collides with existing schedule "
                    f"for mission {existing.mission_id!r}"
                )
            data = {
                "id": loop.id,
                "mission_id": mission_row.id,
                "cron": loop.cron,
                "input_data": loop.input,
                "enabled": loop.enabled,
                "tenant": "default",
            }
            if existing is not None:
                # Stored enabled wins (same rule as seed_schedules):
                # a loop enabled via the API stays enabled on re-seed.
                data["enabled"] = existing.enabled
            schedule = schedule_service.upsert_schedule(db, data)
            seeded += 1
            logger.info(
                "Seeded loop schedule %s (mission=%s, cron=%s, enabled=%s)",
                schedule.id,
                schedule.mission_id,
                schedule.cron,
                schedule.enabled,
            )
            print(
                f"Seeded loop schedule {schedule.id} "
                f"(mission={schedule.mission_id}, cron={schedule.cron}, "
                f"enabled={schedule.enabled})"
            )
        except Exception as exc:  # noqa: BLE001 — one bad loop skips itself
            db.rollback()
            failures.append((loop_source, str(exc)))
            logger.warning("Skipping harness loop %s: %s", loop.id, exc)
            print(f"FAILED loop {loop.id}: {exc}")
    return seeded, failures


def seed_schedules(db, loop_ids: frozenset[str] | None = None) -> tuple:
    """Upsert schedules from seeds/schedules/*.yaml.

    A re-seed never flips a schedule the user enabled: the stored
    ``enabled`` value wins over the YAML one. Per-file isolation like
    ``main()``: a broken file is skipped. A file schedule id that
    collides with a harness loop id fails loudly (shared id namespace
    — a loop and a file schedule must never silently overwrite each
    other). Returns (seeded_count, [(path, error), ...]).
    """
    failures: list = []
    seeded = 0
    if loop_ids is None:
        loop_ids = frozenset(_collect_loop_ids())
    schedules_dir = BASE / "schedules"
    if not schedules_dir.is_dir():
        return seeded, failures
    for path in sorted(schedules_dir.glob("*.yaml")):
        try:
            data = load_schedule_file(path)
            if data["id"] in loop_ids:
                raise ValueError(
                    f"schedule id {data['id']!r} collides with a harness "
                    "loop id (shared id namespace — rename one of them)"
                )
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


def validate_all_mission_harness_policies() -> None:
    """Validate harness across every mission seed file (fail fast on conflict)."""
    import yaml

    from runtime.harness.loader import parse_mission_harness
    from runtime.harness.validation import (
        validate_cross_mission_harness,
        validate_output_artifacts_policy,
    )

    items: list = []
    for path in _select_files(BASE / "missions", None, "mission"):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not data.get("mission_id"):
                continue
            harness = parse_mission_harness(data.get("harness"), source=str(path))
            validate_output_artifacts_policy(
                harness,
                data.get("output_schema") or {},
                source=str(path),
            )
            items.append((data["mission_id"], harness))
        except Exception as exc:  # noqa: BLE001 — broken files skip validation row
            logger.warning("Harness policy skip %s: %s", path, exc)
            continue
    if items:
        validate_cross_mission_harness(items)


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
        validate_all_mission_harness_policies()
        # Shared id namespace (loops vs file schedules): collected once
        # so both seed passes fail loudly on a collision instead of
        # silently overwriting each other's rows.
        file_schedule_ids = _collect_file_schedule_ids()
        loop_ids = _collect_loop_ids()
        both_filters = agent is not None and mission is not None
        seed_missions = agent is None or both_filters
        seed_agents = mission is None or both_filters
        if seed_missions:
            for path in _select_files(BASE / "missions", mission, "mission"):
                try:
                    data = load_mission_file(path)
                    row = upsert_mission(db, data)
                    counts["missions"] += 1
                    logger.info("Seeded mission %s (v%s)", row.id, row.version)
                    print(f"Seeded mission {row.id} (v{row.version})")
                    # Harness loops materialize into Schedule rows right
                    # after their mission is upserted — including
                    # filtered runs (make seed-agent ... MISSION=...),
                    # since the loop belongs to the mission just seeded.
                    seeded_loops, loop_failures = seed_mission_loops(
                        db,
                        row,
                        data["execution_config"].get("harness"),
                        file_schedule_ids=file_schedule_ids,
                        source=str(path),
                    )
                    counts["schedules"] += seeded_loops
                    failures.extend(loop_failures)
                except Exception as exc:  # noqa: BLE001 — one bad file skips itself
                    db.rollback()
                    failures.append((str(path), str(exc)))
                    logger.warning("Skipping mission file %s: %s", path, exc)
                    print(f"FAILED mission {path.name}: {exc}")
        if seed_agents:
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
            seeded_schedules, schedule_failures = seed_schedules(
                db, loop_ids=frozenset(loop_ids)
            )
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
