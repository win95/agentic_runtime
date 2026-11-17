"""Load, merge, and validate harness YAML."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import yaml

from runtime.harness.schema import AgentHarness, MissionHarness

_DEFAULTS_PATH = pathlib.Path(__file__).resolve().parent / "defaults.yaml"
_SEEDS_MISSIONS = (
    pathlib.Path(__file__).resolve().parents[2] / "seeds" / "missions"
)
_SEEDS_AGENTS = pathlib.Path(__file__).resolve().parents[2] / "seeds" / "agents"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_default_harness_dict() -> dict[str, Any]:
    if not _DEFAULTS_PATH.is_file():
        return {}
    data = yaml.safe_load(_DEFAULTS_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Harness defaults must be a mapping: {_DEFAULTS_PATH}")
    return data


def parse_mission_harness(
    raw: dict[str, Any] | None,
    *,
    source: str = "mission",
) -> MissionHarness:
    """Merge *raw* ``harness:`` block with shared defaults and validate."""
    defaults = load_default_harness_dict()
    merged = _deep_merge(defaults, raw or {})
    try:
        return MissionHarness.model_validate(merged)
    except Exception as exc:
        raise ValueError(f"Invalid harness for {source}: {exc}") from exc


def parse_agent_harness(
    raw: dict[str, Any] | None,
    *,
    source: str = "agent",
) -> AgentHarness:
    if not raw:
        return AgentHarness()
    try:
        return AgentHarness.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid agent harness for {source}: {exc}") from exc


def harness_dict_from_model(model: MissionHarness) -> dict[str, Any]:
    return model.model_dump(mode="json")


def load_mission_harness_from_file(path: pathlib.Path) -> MissionHarness:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Mission file must be a mapping: {path}")
    return parse_mission_harness(data.get("harness"), source=str(path))


def load_agent_harness_from_file(path: pathlib.Path) -> AgentHarness:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Agent file must be a mapping: {path}")
    return parse_agent_harness(data.get("harness"), source=str(path))


def iter_mission_seed_paths() -> list[pathlib.Path]:
    if not _SEEDS_MISSIONS.is_dir():
        return []
    return sorted(_SEEDS_MISSIONS.glob("*.yaml"))


def iter_agent_seed_paths() -> list[pathlib.Path]:
    if not _SEEDS_AGENTS.is_dir():
        return []
    return sorted(_SEEDS_AGENTS.glob("*.yaml"))