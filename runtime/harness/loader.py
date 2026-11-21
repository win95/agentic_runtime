"""Load, merge, and validate harness YAML."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import yaml

from runtime.harness.schema import MissionHarness

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


def harness_dict_from_model(model: MissionHarness) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _read_mission_seed_file(path: pathlib.Path) -> tuple[str, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Mission file must be a mapping: {path}")
    mission_id = str(data.get("mission_id") or path.stem)
    return mission_id, data


def iter_seed_mission_harnesses(
) -> list[tuple[str, MissionHarness, pathlib.Path]]:
    """``(mission_id, harness, path)`` from each seeds/missions YAML (one read)."""
    out: list[tuple[str, MissionHarness, pathlib.Path]] = []
    for path in iter_mission_seed_paths():
        try:
            mission_id, data = _read_mission_seed_file(path)
            harness = parse_mission_harness(
                data.get("harness"), source=str(path)
            )
            out.append((mission_id, harness, path))
        except ValueError:
            continue
    return out


def load_mission_harness_from_file(path: pathlib.Path) -> MissionHarness:
    mission_id, data = _read_mission_seed_file(path)
    return parse_mission_harness(data.get("harness"), source=str(path))


def iter_mission_seed_paths() -> list[pathlib.Path]:
    if not _SEEDS_MISSIONS.is_dir():
        return []
    return sorted(_SEEDS_MISSIONS.glob("*.yaml"))


def iter_agent_seed_paths() -> list[pathlib.Path]:
    if not _SEEDS_AGENTS.is_dir():
        return []
    return sorted(_SEEDS_AGENTS.glob("*.yaml"))


def guest_output_capabilities_from_seed_files() -> list[str]:
    """Capabilities for guest reply tools — seed YAML only, never raises."""
    caps: list[str] = []
    seen: set[str] = set()
    for _mid, harness, _path in iter_seed_mission_harnesses():
        if not harness.guest_output.register_reply_tools:
            continue
        cap = (harness.guest_output.capability or "").strip()
        if cap and cap not in seen:
            seen.add(cap)
            caps.append(cap)
    return caps