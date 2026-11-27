"""Load, merge, and validate harness YAML."""

from __future__ import annotations

import copy
import logging
import pathlib
import types
import typing
from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel

from runtime.harness.schema import MissionHarness

logger = logging.getLogger(__name__)

# Harness schema version this code understands. Rows/YAML stamped with a
# different version are rejected by the strict path and warned about (then
# read leniently) by the runtime path.
CURRENT_HARNESS_VERSION = 1

_DEFAULTS_PATH = pathlib.Path(__file__).resolve().parent / "defaults.yaml"
_SEEDS_MISSIONS = (
    pathlib.Path(__file__).resolve().parents[2] / "seeds" / "missions"
)
_SEEDS_AGENTS = pathlib.Path(__file__).resolve().parents[2] / "seeds" / "agents"

# Once-per-process dedupe for lenient-read warnings: an unknown field (or
# version) is warned about the first time it is seen, not on every
# index refresh / execution, or the 2026-11-26 flood comes back.
_WARNED_UNKNOWN_FIELDS: set[str] = set()
_WARNED_VERSIONS: set[Any] = set()


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


def _read_default_harness_file() -> dict[str, Any]:
    if not _DEFAULTS_PATH.is_file():
        return {}
    data = yaml.safe_load(_DEFAULTS_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Harness defaults must be a mapping: {_DEFAULTS_PATH}")
    return data


@lru_cache(maxsize=1)
def _cached_default_harness_dict() -> dict[str, Any]:
    """The shared defaults file, parsed once per process (not per parse)."""
    return _read_default_harness_file()


def load_default_harness_dict() -> dict[str, Any]:
    """A fresh copy of the shared defaults (callers may merge into it)."""
    return copy.deepcopy(_cached_default_harness_dict())


def reset_lenient_warnings_for_tests() -> None:
    """Forget once-per-process lenient warnings (tests only)."""
    _WARNED_UNKNOWN_FIELDS.clear()
    _WARNED_VERSIONS.clear()


def _check_harness_version(
    merged: dict[str, Any], *, source: str, strict: bool
) -> None:
    """Version gate. Strict: unknown version is an error naming the
    version. Lenient: warn once per distinct unknown version, then the
    caller reads the known version-1 fields anyway (a newer row must
    never kill a run on older code). Missing version reads as 1."""
    version = merged.get("version", CURRENT_HARNESS_VERSION)
    if version == CURRENT_HARNESS_VERSION:
        return
    if strict:
        raise ValueError(
            f"Unsupported harness version {version!r} for {source}: "
            f"this code understands harness version {CURRENT_HARNESS_VERSION}"
        )
    if version not in _WARNED_VERSIONS:
        _WARNED_VERSIONS.add(version)
        logger.warning(
            "Unknown harness version %r for %s: reading known "
            "version-%s fields only",
            version,
            source,
            CURRENT_HARNESS_VERSION,
        )


def _submodel_of(annotation: Any) -> type[BaseModel] | None:
    """The BaseModel subclass behind an annotation, if any (unwraps
    Annotated / Union / Optional). Used to recurse the strip only into
    real sub-models — ``dict[str, int]`` fields (e.g.
    ``tool_budget.max_calls``) keep their arbitrary keys."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _submodel_of(typing.get_args(annotation)[0])
    if origin is typing.Union or origin is types.UnionType:
        for arg in typing.get_args(annotation):
            found = _submodel_of(arg)
            if found is not None:
                return found
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _strip_unknown_fields(
    data: dict[str, Any],
    model_cls: type[BaseModel],
    *,
    path: str,
    unknown: list[str],
) -> dict[str, Any]:
    """Drop keys no field of *model_cls* knows, at every depth, and
    collect their dotted paths (e.g. ``notifications.future_flag``)."""
    fields = model_cls.model_fields
    pruned: dict[str, Any] = {}
    for key, value in data.items():
        field_path = f"{path}.{key}" if path else key
        field = fields.get(key)
        if field is None:
            unknown.append(field_path)
            continue
        submodel = _submodel_of(field.annotation)
        if submodel is not None and isinstance(value, dict):
            pruned[key] = _strip_unknown_fields(
                value, submodel, path=field_path, unknown=unknown
            )
        else:
            pruned[key] = value
    return pruned


def parse_mission_harness(
    raw: dict[str, Any] | None,
    *,
    source: str = "mission",
) -> MissionHarness:
    """Merge *raw* ``harness:`` block with shared defaults and validate.

    STRICT path (seed / load time): unknown fields and unknown versions
    fail fast — a typo or a schema the code does not know must be
    caught before it reaches the DB. Runtime reads use
    :func:`parse_mission_harness_lenient` instead.
    """
    defaults = load_default_harness_dict()
    merged = _deep_merge(defaults, raw or {})
    _check_harness_version(merged, source=source, strict=True)
    try:
        return MissionHarness.model_validate(merged)
    except Exception as exc:
        raise ValueError(f"Invalid harness for {source}: {exc}") from exc


def parse_mission_harness_lenient(
    raw: dict[str, Any] | None,
    *,
    source: str = "mission",
) -> MissionHarness:
    """Runtime read path: same merge + validation as
    :func:`parse_mission_harness`, except unknown fields are dropped
    (warned about once per process per dotted path) and unknown
    versions are warned about once, then the known version-1 fields
    are read. Genuine semantic errors in known fields (e.g. a deny
    without its message template) still raise ``ValueError`` — only
    forward-compat skew is tolerated, never bad values."""
    defaults = load_default_harness_dict()
    merged = _deep_merge(defaults, raw or {})
    _check_harness_version(merged, source=source, strict=False)
    unknown: list[str] = []
    pruned = _strip_unknown_fields(merged, MissionHarness, path="", unknown=unknown)
    for field_path in unknown:
        if field_path not in _WARNED_UNKNOWN_FIELDS:
            _WARNED_UNKNOWN_FIELDS.add(field_path)
            logger.warning(
                "Ignoring unknown harness field %s for %s",
                field_path,
                source,
            )
    try:
        return MissionHarness.model_validate(pruned)
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
        except ValueError as exc:
            # Swallow + skip (unchanged): one bad seed file must not
            # kill the index — but it must not disappear silently
            # either, or the mission quietly loses its harness.
            logger.warning(
                "Skipping invalid seed mission harness %s: %s", path, exc
            )
            continue
    return out


def load_mission_harness_from_file(path: pathlib.Path) -> MissionHarness:
    mission_id, data = _read_mission_seed_file(path)
    return parse_mission_harness(data.get("harness"), source=str(path))


def load_seed_harness_for_mission(
    mission_id: str,
) -> tuple[Any, MissionHarness] | None:
    """``(cache_key, harness)`` from ONE seed file, or None when the file
    is missing. A present-but-invalid file raises ``ValueError`` (strict,
    like seed/load) — callers turn that into a logged fallback, the same
    way the index skips a bad seed file.

    cache_key is ``(file mtime_ns, harness version)`` so a re-seeded file
    is picked up without a restart.
    """
    path = _SEEDS_MISSIONS / f"{mission_id}.yaml"
    if not path.is_file():
        return None
    harness = load_mission_harness_from_file(path)
    return ((path.stat().st_mtime_ns, harness.version), harness)


def iter_mission_seed_paths() -> list[pathlib.Path]:
    if not _SEEDS_MISSIONS.is_dir():
        return []
    return sorted(_SEEDS_MISSIONS.glob("*.yaml"))


def iter_agent_seed_paths() -> list[pathlib.Path]:
    if not _SEEDS_AGENTS.is_dir():
        return []
    return sorted(_SEEDS_AGENTS.glob("*.yaml"))