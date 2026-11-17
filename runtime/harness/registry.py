"""Cached harness lookups used by runtime, API, and manual checks."""

from __future__ import annotations

import threading
from typing import Any

from runtime.harness.loader import (
    harness_dict_from_model,
    iter_mission_seed_paths,
    load_mission_harness_from_file,
    parse_mission_harness,
)
from runtime.harness.schema import MissionHarness

_LOCK = threading.Lock()
_SEED_CACHE: dict[str, MissionHarness] | None = None
_GUEST_CHAT_MISSION: str | None = None

def _refresh_seed_cache() -> None:
    global _SEED_CACHE, _GUEST_CHAT_MISSION
    import yaml

    cache: dict[str, MissionHarness] = {}
    guest_mission: str | None = None
    for path in iter_mission_seed_paths():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mission_id = data.get("mission_id") or path.stem
        harness = load_mission_harness_from_file(path)
        cache[str(mission_id)] = harness
        if harness.guest_channel.exposes_public_chat:
            if guest_mission is not None and guest_mission != mission_id:
                raise ValueError(
                    "Only one mission may set guest_channel.exposes_public_chat; "
                    f"found {guest_mission!r} and {mission_id!r}"
                )
            guest_mission = str(mission_id)
    _SEED_CACHE = cache
    _GUEST_CHAT_MISSION = guest_mission


def reset_harness_cache_for_tests() -> None:
    global _SEED_CACHE, _GUEST_CHAT_MISSION
    with _LOCK:
        _SEED_CACHE = None
        _GUEST_CHAT_MISSION = None


def _ensure_seed_cache() -> dict[str, MissionHarness]:
    global _SEED_CACHE
    with _LOCK:
        if _SEED_CACHE is None:
            _refresh_seed_cache()
        assert _SEED_CACHE is not None
        return _SEED_CACHE


def effective_harness(
    mission_id: str | None,
    execution_config: dict[str, Any] | None = None,
) -> MissionHarness:
    if not mission_id:
        return MissionHarness()
    if execution_config and isinstance(execution_config.get("harness"), dict):
        return parse_mission_harness(
            execution_config["harness"], source=f"mission {mission_id}"
        )
    cache = _ensure_seed_cache()
    return cache.get(mission_id, MissionHarness())


def harness_for_mission_dict(mission_row: dict[str, Any]) -> MissionHarness:
    mission_id = mission_row.get("id") or mission_row.get("mission_id")
    execution_config = mission_row.get("execution_config") or {}
    return effective_harness(mission_id, execution_config)


def guest_chat_mission_id() -> str:
    _ensure_seed_cache()
    if _GUEST_CHAT_MISSION is None:
        raise RuntimeError(
            "No mission declares harness.guest_channel.exposes_public_chat"
        )
    return _GUEST_CHAT_MISSION


def user_runnable_mission_ids() -> frozenset[str]:
    cache = _ensure_seed_cache()
    return frozenset(
        mid for mid, h in cache.items() if h.security.user_runnable
    )


def mission_denies_builtin_tools(
    mission_id: str | None,
    execution_config: dict[str, Any] | None = None,
) -> bool:
    return effective_harness(mission_id, execution_config).builtin_tools.deny


def builtin_deny_message(
    mission_id: str | None,
    tool_name: str,
    execution_config: dict[str, Any] | None = None,
) -> str | None:
    harness = effective_harness(mission_id, execution_config)
    if not harness.builtin_tools.deny:
        return None
    template = harness.builtin_tools.deny_message_template or ""
    return template.format(tool_name=tool_name)


def use_flow_subgraph_harness(
    mission_id: str | None,
    mapped_skills,
    execution_config: dict[str, Any] | None = None,
) -> bool:
    harness = effective_harness(mission_id, execution_config)
    return harness.flow_subgraph.enabled and bool(mapped_skills)


def guest_output_capabilities() -> list[str]:
    caps: list[str] = []
    for _, harness in _ensure_seed_cache().items():
        if not harness.guest_output.register_reply_tools:
            continue
        cap = harness.guest_output.capability
        if cap:
            caps.append(cap)
    return caps


def resolved_harness_for_seed(
    raw_harness: dict[str, Any] | None,
    *,
    source: str,
) -> dict[str, Any]:
    model = parse_mission_harness(raw_harness, source=source)
    return harness_dict_from_model(model)