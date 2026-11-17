"""Harness lookups — Postgres ``execution_config.harness`` with seed fallback."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from runtime.harness.loader import (
    iter_mission_seed_paths,
    load_mission_harness_from_file,
    parse_mission_harness,
)
from runtime.harness.schema import MissionHarness

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_INDEX: HarnessIndex | None = None
_INDEX_LOADED_AT: float = 0.0
_TTL_SECONDS = 30.0


class GuestChatMisconfiguredError(RuntimeError):
    """Guest chat routing unavailable (mis-seeded or duplicate mission)."""


@dataclass
class HarnessIndex:
    by_mission: dict[str, MissionHarness] = field(default_factory=dict)
    guest_chat_mission: str | None = None
    guest_chat_duplicate: bool = False
    guest_output_capabilities: list[str] = field(default_factory=list)


def reset_harness_cache_for_tests() -> None:
    global _INDEX, _INDEX_LOADED_AT
    with _LOCK:
        _INDEX = None
        _INDEX_LOADED_AT = 0.0


def _harness_from_seed_file(path) -> tuple[str, MissionHarness | None]:
    try:
        harness = load_mission_harness_from_file(path)
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mission_id = str(data.get("mission_id") or path.stem)
        return mission_id, harness
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping harness from seed file %s: %s", path, exc)
        return path.stem, None


def _merge_seed_fallbacks(index: HarnessIndex) -> None:
    for path in iter_mission_seed_paths():
        mission_id, harness = _harness_from_seed_file(path)
        if harness is None:
            continue
        if mission_id not in index.by_mission:
            index.by_mission[mission_id] = harness


def _derive_guest_routing(index: HarnessIndex) -> None:
    chat_missions: list[str] = []
    caps: list[str] = []
    seen_caps: set[str] = set()
    for mission_id, harness in index.by_mission.items():
        if harness.guest_channel.exposes_public_chat:
            chat_missions.append(mission_id)
        if harness.guest_output.register_reply_tools:
            cap = (harness.guest_output.capability or "").strip()
            if cap and cap not in seen_caps:
                seen_caps.add(cap)
                caps.append(cap)
    index.guest_output_capabilities = caps
    if len(chat_missions) > 1:
        index.guest_chat_duplicate = True
        index.guest_chat_mission = None
        logger.warning(
            "Multiple guest chat missions in harness index: %s", chat_missions
        )
    elif len(chat_missions) == 1:
        index.guest_chat_mission = chat_missions[0]
        index.guest_chat_duplicate = False
    else:
        index.guest_chat_mission = None
        index.guest_chat_duplicate = False


def _seed_paths_by_mission_id() -> dict[str, Any]:
    import yaml
    from pathlib import Path

    out: dict[str, Path] = {}
    for path in iter_mission_seed_paths():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            mid = str(data.get("mission_id") or path.stem)
            out[mid] = path
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping mission seed %s: %s", path, exc)
    return out


def _load_index_from_db_fixed() -> HarnessIndex | None:
    try:
        from app import models
        from app.core.db import SessionLocal

        db = SessionLocal()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Harness DB load skipped: %s", exc)
        return None
    index = HarnessIndex()
    seed_by_mid = _seed_paths_by_mission_id()
    try:
        rows = db.query(models.Mission).all()
        for mission in rows:
            ec = mission.execution_config or {}
            raw = ec.get("harness")
            if isinstance(raw, dict):
                try:
                    index.by_mission[mission.id] = parse_mission_harness(
                        raw, source=f"db:{mission.id}"
                    )
                except ValueError as exc:
                    logger.warning(
                        "Invalid harness in DB for mission %s: %s", mission.id, exc
                    )
            else:
                path = seed_by_mid.get(mission.id)
                if path is not None:
                    _, harness = _harness_from_seed_file(path)
                    if harness is not None:
                        index.by_mission[mission.id] = harness
        return index
    except Exception as exc:  # noqa: BLE001
        logger.warning("Harness DB query failed: %s", exc)
        return None
    finally:
        db.close()


def _load_index_from_seeds_only() -> HarnessIndex:
    index = HarnessIndex()
    for path in iter_mission_seed_paths():
        mission_id, harness = _harness_from_seed_file(path)
        if harness is not None:
            index.by_mission[mission_id] = harness
    return index


def _get_index(*, force: bool = False) -> HarnessIndex:
    global _INDEX, _INDEX_LOADED_AT
    now = time.monotonic()
    with _LOCK:
        if (
            not force
            and _INDEX is not None
            and (now - _INDEX_LOADED_AT) < _TTL_SECONDS
        ):
            return _INDEX
        index = _load_index_from_db_fixed()
        if index is None or not index.by_mission:
            index = _load_index_from_seeds_only()
        else:
            _merge_seed_fallbacks(index)
        _derive_guest_routing(index)
        _INDEX = index
        _INDEX_LOADED_AT = now
        return index


def effective_harness(
    mission_id: str | None,
    execution_config: dict[str, Any] | None = None,
) -> MissionHarness:
    if not mission_id:
        return MissionHarness()
    if execution_config and isinstance(execution_config.get("harness"), dict):
        try:
            return parse_mission_harness(
                execution_config["harness"], source=f"mission {mission_id}"
            )
        except ValueError as exc:
            logger.warning(
                "Invalid execution_config.harness for %s: %s", mission_id, exc
            )
    index = _get_index()
    return index.by_mission.get(mission_id, MissionHarness())


def guest_chat_mission_id() -> str:
    index = _get_index()
    if index.guest_chat_duplicate:
        raise GuestChatMisconfiguredError("duplicate guest chat missions")
    if not index.guest_chat_mission:
        raise GuestChatMisconfiguredError("no guest chat mission configured")
    return index.guest_chat_mission


def user_runnable_mission_ids() -> frozenset[str]:
    index = _get_index()
    return frozenset(
        mid for mid, h in index.by_mission.items() if h.security.user_runnable
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
    from runtime.harness.validation import format_deny_message

    harness = effective_harness(mission_id, execution_config)
    if not harness.builtin_tools.deny:
        return None
    template = harness.builtin_tools.deny_message_template or ""
    try:
        return format_deny_message(template, tool_name)
    except ValueError as exc:
        logger.warning(
            "builtin_deny_message failed for %s: %s", mission_id, exc
        )
        return None


def use_flow_subgraph_harness(
    mission_id: str | None,
    mapped_skills,
    execution_config: dict[str, Any] | None = None,
) -> bool:
    harness = effective_harness(mission_id, execution_config)
    return harness.flow_subgraph.enabled and bool(mapped_skills)


def guest_output_capabilities() -> list[str]:
    return list(_get_index().guest_output_capabilities)


def resolved_harness_for_seed(
    raw_harness: dict[str, Any] | None,
    *,
    source: str,
) -> dict[str, Any]:
    from runtime.harness.loader import harness_dict_from_model

    model = parse_mission_harness(raw_harness, source=source)
    return harness_dict_from_model(model)


def resolve_no_version_system_prompt(agent_id: str | None) -> str:
    """Prompt when no AgentVersion exists: harness fallback or seed YAML text."""
    from pathlib import Path

    import yaml

    from runtime.deep_agent import ADMIN_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT

    fb = agent_system_prompt_fallback(agent_id)
    if fb == "admin":
        return ADMIN_SYSTEM_PROMPT
    if fb == "default":
        return DEFAULT_SYSTEM_PROMPT
    if agent_id:
        path = Path(__file__).resolve().parents[2] / "seeds" / "agents" / f"{agent_id}.yaml"
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            prompt = data.get("system_prompt")
            if isinstance(prompt, str) and prompt.strip():
                return prompt
    return DEFAULT_SYSTEM_PROMPT


def agent_system_prompt_fallback(agent_id: str | None) -> str | None:
    """``default`` / ``admin`` from agent seed harness, else None."""
    if not agent_id:
        return None
    from pathlib import Path

    from runtime.harness.loader import load_agent_harness_from_file

    repo = Path(__file__).resolve().parents[2]
    path = repo / "seeds" / "agents" / f"{agent_id}.yaml"
    if not path.is_file():
        return None
    try:
        return load_agent_harness_from_file(path).system_prompt_fallback
    except ValueError as exc:
        logger.warning("Invalid agent harness for %s: %s", agent_id, exc)
        return None