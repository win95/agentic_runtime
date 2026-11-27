"""Harness lookups — Postgres ``execution_config.harness`` with seed fallback."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from runtime.harness.loader import (
    iter_seed_mission_harnesses,
    parse_mission_harness,
)
from runtime.harness.schema import MissionHarness

logger = logging.getLogger(__name__)

_AGENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")

_LOCK = threading.Lock()
_REFRESH_LOCK = threading.Lock()
_INDEX: HarnessIndex | None = None
_LAST_GOOD_INDEX: HarnessIndex | None = None
_INDEX_LOADED_AT: float = 0.0
_DB_BACKOFF_UNTIL: float = 0.0
_DB_BACKOFF_SECONDS: float = 5.0
_INDEX_DEGRADED: bool = False


class GuestChatMisconfiguredError(RuntimeError):
    """Guest chat routing unavailable (mis-seeded or duplicate mission)."""


@dataclass
class HarnessIndex:
    by_mission: dict[str, MissionHarness] = field(default_factory=dict)
    guest_chat_mission: str | None = None
    guest_chat_duplicate: bool = False
    guest_output_capabilities: list[str] = field(default_factory=list)


def reset_harness_cache_for_tests() -> None:
    global _INDEX, _LAST_GOOD_INDEX, _INDEX_LOADED_AT, _DB_BACKOFF_UNTIL
    global _DB_BACKOFF_SECONDS, _INDEX_DEGRADED
    with _LOCK:
        _INDEX = None
        _LAST_GOOD_INDEX = None
        _INDEX_LOADED_AT = 0.0
        _DB_BACKOFF_UNTIL = 0.0
        _DB_BACKOFF_SECONDS = 5.0
        _INDEX_DEGRADED = False


def harness_index_degraded() -> bool:
    with _LOCK:
        return _INDEX_DEGRADED


def _settings():
    from app.config import settings

    return settings


def _harness_source() -> str:
    return _settings().harness_source


def _ttl_seconds() -> float:
    return float(_settings().harness_index_ttl_seconds)


def _mission_allowlist() -> frozenset[str] | None:
    """Mission ids from HARNESS_MISSION_ALLOWLIST (comma-separated), or
    None when unset (no filter — full index). Only a split worker sets
    this (worker-lark: "agent_lark_me_mission"); the API and the shared
    worker leave it empty."""
    raw = (_settings().harness_mission_allowlist or "").strip()
    if not raw:
        return None
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _seed_by_mission_id() -> dict[str, MissionHarness]:
    return {mid: h for mid, h, _p in iter_seed_mission_harnesses()}


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


def _load_index_from_seeds_only() -> HarnessIndex:
    index = HarnessIndex()
    allowlist = _mission_allowlist()
    for mission_id, harness, _path in iter_seed_mission_harnesses():
        if allowlist is not None and mission_id not in allowlist:
            continue
        index.by_mission[mission_id] = harness
    _derive_guest_routing(index)
    return index


def _load_index_from_db() -> HarnessIndex:
    """Load harness for missions present in DB only (per-row seed fallback)."""
    from sqlalchemy import select

    from app import models
    from app.core.db import SessionLocal, engine

    timeout = _settings().harness_index_db_timeout_seconds
    seed_fallback = _seed_by_mission_id()
    allowlist = _mission_allowlist()
    index = HarnessIndex()
    db = SessionLocal()
    try:
        if engine.dialect.name == "postgresql":
            from sqlalchemy import text

            db.execute(
                text(f"SET LOCAL statement_timeout = '{int(timeout * 1000)}'")
            )
        stmt = select(models.Mission.id, models.Mission.execution_config)
        rows = db.execute(stmt).all()
        for mission_id, execution_config in rows:
            if allowlist is not None and str(mission_id) not in allowlist:
                continue
            ec = execution_config or {}
            raw = ec.get("harness")
            if isinstance(raw, dict):
                try:
                    index.by_mission[str(mission_id)] = parse_mission_harness(
                        raw, source=f"db:{mission_id}"
                    )
                except ValueError as exc:
                    logger.error(
                        "Invalid harness in DB for mission %s: %s", mission_id, exc
                    )
                    index.by_mission[str(mission_id)] = MissionHarness()
            elif str(mission_id) in seed_fallback:
                index.by_mission[str(mission_id)] = seed_fallback[str(mission_id)]
            else:
                index.by_mission[str(mission_id)] = MissionHarness()
    finally:
        db.close()
    _derive_guest_routing(index)
    return index


def _build_fresh_index() -> HarnessIndex:
    if _harness_source() == "seed":
        return _load_index_from_seeds_only()
    return _load_index_from_db()


def _refresh_index() -> HarnessIndex:
    """Query outside the hot lock; stale-while-error on failure."""
    global _INDEX, _LAST_GOOD_INDEX, _INDEX_LOADED_AT, _DB_BACKOFF_UNTIL
    global _DB_BACKOFF_SECONDS, _INDEX_DEGRADED

    now = time.monotonic()
    if now < _DB_BACKOFF_UNTIL:
        with _LOCK:
            if _LAST_GOOD_INDEX is not None:
                return _LAST_GOOD_INDEX
            if _INDEX is not None:
                return _INDEX
        return _load_index_from_seeds_only()

    if not _REFRESH_LOCK.acquire(blocking=False):
        with _LOCK:
            if _LAST_GOOD_INDEX is not None:
                return _LAST_GOOD_INDEX
            if _INDEX is not None:
                return _INDEX
        return _load_index_from_seeds_only()

    try:
        try:
            fresh = _build_fresh_index()
        except Exception as exc:  # noqa: BLE001
            logger.error("Harness index refresh failed: %s", exc, exc_info=True)
            with _LOCK:
                _INDEX_DEGRADED = True
                ttl = _ttl_seconds()
                _DB_BACKOFF_SECONDS = min(_DB_BACKOFF_SECONDS * 2, ttl)
                _DB_BACKOFF_UNTIL = now + _DB_BACKOFF_SECONDS
                if _LAST_GOOD_INDEX is not None:
                    return _LAST_GOOD_INDEX
                if _INDEX is not None:
                    return _INDEX
            return _load_index_from_seeds_only()

        with _LOCK:
            _INDEX = fresh
            _LAST_GOOD_INDEX = fresh
            _INDEX_LOADED_AT = now
            _INDEX_DEGRADED = False
            _DB_BACKOFF_SECONDS = 5.0
            _DB_BACKOFF_UNTIL = 0.0
        return fresh
    finally:
        _REFRESH_LOCK.release()


def _get_index(*, force: bool = False) -> HarnessIndex:
    now = time.monotonic()
    with _LOCK:
        cached = _INDEX
        loaded_at = _INDEX_LOADED_AT
        if (
            not force
            and cached is not None
            and (now - loaded_at) < _ttl_seconds()
        ):
            return cached
    return _refresh_index()


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


def harness_requires_output_files(
    mission_id: str | None,
    execution_config: dict[str, Any] | None = None,
) -> bool:
    """True when harness opts into stop-time nudge for missing /output reports."""
    return effective_harness(mission_id, execution_config).output_files.required


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


def resolve_no_version_system_prompt(agent_id: str | None) -> str:
    """Prompt when no AgentVersion exists: the agent's own fallback.

    The agent class declares it (``BaseAgent.fallback_system_prompt``
    reads the agent's seed YAML; ``AdminAgent`` returns its own text
    without a file). The old harness ``system_prompt_fallback``
    Literal (default/admin/seed) and the ``ADMIN_SYSTEM_PROMPT``
    constant are gone — no string dispatch, no shared constant.
    """
    from runtime.deep_agent import DEFAULT_SYSTEM_PROMPT

    if not agent_id or not _AGENT_ID_RE.match(agent_id):
        return DEFAULT_SYSTEM_PROMPT
    from runtime.agents.registry import resolve_agent

    try:
        prompt = resolve_agent(agent_id, {"mission_id": ""}).fallback_system_prompt()
    except Exception:  # noqa: BLE001 — a broken agent class must not kill the run
        return DEFAULT_SYSTEM_PROMPT
    return prompt or DEFAULT_SYSTEM_PROMPT