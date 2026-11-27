"""Harness lookups — Postgres ``execution_config.harness`` with seed fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from runtime.harness.loader import (
    _SEEDS_MISSIONS,
    _read_mission_seed_file,
    iter_mission_seed_paths,
    iter_seed_mission_harnesses,
    load_seed_harness_for_mission,
    parse_mission_harness_lenient,
    reset_lenient_warnings_for_tests,
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

# Per-mission lazy cache (Plan B): mission_id -> (loaded_at, cache_key,
# harness). cache_key is (row updated_at, harness version) in db mode,
# (seed file mtime_ns, harness version) in seed mode. An entry is valid
# only while its key still matches the source AND it is younger than the
# TTL — the key is what makes a re-seed visible immediately, the TTL is
# the safety bound. The returned model is shared (like the old index
# entries) — callers must not mutate it.
_MISSION_CACHE: dict[str, tuple[float, Any, MissionHarness]] = {}
_MISSION_CACHE_LOCK = threading.Lock()

# Snapshot-parse memo: identical execution_config.harness dicts parse
# once per process (content-addressed). The graph resolves the run's
# harness once and passes the model down, but post-run / sandbox / sink
# call sites still call effective_harness with the same snapshot dict —
# the memo keeps those at 1 parse per run too. Failures are never
# cached (the caller falls back); the cache is bounded and cleared
# when full rather than evicted precisely.
_SNAPSHOT_PARSE_CACHE: dict[str, MissionHarness] = {}
_SNAPSHOT_PARSE_CACHE_MAX = 128


class GuestChatMisconfiguredError(RuntimeError):
    """Guest chat routing unavailable (mis-seeded or duplicate mission)."""


@dataclass
class HarnessIndex:
    # Guest routing / guest-output caps used to live here (derived by
    # _derive_guest_routing at index build). Step 3 of plan B removed
    # them: guest_chat_mission_id() / guest_output_capabilities() now
    # run their own light raw scan (_scan_guest_routing) instead of
    # reading the index, so the index is only admin lookups
    # (user_runnable_mission_ids) + GET /health. The fields are gone —
    # no test asserts them (checked 2026-11-27).
    by_mission: dict[str, MissionHarness] = field(default_factory=dict)


def fallback_harness(mission_id: str | None, reason: str) -> MissionHarness:
    """The ONE runtime fallback: log why at warning, return the
    default harness. Every runtime path that cannot read a mission's
    harness goes through here, so a fallback is never silent again.

    The fallback value is ``MissionHarness()`` unchanged — and its
    semantics are deliberately NOT uniformly closed, so read the
    table, do not assume "deny everything":
    - closed: ``security.user_runnable`` False (JWT users cannot run
      the mission), ``guest_channel.exposes_public_chat`` False (no
      guest routing), ``flow_subgraph.enabled`` False (structured
      flow off), ``guest_output.register_reply_tools`` False,
      ``notifications`` both False, ``output_files.required`` False.
    - OPEN: ``builtin_tools.deny`` False — deepagents built-ins stay
      allowed under the fallback. (Q3 decision 2026-11-27: runtime
      invalid harness runs on with default + warn; changing any
      fallback flag needs tran's call, not this helper.)
    """
    logger.warning(
        "Harness fallback for mission %s: %s — using default harness",
        mission_id or "?",
        reason,
    )
    return MissionHarness()


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
    reset_lenient_warnings_for_tests()
    reset_mission_harness_cache_for_tests()


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


def _seed_by_mission_id() -> dict[str, MissionHarness]:
    return {mid: h for mid, h, _p in iter_seed_mission_harnesses()}


def _load_index_from_seeds_only() -> HarnessIndex:
    index = HarnessIndex()
    for mission_id, harness, _path in iter_seed_mission_harnesses():
        index.by_mission[mission_id] = harness
    return index


def _load_index_from_db() -> HarnessIndex:
    """Load harness for missions present in DB only (per-row seed fallback)."""
    from sqlalchemy import select

    from app import models
    from app.core.db import SessionLocal, engine

    timeout = _settings().harness_index_db_timeout_seconds
    seed_fallback = _seed_by_mission_id()
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
            ec = execution_config or {}
            raw = ec.get("harness")
            if isinstance(raw, dict):
                try:
                    # Lenient read: unknown fields / versions from a
                    # newer seed must not break this row (2026-11-26).
                    # Genuine semantic errors still raise -> fallback.
                    index.by_mission[str(mission_id)] = (
                        parse_mission_harness_lenient(
                            raw, source=f"db:{mission_id}"
                        )
                    )
                except ValueError as exc:
                    index.by_mission[str(mission_id)] = fallback_harness(
                        str(mission_id), f"invalid DB harness: {exc}"
                    )
            elif str(mission_id) in seed_fallback:
                index.by_mission[str(mission_id)] = seed_fallback[str(mission_id)]
            else:
                index.by_mission[str(mission_id)] = fallback_harness(
                    str(mission_id),
                    "no harness in DB row and no seed fallback",
                )
    finally:
        db.close()
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


def _mission_cache_ttl_seconds() -> float:
    return float(_settings().harness_index_ttl_seconds)


def _peek_mission_cache_key(mission_id: str) -> Any:
    """Current cache key for one mission without parsing. db mode: the
    row's ``updated_at`` (``onupdate`` bumps it on any row change, so a
    re-seed is visible as soon as the row changes); seed mode: the seed
    file's ``(mtime_ns, harness version)``. Raises on DB/file errors
    so the caller can serve the stale entry instead."""
    if _harness_source() == "seed":
        found = load_seed_harness_for_mission(mission_id)
        if found is None:
            return "missing"
        return found[0]
    from sqlalchemy import select

    from app import models
    from app.core.db import SessionLocal, engine

    timeout = _settings().harness_index_db_timeout_seconds
    db = SessionLocal()
    try:
        if engine.dialect.name == "postgresql":
            from sqlalchemy import text

            db.execute(
                text(f"SET LOCAL statement_timeout = '{int(timeout * 1000)}'")
            )
        updated_at = db.execute(
            select(models.Mission.updated_at).where(
                models.Mission.id == mission_id
            )
        ).scalar_one_or_none()
        if updated_at is None:
            found = load_seed_harness_for_mission(mission_id)
            if found is None:
                return "missing"
            return found[0]
        return (str(updated_at),)
    finally:
        db.close()


def _read_mission_row(mission_id: str) -> tuple[Any, Any] | None:
    """``(cache_key, harness dict or None)`` for one DB row, or None when
    the row does not exist. One indexed SELECT — cheap; the parse is
    what the cache avoids. ``cache_key`` is ``(updated_at,)``: a re-seed
    bumps ``updated_at`` (``onupdate``), so the new row is picked up
    without a restart and without a blind TTL refresh."""
    from sqlalchemy import select

    from app import models
    from app.core.db import SessionLocal, engine

    timeout = _settings().harness_index_db_timeout_seconds
    db = SessionLocal()
    try:
        if engine.dialect.name == "postgresql":
            from sqlalchemy import text

            db.execute(
                text(f"SET LOCAL statement_timeout = '{int(timeout * 1000)}'")
            )
        row = db.execute(
            select(models.Mission.execution_config, models.Mission.updated_at)
            .where(models.Mission.id == mission_id)
        ).first()
        if row is None:
            return None
        execution_config, updated_at = row
        return ((str(updated_at),), (execution_config or {}).get("harness"))
    finally:
        db.close()


def _parse_mission_row_harness(
    mission_id: str, raw: dict[str, Any] | None
) -> MissionHarness:
    """Lenient parse of one mission's harness dict, with the same seed
    fallback the index used: no ``harness`` key in the row → the
    mission's seed file; no seed file either → logged default fallback.
    A present-but-invalid seed file falls back too (strict parse raises
    → logged, like the index skipping a bad seed file)."""
    if isinstance(raw, dict):
        try:
            return parse_mission_harness_lenient(raw, source=f"db:{mission_id}")
        except ValueError as exc:
            return fallback_harness(mission_id, f"invalid DB harness: {exc}")
    try:
        found = load_seed_harness_for_mission(mission_id)
    except ValueError as exc:
        return fallback_harness(mission_id, f"invalid seed harness: {exc}")
    if found is not None:
        return found[1]
    return fallback_harness(
        mission_id, "no harness in DB row and no seed fallback"
    )


def _load_mission_harness_uncached(mission_id: str) -> tuple[Any, MissionHarness]:
    """``(cache_key, harness)`` for one mission, straight from the
    source — no cache read. db mode: the mission's DB row (seed
    fallback per row); seed mode: the mission's seed file only."""
    if _harness_source() == "seed":
        try:
            found = load_seed_harness_for_mission(mission_id)
        except ValueError as exc:
            return ("invalid-seed", fallback_harness(
                mission_id, f"invalid seed harness: {exc}"
            ))
        if found is None:
            return ("missing", fallback_harness(
                mission_id, "no seed file for mission"
            ))
        return found
    row = _read_mission_row(mission_id)
    if row is None:
        try:
            found = load_seed_harness_for_mission(mission_id)
        except ValueError as exc:
            return ("invalid-seed", fallback_harness(
                mission_id, f"invalid seed harness: {exc}"
            ))
        if found is not None:
            return found
        return ("missing", fallback_harness(
            mission_id, "mission not in DB and no seed fallback"
        ))
    cache_key, raw = row
    return (cache_key, _parse_mission_row_harness(mission_id, raw))


def load_mission_harness(mission_id: str) -> MissionHarness:
    """THE one load path (Plan B): one mission's harness, parsed once,
    cached per mission. Every per-execution read goes through here (via
    :func:`effective_harness`); the whole-index build is no longer on
    the hot path — it only serves admin lookups
    (``user_runnable_mission_ids``) and ``GET /health``.

    Cache key is ``(updated_at,)`` (db) or ``(seed file mtime_ns,
    harness version)`` (seed), plus the short TTL as a safety bound —
    a re-seed is visible as soon as the row changes, not after a blind
    30s refresh. A missing/invalid mission falls back through
    :func:`fallback_harness` (logged), never silently.
    """
    if not mission_id:
        return fallback_harness(None, "no mission id")
    now = time.monotonic()
    ttl = _mission_cache_ttl_seconds()
    with _MISSION_CACHE_LOCK:
        entry = _MISSION_CACHE.get(mission_id)
    # One key read per call (cheap indexed SELECT / file stat); the
    # parse is what the cache saves. A match means the cached parse is
    # still valid; a miss (or a stale entry) re-parses from source.
    try:
        current_key = _peek_mission_cache_key(mission_id)
    except Exception:  # noqa: BLE001 — key read must not kill a run
        if entry is not None:
            return entry[2]
        current_key = None
    if (
        entry is not None
        and (now - entry[0]) < ttl
        and current_key is not None
        and current_key == entry[1]
    ):
        return entry[2]
    cache_key, harness = _load_mission_harness_uncached(mission_id)
    with _MISSION_CACHE_LOCK:
        _MISSION_CACHE[mission_id] = (time.monotonic(), cache_key, harness)
    return harness


def _parse_snapshot_harness(
    mission_id: str, raw: dict[str, Any]
) -> MissionHarness:
    """Lenient parse of an ``execution_config.harness`` snapshot dict,
    memoized by content hash. The graph resolves the run's harness once
    and passes the model down, but post-run / sandbox / sink call sites
    still hand the same snapshot dict to ``effective_harness`` — the
    memo keeps those at 1 parse per run too. Failures are never cached
    (the caller falls back instead)."""
    key = hashlib.sha256(
        json.dumps(
            {"m": mission_id, "h": raw}, sort_keys=True, default=str
        ).encode("utf-8")
    ).hexdigest()
    cached = _SNAPSHOT_PARSE_CACHE.get(key)
    if cached is not None:
        return cached
    parsed = parse_mission_harness_lenient(raw, source=f"mission {mission_id}")
    if len(_SNAPSHOT_PARSE_CACHE) >= _SNAPSHOT_PARSE_CACHE_MAX:
        _SNAPSHOT_PARSE_CACHE.clear()
    _SNAPSHOT_PARSE_CACHE[key] = parsed
    return parsed


def reset_mission_harness_cache_for_tests() -> None:
    """Forget per-mission + snapshot caches (tests only)."""
    with _MISSION_CACHE_LOCK:
        _MISSION_CACHE.clear()
    _SNAPSHOT_PARSE_CACHE.clear()


def effective_harness(
    mission_id: str | None,
    execution_config: dict[str, Any] | None = None,
) -> MissionHarness:
    """Thin accessor over :func:`load_mission_harness` (Plan B).

    Snapshot semantics are unchanged: when the execution carries its
    own ``execution_config.harness`` (the run's snapshot), that dict is
    what is parsed — memoized by content, so repeated reads of the same
    snapshot parse once. An invalid snapshot falls back to the mission
    load (same source data the old index entry came from), not straight
    to default. Without a snapshot, the mission load path serves it
    (cached per mission, keyed by row updated_at/version).
    """
    if not mission_id:
        return fallback_harness(None, "no mission id")
    if execution_config and isinstance(execution_config.get("harness"), dict):
        try:
            return _parse_snapshot_harness(
                mission_id, execution_config["harness"]
            )
        except ValueError as exc:
            logger.warning(
                "Harness fallback for mission %s: invalid "
                "execution_config.harness (%s) — falling back to mission load",
                mission_id,
                exc,
            )
    return load_mission_harness(mission_id)


def _raw_guest_flags(raw: Any) -> tuple[bool, str | None]:
    """``(exposes_public_chat, capability-or-None)`` read RAW from a
    harness dict — no defaults merge, no full parse.

    Correct without the merge because ``defaults.yaml`` pins all three
    guest keys to their closed values (``exposes_public_chat`` False,
    ``register_reply_tools`` False, ``capability`` null): a mission
    that does not declare them reads as absent, exactly as the merged
    default would. Only a literal ``True`` counts (not truthy values).
    """
    if not isinstance(raw, dict):
        return False, None
    guest_channel = raw.get("guest_channel")
    exposes = isinstance(guest_channel, dict) and guest_channel.get("exposes_public_chat") is True
    guest_output = raw.get("guest_output")
    cap: str | None = None
    if isinstance(guest_output, dict) and guest_output.get("register_reply_tools") is True:
        candidate = guest_output.get("capability")
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate:
                cap = candidate
    return exposes, cap


def _guest_flags_validated(
    mission_id: str, raw: Any
) -> tuple[bool, str | None]:
    """Raw guest flags, but only from a harness dict that lenient-parses.

    The scan reads flags raw (cheap, no full parse) — but a mission
    whose harness is semantically invalid (a known field with a bad
    value, e.g. ``deny`` without its message template) must NOT route
    guest chat: under the old index build such a row failed its
    lenient parse and fell back to the closed default. So entries whose
    raw flags say anything are validated with one lenient parse; a
    failure warns and the entry is dropped. Entries with no guest flags
    at all skip the parse entirely (the common case stays cheap).
    """
    exposes, cap = _raw_guest_flags(raw)
    if not exposes and cap is None:
        return False, None
    if not isinstance(raw, dict):
        return False, None
    try:
        parse_mission_harness_lenient(raw, source=f"guest-scan:{mission_id}")
    except ValueError as exc:
        logger.warning(
            "Guest routing ignoring mission %s: invalid harness (%s)",
            mission_id,
            exc,
        )
        return False, None
    return exposes, cap


def _raw_harness_from_seed_file(mission_id: str) -> dict[str, Any] | None:
    """Raw ``harness:`` dict from one mission's seed file, or None.

    The same per-row seed fallback the old index build had: a DB row
    without a ``harness`` dict reads its seed file. Missing file is
    None; a broken file is warned about and skipped — one bad seed
    must not take guest chat down.
    """
    path = _SEEDS_MISSIONS / f"{mission_id}.yaml"
    if not path.is_file():
        return None
    try:
        _mid, data = _read_mission_seed_file(path)
    except Exception as exc:  # noqa: BLE001 — broken seed file, skip it
        logger.warning("Skipping invalid seed mission %s: %s", path, exc)
        return None
    raw = data.get("harness")
    return raw if isinstance(raw, dict) else None


def _scan_guest_routing_from_seeds() -> tuple[list[str], list[str]]:
    """Seed-file scan: each seed YAML read raw via
    ``_read_mission_seed_file``, in ``iter_mission_seed_paths``
    (sorted) order; a broken file is warned about and skipped."""
    chat_missions: list[str] = []
    caps: list[str] = []
    seen_caps: set[str] = set()

    def _collect(mission_id: str, raw: Any) -> None:
        exposes, cap = _guest_flags_validated(mission_id, raw)
        if exposes:
            chat_missions.append(mission_id)
        if cap is not None and cap not in seen_caps:
            seen_caps.add(cap)
            caps.append(cap)

    for path in iter_mission_seed_paths():
        try:
            mission_id, data = _read_mission_seed_file(path)
        except Exception as exc:  # noqa: BLE001 — broken seed, skip it
            logger.warning("Skipping invalid seed mission %s: %s", path, exc)
            continue
        _collect(mission_id, data.get("harness"))
    return chat_missions, caps


def _scan_guest_routing_from_db() -> tuple[list[str], list[str]]:
    """DB scan: one ``SELECT (id, execution_config)``, no ORDER BY —
    the same row order the old index build saw. A row without a
    ``harness`` dict falls back to that mission's seed file (see
    ``_raw_harness_from_seed_file``)."""
    from sqlalchemy import select

    from app import models
    from app.core.db import SessionLocal, engine

    chat_missions: list[str] = []
    caps: list[str] = []
    seen_caps: set[str] = set()

    def _collect(mission_id: str, raw: Any) -> None:
        exposes, cap = _guest_flags_validated(mission_id, raw)
        if exposes:
            chat_missions.append(mission_id)
        if cap is not None and cap not in seen_caps:
            seen_caps.add(cap)
            caps.append(cap)

    timeout = _settings().harness_index_db_timeout_seconds
    db = SessionLocal()
    try:
        if engine.dialect.name == "postgresql":
            from sqlalchemy import text

            db.execute(text(f"SET LOCAL statement_timeout = '{int(timeout * 1000)}'"))
        rows = db.execute(select(models.Mission.id, models.Mission.execution_config)).all()
        for mission_id, execution_config in rows:
            raw = (execution_config or {}).get("harness")
            if not isinstance(raw, dict):
                raw = _raw_harness_from_seed_file(str(mission_id))
            _collect(str(mission_id), raw)
    finally:
        db.close()
    return chat_missions, caps


def _scan_guest_routing() -> tuple[list[str], list[str]]:
    """ONE light scan for guest routing + guest-output caps.

    Returns ``(chat_missions, capabilities)`` in source order, caps
    deduped preserving first-seen order. Reads ONLY the two raw guest
    keys — never parses a full harness, never builds the index
    (``_get_index`` / ``_load_index_from_db`` / ``_build_fresh_index``
    are not called) — except for one safety check: an entry whose raw
    flags say anything is validated with one lenient parse
    (``_guest_flags_validated``), so a semantically invalid harness
    cannot route guest chat the way the old index build would have
    rejected it.

    - db mode (``HARNESS_SOURCE=db``): ``_scan_guest_routing_from_db``.
    - seed mode: ``_scan_guest_routing_from_seeds``.

    On a DB error the scan falls back to the seed files (logged) —
    the same fallback the old index refresh (``_refresh_index``) had,
    and what keeps the import-time ``register_guest_channel_tools…``
    call working when Postgres is down: guest routing/caps degrade to
    the seeds instead of taking the import (or ``/guest/chat``) down
    with the DB.

    Deliberately NOT cached: one scan per public call is cheap (one
    SELECT / a handful of small YAML reads) and keeps the code easy to
    read; guest chat is not a hot path. Both public functions share
    this one scan so the DB is never queried twice for one question.
    """
    if _harness_source() == "seed":
        return _scan_guest_routing_from_seeds()
    try:
        return _scan_guest_routing_from_db()
    except Exception as exc:  # noqa: BLE001 — DB down: degrade to seeds
        logger.warning(
            "Guest routing DB scan failed (%s) — falling back to seed files",
            exc,
        )
        return _scan_guest_routing_from_seeds()


def guest_chat_mission_id() -> str:
    """Guest chat mission, from the light raw scan — NOT the index.

    Same contract as before (same error messages): >1 chat mission or
    0 chat missions raise ``GuestChatMisconfiguredError``.
    """
    chat_missions, _caps = _scan_guest_routing()
    if len(chat_missions) > 1:
        raise GuestChatMisconfiguredError("duplicate guest chat missions")
    if not chat_missions:
        raise GuestChatMisconfiguredError("no guest chat mission configured")
    return chat_missions[0]


def user_runnable_mission_ids() -> frozenset[str]:
    index = _get_index()
    return frozenset(
        mid for mid, h in index.by_mission.items() if h.security.user_runnable
    )


def harness_requires_output_files(
    mission_id: str | None,
    execution_config: dict[str, Any] | None = None,
    *,
    harness: MissionHarness | None = None,
) -> bool:
    """True when harness opts into stop-time nudge for missing /output reports.

    Pass the run's already-resolved ``harness`` when the caller has it
    (the graph resolves once per run) — otherwise the harness is loaded
    (cached per mission) via :func:`effective_harness`.
    """
    if harness is None:
        harness = effective_harness(mission_id, execution_config)
    return harness.output_files.required


def mission_denies_builtin_tools(
    mission_id: str | None,
    execution_config: dict[str, Any] | None = None,
    *,
    harness: MissionHarness | None = None,
) -> bool:
    if harness is None:
        harness = effective_harness(mission_id, execution_config)
    return harness.builtin_tools.deny


def builtin_deny_message(
    mission_id: str | None,
    tool_name: str,
    execution_config: dict[str, Any] | None = None,
    *,
    harness: MissionHarness | None = None,
) -> str | None:
    from runtime.harness.validation import format_deny_message

    if harness is None:
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
    *,
    harness: MissionHarness | None = None,
) -> bool:
    if harness is None:
        harness = effective_harness(mission_id, execution_config)
    return harness.flow_subgraph.enabled and bool(mapped_skills)


def guest_output_capabilities() -> list[str]:
    """Guest-output capabilities, from the SAME light raw scan as
    guest routing, following ``HARNESS_SOURCE`` (db rows or seed YAML)
    — the single source for caps, replacing the old seed-only loader
    function. Order is source order, deduped (see ``_scan_guest_routing``).
    """
    _chat_missions, caps = _scan_guest_routing()
    return caps


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