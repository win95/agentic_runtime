"""Write-tool preconditions — loaded from declarative config, not tool-name branches."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from runtime.scope import SCOPE_UNKNOWN, get_bound_guest_id
from runtime.tool_schema import schema_from_tool, validate_args_against_schema
from runtime.tools.registry import registry

logger = logging.getLogger(__name__)

_CATALOG: dict[str, dict] | None = None
_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "seeds" / "tool_write_preconditions.yaml"
)

_GUEST_READ_TOOLS = frozenset(
    {"find_guest", "get_booking", "list_payments", "get_task", "search_policy_documents"}
)


def _load_yaml_catalog() -> dict[str, dict]:
    if not _CONFIG_PATH.is_file():
        return {}
    try:
        raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load tool write preconditions: %s", exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict] = {}
    for name, spec in raw.items():
        if isinstance(name, str) and isinstance(spec, dict):
            out[name.strip()] = dict(spec)
    return out


def _catalog_from_skill_maps() -> dict[str, dict]:
    """Merge optional ``map.write_preconditions`` from loaded skills."""
    try:
        from runtime.skills import load_skills
    except Exception:
        return {}
    merged: dict[str, dict] = {}
    for skill in load_skills() or []:
        skill_map = getattr(skill, "map", None)
        if skill_map is None:
            continue
        raw = getattr(skill_map, "write_preconditions", None)
        if not isinstance(raw, dict):
            continue
        for tool_name, spec in raw.items():
            if isinstance(tool_name, str) and isinstance(spec, dict):
                merged[tool_name.strip()] = {**merged.get(tool_name.strip(), {}), **spec}
    return merged


def precondition_catalog() -> dict[str, dict]:
    """Merged declarative catalog (YAML seed + skill-map overrides)."""
    global _CATALOG
    if _CATALOG is None:
        base = _load_yaml_catalog()
        overrides = _catalog_from_skill_maps()
        _CATALOG = {**base, **overrides}
    return dict(_CATALOG)


def reset_precondition_catalog_for_tests() -> None:
    global _CATALOG
    _CATALOG = None


def _booking_exists_for_session(booking_id: str, session_id: str) -> str | None:
    from runtime.tools.registry import _booking_owner

    if not booking_id:
        return "missing booking_id"
    owner = _booking_owner(booking_id)
    if owner is None:
        return f"booking {booking_id!r} does not exist or is not readable"
    bound = get_bound_guest_id(session_id)
    if bound is SCOPE_UNKNOWN:
        return "could not verify session guest scope"
    if isinstance(bound, str) and owner != bound:
        return f"booking {booking_id!r} belongs to another guest"
    return None


def check_write_preconditions(
    tool_name: str,
    args: dict,
    *,
    session_id: str = "",
    execution_id: str = "",
) -> str | None:
    """Return a refusal message, or None when the call may proceed."""
    _ = execution_id
    entry = registry.get(tool_name)
    if entry is None:
        return None
    if entry.read_only:
        return None
    catalog = precondition_catalog().get(tool_name) or {}
    tool = entry.tool
    schema = schema_from_tool(tool) if tool else entry.input_schema
    schema_err = validate_args_against_schema(tool_name, args or {}, schema)
    if schema_err:
        return schema_err
    booking_arg = catalog.get("requires_booking_arg")
    if booking_arg and session_id:
        bid = (args or {}).get(booking_arg)
        if not isinstance(bid, str) or not bid.strip():
            return f"{tool_name}: {booking_arg} is required"
        problem = _booking_exists_for_session(bid.strip(), session_id)
        if problem:
            return f"{tool_name} refused: {problem}"
    if catalog.get("requires_bound_guest") and session_id:
        bound = get_bound_guest_id(session_id)
        if bound is SCOPE_UNKNOWN:
            return f"{tool_name} refused: could not verify session guest scope"
        if not isinstance(bound, str) or not bound:
            return f"{tool_name} refused: session has no bound guest yet"
        guest_arg = (args or {}).get("guest_id")
        if isinstance(guest_arg, str) and guest_arg and guest_arg != bound:
            return f"{tool_name} refused: guest_id outside session scope"
    return None


def guest_read_tool_names() -> frozenset[str]:
    return _GUEST_READ_TOOLS