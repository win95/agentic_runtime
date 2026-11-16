"""Write-tool preconditions — data-driven, not prompt-specific."""

from __future__ import annotations

from runtime.scope import SCOPE_UNKNOWN, get_bound_guest_id
from runtime.tool_schema import schema_from_tool, validate_args_against_schema
from runtime.tools.registry import registry


def precondition_catalog() -> dict[str, dict]:
    """Per-tool precondition metadata (generic write gates)."""
    return {
        "record_payment": {"requires_booking_arg": "booking_id"},
        "confirm_payment": {"requires_booking_arg": "booking_id"},
        "cancel_booking": {"requires_booking_arg": "booking_id"},
        "change_booking_dates": {"requires_booking_arg": "booking_id"},
        "create_booking": {"requires_bound_guest": True},
    }


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