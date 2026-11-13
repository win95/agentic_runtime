"""Tool Registry — candidates for agents, with runtime permission checks.

The registry only provides the *candidate* list. Whether a tool may
actually run is decided per execution by `check()` against the execution
context (mission capabilities, agent allowed tools, tenant permissions).
A Deep Agent never gets free access to every registered tool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolEntry:
    name: str
    description: str
    capability: str
    permission: str = ""
    # Layer 4 (prompt-injection defense): a tool that writes, deletes or
    # dispatches is read_only=False. Default True — least privilege.
    # Write tools are only visible to missions whose capability matches,
    # and a human-in-the-loop confirmation step is still required before
    # a write tool may run unsupervised (not built yet — see
    # docs/prompt-injection-defense.md).
    read_only: bool = True
    implementation: str = ""
    timeout_seconds: int = 60
    retry_count: int = 1
    input_schema: dict = field(default_factory=dict)
    tool: object | None = None  # the langchain @tool object


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(self, tool, capability: str, permission: str = "",
                 timeout_seconds: int = 60, retry_count: int = 1,
                 read_only: bool = True) -> ToolEntry:
        """Register a langchain @tool under a business capability."""
        entry = ToolEntry(
            name=tool.name,
            description=tool.description or "",
            capability=capability,
            permission=permission,
            read_only=read_only,
            implementation=f"{tool.__module__}.{tool.name}",
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            input_schema=dict(getattr(tool, "args", {}) or {}),
            tool=tool,
        )
        self._tools[entry.name] = entry
        logger.debug("Registered tool %s (capability=%s)", entry.name, capability)
        return entry

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

    def list(self) -> list[ToolEntry]:
        return sorted(self._tools.values(), key=lambda e: e.name)

    def check(self, tool_name: str, context: dict) -> bool:
        """Runtime permission check for one tool in one execution context.

        context: {"mission_id", "tenant", "execution_id",
                  "capabilities": [...], "allowed_tools": [...],
                  "permissions": [...]}
        A tool is allowed only if:
        - it exists, and
        - its capability is in the mission's capabilities (when the mission
          declares capabilities), and
        - for WRITE tools (``read_only=False``), its name is in
          allowed_tools (when the agent declares any), and
        - its required permission is held (when the context lists any
          permissions at all — an empty permission list means "no
          permission restriction beyond capabilities").

        Layer 1 (read-open): READ tools are NOT gated by allowed_tools
        — a read tool whose capability matches the mission is listed,
        so the agent is never starved of the data it needs to build
        context. What is enforced for reads is the *data* scope
        (asset/guest wrappers in :func:`get_langchain_tools`), not the
        tool name. Writes stay gated by allowed_tools AND by the
        call-time layer hook (``enforce_layer``).
        """
        entry = self._tools.get(tool_name)
        if entry is None:
            return False
        capabilities = context.get("capabilities")
        if capabilities is not None and entry.capability not in capabilities:
            return False
        allowed_tools = context.get("allowed_tools")
        if (
            allowed_tools is not None
            and not entry.read_only
            and tool_name not in allowed_tools
        ):
            return False
        permissions = context.get("permissions") or []
        if permissions and entry.permission and entry.permission not in permissions:
            return False
        return True

    def list_for_context(self, context: dict) -> list[ToolEntry]:
        return [e for e in self.list() if self.check(e.name, context)]

    def get_langchain_tools(self, context: dict) -> list:
        """Langchain tool objects the agent may actually call in this context.

        Layer 4 (execution scoping): when the context carries the
        execution's input ``asset_id`` and the mission does not opt out
        via ``execution_config.allow_cross_asset``, a tool that takes an
        ``asset_id`` argument is wrapped so calls for any other asset are
        refused with a structured error instead of running. The wrapper
        is a copy — the registered tool itself is never mutated.

        Layer 1 (guest scope): when the context carries a ``session_id``,
        the guest/booking tools are wrapped so one session can only
        touch one guest (guest identity IS the session — see
        ``runtime/scope.py``): the first guest a session touches binds
        it, later calls naming another guest are refused (writes) or
        have the other guest's data withheld (reads). Reads stay open —
        the wrapper never blocks a read tool by name, only the data.
        """
        scope_asset = (context.get("input") or {}).get("asset_id")
        allow_cross = bool(
            (context.get("execution_config") or {}).get("allow_cross_asset")
        )
        session_id = str(context.get("session_id") or "")
        mission_id = str(context.get("mission_id") or "")
        tools = []
        for entry in self.list_for_context(context):
            if entry.tool is None:
                continue
            tool = entry.tool
            if scope_asset and not allow_cross and _takes_asset_id(tool):
                tool = _scoped_tool(tool, str(scope_asset))
            if session_id and tool.name in _GUEST_SCOPED_TOOLS:
                tool = _guest_scoped_tool(tool, session_id, mission_id)
            tools.append(tool)
        return tools


def _takes_asset_id(tool) -> bool:
    """True when the langchain tool accepts an ``asset_id`` argument."""
    schema = getattr(tool, "args_schema", None)
    if schema is not None:
        try:
            return "asset_id" in (schema.model_fields or {})
        except Exception:  # noqa: BLE001 — fall through to the args dict
            pass
    return "asset_id" in (getattr(tool, "args", None) or {})


def _scoped_tool(tool, scope_asset_id: str):
    """Copy a langchain tool, refusing calls for another ``asset_id``.

    The refusal is a structured error dict (the codebase's pattern for
    tool failures), so the agent sees it as a tool result, not a crash.
    """
    def _guarded(**kwargs):
        requested = kwargs.get("asset_id")
        if requested is not None and str(requested) != scope_asset_id:
            return {
                "error": (
                    f"asset_id '{requested}' is outside this execution's "
                    f"scope (asset_id '{scope_asset_id}'); refused"
                ),
                "asset_id": requested,
                "stub": False,
            }
        return tool.invoke(kwargs)

    return tool.model_copy(update={"func": _guarded})


# Guest/booking tools whose data is scoped to the session's guest
# (Layer 1: guest identity IS the session). Reads are never blocked by
# name — out-of-scope data is withheld from the result instead.
_GUEST_SCOPED_TOOLS = frozenset(
    {
        "find_guest",
        "register_guest",
        "get_booking",
        "create_booking",
        "cancel_booking",
        "change_booking_dates",
    }
)


def _booking_owner(booking_id: str | None) -> str | None:
    """Guest id owning one booking, via the original get_booking tool.

    Returns None when the booking cannot be read (missing, error, or
    get_booking not registered) — the caller treats that as
    unverifiable and refuses the write (fail-closed).
    """
    if not booking_id:
        return None
    entry = registry.get("get_booking")
    if entry is None or entry.tool is None:
        return None
    try:
        result = entry.tool.invoke({"booking_id": booking_id})
    except Exception:  # noqa: BLE001 — unverifiable means refuse
        return None
    if not isinstance(result, dict) or result.get("error"):
        return None
    guest_id = result.get("guest_id")
    return guest_id if isinstance(guest_id, str) and guest_id else None


def _guest_scoped_tool(tool, session_id: str, mission_id: str = ""):
    """Copy a guest/booking tool, enforcing the session's guest scope.

    First sight binds the session: the first guest_id a session
    touches (a find_guest hit, a new registration, a booking read or
    a booking create) becomes the session's guest, and later calls
    naming another guest are refused (writes) or have that guest's
    data withheld (reads). The binding lives in ``session_guest_scopes``
    (see ``runtime/scope.py``) so it survives across the session's
    executions. The refusal is a structured error dict, so the agent
    sees it as a tool result, not a crash. Enforcement is on the sync
    path the runtime uses, like the asset wrapper above. Scope helpers
    are resolved per call (not per wrap) so tests can stub the binding
    store without rebuilding the tools.
    """
    name = tool.name

    def _refusal(message: str, **extra):
        return {"error": message, "stub": False, **extra}

    def _guarded(**kwargs):
        from runtime.scope import bind_guest, get_bound_guest_id

        bound = get_bound_guest_id(session_id)
        if name == "find_guest":
            result = tool.invoke(kwargs)
            if not isinstance(result, dict) or result.get("error"):
                return result
            guests = result.get("guests") or []
            if bound:
                kept = [
                    g
                    for g in guests
                    if isinstance(g, dict) and g.get("guest_id") == bound
                ]
                return {"guests": kept, "count": len(kept)}
            if len(guests) == 1 and isinstance(guests[0], dict):
                guest_id = guests[0].get("guest_id")
                if isinstance(guest_id, str) and guest_id:
                    bind_guest(session_id, guest_id, guests[0].get("phone"), mission_id)
            return result
        if name == "register_guest":
            if bound:
                return _refusal(
                    f"session is already bound to guest '{bound}'; "
                    "register_guest refused"
                )
            result = tool.invoke(kwargs)
            if isinstance(result, dict) and not result.get("error"):
                guest_id = result.get("guest_id")
                if isinstance(guest_id, str) and guest_id:
                    bind_guest(session_id, guest_id, result.get("phone"), mission_id)
            return result
        if name == "get_booking":
            result = tool.invoke(kwargs)
            if not isinstance(result, dict) or result.get("error"):
                return result
            guest_id = result.get("guest_id")
            if not isinstance(guest_id, str) or not guest_id:
                return result
            if bound and guest_id != bound:
                return _refusal(
                    f"booking {kwargs.get('booking_id')!r} belongs to "
                    "another guest; refused",
                    booking_id=kwargs.get("booking_id"),
                )
            if not bound:
                bind_guest(session_id, guest_id, None, mission_id)
            return result
        if name == "create_booking":
            guest_id = kwargs.get("guest_id")
            if bound and isinstance(guest_id, str) and guest_id and guest_id != bound:
                return _refusal(
                    f"guest_id '{guest_id}' is outside this session's "
                    f"scope (guest '{bound}'); refused",
                    guest_id=guest_id,
                )
            result = tool.invoke(kwargs)
            if (
                isinstance(result, dict)
                and not result.get("error")
                and isinstance(guest_id, str)
                and guest_id
            ):
                bind_guest(session_id, guest_id, None, mission_id)
            return result
        if name in ("cancel_booking", "change_booking_dates"):
            booking_id = kwargs.get("booking_id")
            owner = _booking_owner(booking_id)
            if owner is None:
                return _refusal(
                    f"cannot verify owner of booking {booking_id!r}; refused",
                    booking_id=booking_id,
                )
            if bound and owner != bound:
                return _refusal(
                    f"booking {booking_id} belongs to another guest; refused",
                    booking_id=booking_id,
                )
            result = tool.invoke(kwargs)
            if isinstance(result, dict) and not result.get("error") and not bound:
                bind_guest(session_id, owner, None, mission_id)
            return result
        return tool.invoke(kwargs)

    return tool.model_copy(update={"func": _guarded})


# Global registry — tools register themselves on import (see fault_tools).
registry = ToolRegistry()
