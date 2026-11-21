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
    # Layer 4 (prompt-injection defense): argument names holding the
    # id of the thing a destructive call targets (e.g. delete_mission
    # targets ``mission_id``). The guard hook denies the call when the
    # named id does not appear in the execution input — declared here
    # at register time, not in a module-level map in the hook.
    destructive_target_args: tuple[str, ...] = ()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(self, tool, capability: str, permission: str = "",
                 timeout_seconds: int = 60, retry_count: int = 1,
                 read_only: bool = True,
                 destructive_target_args: tuple[str, ...] | str | None = None) -> ToolEntry:
        """Register a langchain @tool under a business capability."""
        if isinstance(destructive_target_args, str):
            destructive_target_args = (destructive_target_args,)
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
            destructive_target_args=tuple(destructive_target_args or ()),
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

        Layer 1 (guest scope): when the context carries a ``session_id``
        and an ``agent`` (resolved by the Mission Graph), the agent's
        own scope policy wraps its tools
        (``agent.scope_tool``) — for the hotel agent, one session can
        only touch one guest (guest identity IS the session — see
        ``runtime/scope.py``). Agents without a policy return their
        tools unchanged, so this changes nothing for other agents.
        """
        scope_asset = (context.get("input") or {}).get("asset_id")
        allow_cross = bool(
            (context.get("execution_config") or {}).get("allow_cross_asset")
        )
        session_id = str(context.get("session_id") or "")
        mission_id = str(context.get("mission_id") or "")
        agent = context.get("agent")
        tools = []
        for entry in self.list_for_context(context):
            if entry.tool is None:
                continue
            tool = entry.tool
            if scope_asset and not allow_cross and _takes_asset_id(tool):
                tool = _scoped_tool(tool, str(scope_asset))
            if session_id and agent is not None:
                tool = agent.scope_tool(tool, session_id, mission_id)
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


# Global registry — tools register themselves on import (see fault_tools).
registry = ToolRegistry()
