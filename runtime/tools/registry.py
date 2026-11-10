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
        - its name is in allowed_tools (when the agent declares any), and
        - its required permission is held (when the context lists any
          permissions at all — an empty permission list means "no
          permission restriction beyond capabilities").
        """
        entry = self._tools.get(tool_name)
        if entry is None:
            return False
        capabilities = context.get("capabilities")
        if capabilities is not None and entry.capability not in capabilities:
            return False
        allowed_tools = context.get("allowed_tools")
        if allowed_tools is not None and tool_name not in allowed_tools:
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
        """
        scope_asset = (context.get("input") or {}).get("asset_id")
        allow_cross = bool(
            (context.get("execution_config") or {}).get("allow_cross_asset")
        )
        tools = []
        for entry in self.list_for_context(context):
            if entry.tool is None:
                continue
            tool = entry.tool
            if scope_asset and not allow_cross and _takes_asset_id(tool):
                tool = _scoped_tool(tool, str(scope_asset))
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
