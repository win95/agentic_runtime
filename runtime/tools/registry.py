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
    implementation: str = ""
    timeout_seconds: int = 60
    retry_count: int = 1
    input_schema: dict = field(default_factory=dict)
    tool: object | None = None  # the langchain @tool object


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(self, tool, capability: str, permission: str = "",
                 timeout_seconds: int = 60, retry_count: int = 1) -> ToolEntry:
        """Register a langchain @tool under a business capability."""
        entry = ToolEntry(
            name=tool.name,
            description=tool.description or "",
            capability=capability,
            permission=permission,
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
        """Langchain tool objects the agent may actually call in this context."""
        return [e.tool for e in self.list_for_context(context) if e.tool is not None]


# Global registry — tools register themselves on import (see fault_tools).
registry = ToolRegistry()
