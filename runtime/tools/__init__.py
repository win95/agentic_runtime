"""Tools (Layer 5) — small callable actions for agents and graph nodes.

Importing this package registers the built-in tools in the Tool Registry.
"""

from runtime.tools import admin_tools, fault_tools  # noqa: F401  (register on import)
from runtime.tools.registry import registry

__all__ = ["registry", "fault_tools", "admin_tools"]
