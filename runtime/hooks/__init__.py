"""Hooks — lifecycle events for one agent execution (Layer 4).

Self-hosted equivalent of the Hooks v2 system in Deep Agents Code
(CLI ``dcode``). See ``runtime/hooks/events.py`` for the full story:
same 12 events and per-event effects, implemented with Python
callables + native ``langchain`` ``AgentMiddleware`` instead of shell
scripts from ``hooks.json``.
"""

from __future__ import annotations

from runtime.hooks.events import BEHAVIORAL_EVENTS, HookContext, HookDecision, HookEvent
from runtime.hooks.middleware import HookMiddleware
from runtime.hooks.registry import HookRegistry, default_registry, select_hooks

__all__ = [
    "BEHAVIORAL_EVENTS",
    "HookContext",
    "HookDecision",
    "HookEvent",
    "HookMiddleware",
    "HookRegistry",
    "default_registry",
    "select_hooks",
]
