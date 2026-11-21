"""Agent + capability registries (plan Phase 0–1).

Agent classes register by ``agent_id``; an agent with no class resolves
to ``BaseAgent``, so a YAML-only agent (``scripts/new_agent.py``) keeps
working with no Python at all. Capability subgraphs register by name —
a new capability no longer requires editing ``mission_graph.py``; the
module that builds the subgraph registers itself on import.

Both registries are plain dicts on purpose: no import-time magic
beyond the capability modules' own ``@register_capability`` decorator.
"""

from __future__ import annotations

from typing import Any, Callable

from runtime.agents.base import BaseAgent

_AGENT_CLASSES: dict[str, type[BaseAgent]] = {}

_CAPABILITY_BUILDERS: dict[str, Callable[[], Any]] = {}
_CAPABILITY_ALIASES: dict[str, str] = {}


def register_agent_class(agent_id: str, cls: type[BaseAgent]) -> type[BaseAgent]:
    """Register an agent subclass for one agent id. Returns the class,
    so it works as a decorator."""
    _AGENT_CLASSES[agent_id] = cls
    return cls


def resolve_agent(
    agent_id: str, mission: dict, agent_version: Any = None
) -> BaseAgent:
    """The agent object for one run — the registered subclass, or
    ``BaseAgent`` when the agent declares no class."""
    cls = _AGENT_CLASSES.get(agent_id or "") or BaseAgent
    return cls(agent_id or "", mission, agent_version)


def register_capability(
    name: str, builder: Callable[[], Any] | None = None, *, aliases: tuple[str, ...] = ()
):
    """Register a capability subgraph builder, as a decorator or a
    direct call. ``aliases`` are the business capability names that
    route to this subgraph (e.g. the fault tools' capability names
    all map to the one fault_analysis subgraph — see the NOTE in
    ``capability_node`` about capability names doing double duty).
    """
    def _register(fn: Callable[[], Any]) -> Callable[[], Any]:
        _CAPABILITY_BUILDERS[name] = fn
        for alias in aliases:
            _CAPABILITY_ALIASES[alias] = name
        return fn

    if builder is not None:
        return _register(builder)
    return _register


def canonical_capability_name(name: str) -> str | None:
    """The subgraph name one capability routes to, or None when the
    capability is not implemented."""
    if name in _CAPABILITY_BUILDERS:
        return name
    return _CAPABILITY_ALIASES.get(name)


def resolve_capability(name: str) -> Callable[[], Any] | None:
    """The subgraph builder for one capability name, or None."""
    graph_name = canonical_capability_name(name)
    if graph_name is None:
        return None
    return _CAPABILITY_BUILDERS.get(graph_name)
