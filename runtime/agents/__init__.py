"""Agent classes — the shared core (BaseAgent) + per-agent subclasses.

Phase 0 ships only the core: ``BaseAgent`` + the registries. Per-agent
modules (hotel / fault / admin / analytis) arrive with the phase that
gives them a real override — an empty subclass is noise, not structure.
"""

from runtime.agents.base import AgentDefinition, BaseAgent
from runtime.agents.registry import (
    canonical_capability_name,
    register_agent_class,
    register_capability,
    resolve_agent,
    resolve_capability,
)

__all__ = [
    "AgentDefinition",
    "BaseAgent",
    "canonical_capability_name",
    "register_agent_class",
    "register_capability",
    "resolve_agent",
    "resolve_capability",
]
