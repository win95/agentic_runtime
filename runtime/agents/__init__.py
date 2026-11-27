"""Agent classes — the shared core (BaseAgent) + per-agent subclasses.

AdminAgent arrived with Phase 5.5 (its own fallback prompt).
HotelReceptionistAgent (Phase 2+3, guest scope + write pipeline) was
archived 2026-11-27 — see archive/agents/hotel_receptionist/.
Further per-agent modules (fault / analytis) arrive with the phase
that gives them a real override — an empty subclass is noise, not
structure.
"""

from runtime.agents.admin import AdminAgent  # noqa: F401 — self-registers on import
from runtime.agents.base import AgentDefinition, BaseAgent
from runtime.agents.lark import LarkAgent  # noqa: F401 — self-registers on import
from runtime.agents.registry import (
    canonical_capability_name,
    register_agent_class,
    register_capability,
    resolve_agent,
    resolve_capability,
)

__all__ = [
    "AdminAgent",
    "AgentDefinition",
    "BaseAgent",
    "LarkAgent",
    "canonical_capability_name",
    "register_agent_class",
    "register_capability",
    "resolve_agent",
    "resolve_capability",
]
