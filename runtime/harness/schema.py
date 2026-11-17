"""Pydantic models for declarative mission/agent harness profiles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SecurityHarness(BaseModel):
    """Who may invoke this mission through authenticated APIs."""

    user_runnable: bool = False


class GuestChannelHarness(BaseModel):
    """Public guest chat routing (``POST /guest/chat``)."""

    exposes_public_chat: bool = False


class BuiltinToolsHarness(BaseModel):
    """deepagents built-in tool exclusion (model + hook gate)."""

    deny: bool = False
    deny_message_template: str | None = None

    @model_validator(mode="after")
    def _require_template_when_deny(self) -> BuiltinToolsHarness:
        if self.deny and not (self.deny_message_template or "").strip():
            raise ValueError(
                "builtin_tools.deny is true but deny_message_template is empty"
            )
        return self


class FlowSubgraphHarness(BaseModel):
    """Structured flow-router subgraph (skill-mapped missions)."""

    enabled: bool = False


class GuestOutputHarness(BaseModel):
    """Synthetic guest reply / confirmation tools in the registry."""

    register_reply_tools: bool = False
    capability: str | None = None


class MissionHarness(BaseModel):
    """Effective harness for one mission (merged with defaults)."""

    security: SecurityHarness = Field(default_factory=SecurityHarness)
    guest_channel: GuestChannelHarness = Field(default_factory=GuestChannelHarness)
    builtin_tools: BuiltinToolsHarness = Field(default_factory=BuiltinToolsHarness)
    flow_subgraph: FlowSubgraphHarness = Field(default_factory=FlowSubgraphHarness)
    guest_output: GuestOutputHarness = Field(default_factory=GuestOutputHarness)


class AgentHarness(BaseModel):
    """Agent seed metadata — prompt fallback identity for drift checks."""

    system_prompt_fallback: Literal["default", "admin"] | None = None