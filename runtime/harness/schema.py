"""Pydantic models for declarative mission/agent harness profiles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from runtime.harness.validation import validate_deny_message_template


class SecurityHarness(BaseModel):
    """Who may invoke this mission through authenticated APIs."""

    model_config = ConfigDict(extra="forbid")

    user_runnable: StrictBool = False


class GuestChannelHarness(BaseModel):
    """Public guest chat routing (``POST /guest/chat``)."""

    model_config = ConfigDict(extra="forbid")

    exposes_public_chat: StrictBool = False


class BuiltinToolsHarness(BaseModel):
    """deepagents built-in tool exclusion (model + hook gate)."""

    model_config = ConfigDict(extra="forbid")

    deny: StrictBool = False
    deny_message_template: str | None = None

    @model_validator(mode="after")
    def _validate_builtin(self) -> BuiltinToolsHarness:
        if self.deny:
            if not (self.deny_message_template or "").strip():
                raise ValueError(
                    "builtin_tools.deny is true but deny_message_template is empty"
                )
            validate_deny_message_template(self.deny_message_template)
        return self


class FlowSubgraphHarness(BaseModel):
    """Structured flow-router subgraph (skill-mapped missions)."""

    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool = False


class GuestOutputHarness(BaseModel):
    """Synthetic guest reply / confirmation tools in the registry."""

    model_config = ConfigDict(extra="forbid")

    register_reply_tools: StrictBool = False
    capability: str | None = None

    @model_validator(mode="after")
    def _require_capability(self) -> GuestOutputHarness:
        if self.register_reply_tools and not (self.capability or "").strip():
            raise ValueError(
                "guest_output.capability is required when register_reply_tools is true"
            )
        return self


class ContextBudgetHarness(BaseModel):
    """Optional context-window reallocations (opt-in per mission)."""

    model_config = ConfigDict(extra="forbid")

    # When true, unused retrieval slice (8,024 tok) is added to the history
    # trim budget — for missions that never fill the retrieval block.
    reclaim_retrieval_for_history: StrictBool = False


class MissionHarness(BaseModel):
    """Effective harness for one mission (merged with defaults)."""

    model_config = ConfigDict(extra="forbid")

    security: SecurityHarness = Field(default_factory=SecurityHarness)
    guest_channel: GuestChannelHarness = Field(default_factory=GuestChannelHarness)
    builtin_tools: BuiltinToolsHarness = Field(default_factory=BuiltinToolsHarness)
    flow_subgraph: FlowSubgraphHarness = Field(default_factory=FlowSubgraphHarness)
    guest_output: GuestOutputHarness = Field(default_factory=GuestOutputHarness)
    context_budget: ContextBudgetHarness = Field(default_factory=ContextBudgetHarness)


class AgentHarness(BaseModel):
    """Agent seed metadata — prompt fallback identity for drift checks."""

    model_config = ConfigDict(extra="forbid")

    system_prompt_fallback: Literal["default", "admin", "seed"] | None = None