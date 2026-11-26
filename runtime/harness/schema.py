"""Pydantic models for declarative mission harness profiles."""

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


class NotificationsHarness(BaseModel):
    """Slack notification scope (opt-in per mission).

    When true, a SUCCEEDED execution of this mission notifies Slack
    when its output reports a non-empty ``actions_taken`` list — the
    "admin action" scope (tran's choice, 2026-11-11). Failures notify
    for every mission regardless of this flag.
    """

    model_config = ConfigDict(extra="forbid")

    notify_admin_actions: StrictBool = False


class ContextBudgetHarness(BaseModel):
    """Optional context-window reallocations (opt-in per mission)."""

    model_config = ConfigDict(extra="forbid")

    # When true, unused retrieval slice (8,024 tok) is added to the history
    # trim budget — for missions that never fill the retrieval block.
    reclaim_retrieval_for_history: StrictBool = False


class ToolCallsHarness(BaseModel):
    """Per-mission tool invocation policy."""

    model_config = ConfigDict(extra="forbid")

    # Tools in this list must not run in parallel within one model message.
    sequential: list[str] = Field(default_factory=list)


class ToolBudgetHarness(BaseModel):
    """Shared execution budget for web tool results (opt-in)."""

    model_config = ConfigDict(extra="forbid")

    tools: list[str] = Field(default_factory=list)
    max_result_tokens: int = 0
    max_calls: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_budget(self) -> ToolBudgetHarness:
        if self.max_result_tokens < 0:
            raise ValueError("tool_budget.max_result_tokens must be >= 0")
        for name, cap in self.max_calls.items():
            if not (name or "").strip():
                raise ValueError("tool_budget.max_calls keys must be non-empty")
            if cap < 0:
                raise ValueError(f"tool_budget.max_calls[{name!r}] must be >= 0")
        return self


class SandboxHarness(BaseModel):
    """OpenSandbox command and lifetime policy for sandbox missions."""

    model_config = ConfigDict(extra="forbid")

    command_timeout_seconds: int = 90
    command_timeout_max_seconds: int = 600

    @model_validator(mode="after")
    def _validate_timeout(self) -> SandboxHarness:
        if self.command_timeout_seconds < 1:
            raise ValueError("sandbox.command_timeout_seconds must be >= 1")
        if self.command_timeout_max_seconds < 1:
            raise ValueError("sandbox.command_timeout_max_seconds must be >= 1")
        if self.command_timeout_max_seconds > 7200:
            raise ValueError("sandbox.command_timeout_max_seconds must be <= 7200")
        if self.command_timeout_seconds > self.command_timeout_max_seconds:
            raise ValueError(
                "sandbox.command_timeout_seconds must be <= command_timeout_max_seconds"
            )
        return self


class OutputFilesHarness(BaseModel):
    """Sandbox report export expectations (opt-in per mission)."""

    model_config = ConfigDict(extra="forbid")

    # When true, ``stop_completion`` nudges the agent to write top-level
    # report files under ``/output`` before finishing (sandbox + marker only).
    required: StrictBool = False


class OutputArtifactsHarness(BaseModel):
    """Where sandbox ``output.files`` payloads are stored after a run."""

    model_config = ConfigDict(extra="forbid")

    sink: Literal["none", "minio"] = "none"
    # Drop inline ``content`` from each file entry after a successful upload.
    strip_inline: StrictBool = True
    # When upload fails, keep inline ``content`` and mark storage as inline.
    inline_fallback: StrictBool = True


class MissionHarness(BaseModel):
    """Effective harness for one mission (merged with defaults)."""

    model_config = ConfigDict(extra="forbid")

    security: SecurityHarness = Field(default_factory=SecurityHarness)
    guest_channel: GuestChannelHarness = Field(default_factory=GuestChannelHarness)
    builtin_tools: BuiltinToolsHarness = Field(default_factory=BuiltinToolsHarness)
    flow_subgraph: FlowSubgraphHarness = Field(default_factory=FlowSubgraphHarness)
    guest_output: GuestOutputHarness = Field(default_factory=GuestOutputHarness)
    notifications: NotificationsHarness = Field(default_factory=NotificationsHarness)
    context_budget: ContextBudgetHarness = Field(default_factory=ContextBudgetHarness)
    output_files: OutputFilesHarness = Field(default_factory=OutputFilesHarness)
    output_artifacts: OutputArtifactsHarness = Field(
        default_factory=OutputArtifactsHarness
    )
    tool_calls: ToolCallsHarness = Field(default_factory=ToolCallsHarness)
    tool_budget: ToolBudgetHarness = Field(default_factory=ToolBudgetHarness)
    sandbox: SandboxHarness = Field(default_factory=SandboxHarness)


class AgentHarness(BaseModel):
    """Agent seed metadata — prompt fallback identity for drift checks."""

    model_config = ConfigDict(extra="forbid")

    system_prompt_fallback: Literal["default", "admin", "seed"] | None = None
