"""Hooks — lifecycle events for one agent execution (Layer 4).

Self-hosted equivalent of the Hooks v2 system in Deep Agents Code
(CLI ``dcode``). Hooks v2 itself is CLI-only: it runs external commands
from ``hooks.json`` and ``create_deep_agent`` has no hooks parameter, so
it cannot be used by a server-side SDK deployment like this one. What we
extract here is the *semantics* — the same 12 events, the same matcher
idea, the same per-event effect (deny a tool, inject feedback, force
another turn, block a prompt) — implemented with plain Python callables
and the native ``langchain`` ``AgentMiddleware`` (``wrap_tool_call``,
``after_model`` + ``jump_to``, ``before_agent``/``after_agent``).

A hook handler receives a :class:`HookContext` and returns a
:class:`HookDecision` (or ``None`` for "no opinion"). Handlers never
raise into the run: the registry catches handler exceptions, logs them,
and treats the handler as silent. A hook therefore can never fail an
execution by itself — it can only deny, give feedback, or ask for
another turn through the decisions below.

Per-event effect (mirrors Hooks v2 exit-code 2):
- ``PreToolUse`` — ``deny`` blocks the tool call (tool never runs).
- ``PostToolUse`` / ``PostToolUseFailure`` — ``feedback`` text is
  appended to the tool result the agent sees.
- ``Stop`` — ``continue`` forces one more model turn (capped per run);
  anything else lets the agent stop.
- ``UserPromptSubmit`` — ``block`` rejects the execution input before
  the agent is built.
- ``SessionStart`` — ``additional_context`` is injected as a system
  message when the agent starts.
- ``SessionEnd``, ``Notification``, ``PermissionRequest``,
  ``PreCompact``, ``SubagentStart``, ``SubagentStop`` — diagnostic
  only in this codebase (see ``docs/hooks.md`` for which ones have a
  live dispatch site and which are defined but unwired).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class HookEvent(str, Enum):
    """Lifecycle events. Values are the Hooks v2 event names."""

    SESSION_START = "SessionStart"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    SESSION_END = "SessionEnd"
    PERMISSION_REQUEST = "PermissionRequest"
    NOTIFICATION = "Notification"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"
    PRE_COMPACT = "PreCompact"
    STOP = "Stop"
    SUBAGENT_START = "SubagentStart"
    SUBAGENT_STOP = "SubagentStop"


# Events whose decisions can change what the run does. All others are
# diagnostic: their handlers are called and recorded, but no decision
# they return alters the run.
BEHAVIORAL_EVENTS = frozenset(
    {
        HookEvent.SESSION_START,
        HookEvent.USER_PROMPT_SUBMIT,
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_USE,
        HookEvent.POST_TOOL_USE_FAILURE,
        HookEvent.STOP,
    }
)


@dataclass
class HookContext:
    """Everything a hook handler may need, for one dispatch.

    Only the fields relevant to the dispatched event are filled; the
    rest stay at their defaults. ``data`` carries event-specific extras
    (e.g. ``tool_calls_made``, ``evidence_count``, ``continues`` for
    the Stop event) so handlers stay small and the context stays flat.
    """

    execution_id: str = ""
    mission_id: str = ""
    tenant: str = "default"
    agent_id: str = ""
    mission: dict = field(default_factory=dict)
    input: dict = field(default_factory=dict)
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_response: str = ""
    error: str = ""
    reason: str = ""
    messages: list = field(default_factory=list)
    data: dict = field(default_factory=dict)
    # Layer enforcement (L2): the run's scope, carried so the
    # enforce_layer hook can decide a tool call without re-resolving
    # the version. Empty allowed_tools means "no version gate" (the
    # pre-versioning fallback) — reads still pass, writes need the
    # gate only when a version is active.
    session_id: str = ""
    allowed_tools: list = field(default_factory=list)
    capabilities: list = field(default_factory=list)


@dataclass
class HookDecision:
    """What one handler wants to happen.

    ``action`` is one of: ``allow`` (no opinion / proceed), ``deny``
    (PreToolUse: do not run the tool), ``feedback`` (PostToolUse: text
    for the agent), ``continue`` (Stop: run one more model turn),
    ``block`` (UserPromptSubmit: reject the input), ``stop`` (Stop:
    let the agent finish). ``message`` explains the decision and is
    recorded in the run's hook events; ``additional_context`` (Session
    Start) is injected into the agent's messages.
    """

    action: str = "allow"
    message: str = ""
    additional_context: str = ""

    @classmethod
    def allow(cls, message: str = "") -> HookDecision:
        return cls(action="allow", message=message)

    @classmethod
    def deny(cls, message: str) -> HookDecision:
        return cls(action="deny", message=message)

    @classmethod
    def feedback(cls, message: str) -> HookDecision:
        return cls(action="feedback", message=message)

    @classmethod
    def continue_turn(cls, message: str) -> HookDecision:
        return cls(action="continue", message=message)

    @classmethod
    def block(cls, message: str) -> HookDecision:
        return cls(action="block", message=message)
