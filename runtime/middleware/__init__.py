"""Middleware — LangChain callbacks for token + audit logging (Layer 4).

This is our self-hosted equivalent of the "middleware" piece in
LangSmith Managed Deep Agents: logic that runs around model and tool
calls. Two jobs:

- token logging (``LLMUsage``): every LLM call's usage as reported by
  the provider — tokens ONLY, no cost (see README section 13);
- audit log (``AuditLog``): every tool call, with secrets redacted.

One handler instance is built per run with that run's execution context
(``deep_agent_node`` constructs it) — no globals, nothing leaks across
concurrent runs. All DB writes are best-effort: a logging failure is
logged and swallowed, never raised, so middleware can never fail an
execution.
"""

from __future__ import annotations

from runtime.middleware.callbacks import (
    MAX_ARGS_CHARS,
    TokenAuditCallbackHandler,
    redact_value,
    truncate_args,
)
from runtime.middleware.queries import (
    get_audit_for_execution,
    get_tool_call_counts,
    get_usage_for_execution,
    get_usage_totals,
)

__all__ = [
    "MAX_ARGS_CHARS",
    "TokenAuditCallbackHandler",
    "get_audit_for_execution",
    "get_tool_call_counts",
    "get_usage_for_execution",
    "get_usage_totals",
    "redact_value",
    "truncate_args",
]
