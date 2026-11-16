"""Guest-safe execution error text (no raw tracebacks)."""

from __future__ import annotations

from runtime.execution_diagnostics import parse_diagnostic_line

_GENERIC_FAILED = "The request could not be completed. Please try again later."


def sanitize_execution_error(
    error: str | None,
    *,
    execution_input: dict | None = None,
) -> str | None:
    """Return a guest-safe error string for API reads."""
    if not error or not str(error).strip():
        return None
    text = str(error).strip()
    for line in text.split(";"):
        line = line.strip()
        if not line:
            continue
        diag = parse_diagnostic_line(line)
        if diag is not None:
            if diag.category in ("llm_unavailable", "llm_rate_limited"):
                return "The assistant is temporarily unavailable. Please try again."
            if diag.category in (
                "agent_runtime_unavailable",
                "router_aborted",
                "flow_aborted",
            ):
                return _GENERIC_FAILED
            if diag.category == "internal_bug":
                return _GENERIC_FAILED
            continue
        lower = line.lower()
        if lower.startswith("dispatch failed"):
            return "Could not start your request. Please try again."
        if "stale running" in lower:
            return _GENERIC_FAILED
    if _looks_like_traceback(text):
        return _GENERIC_FAILED
    if len(text) > 500:
        return _GENERIC_FAILED
    return text


def _looks_like_traceback(text: str) -> bool:
    if "Traceback (most recent call last)" in text:
        return True
    if 'File "' in text and ", line " in text:
        return True
    return False


def sanitize_from_execution(execution) -> str | None:
    inp = getattr(execution, "input", None) or {}
    channel = inp.get("channel") if isinstance(inp, dict) else None
    if channel != "guest_api":
        return getattr(execution, "error", None)
    return sanitize_execution_error(
        getattr(execution, "error", None),
        execution_input=inp if isinstance(inp, dict) else None,
    )