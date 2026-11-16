"""Structured execution diagnostics (no string matching on error prose)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

DIAGNOSTIC_TYPE = "execution_diagnostic"

# Categories that explain why a run produced no collectible evidence.
EMPTY_EVIDENCE_CATEGORIES = frozenset(
    {
        "llm_unavailable",
        "agent_runtime_unavailable",
        "internal_bug",
        "router_aborted",
        "flow_aborted",
    }
)


@dataclass(frozen=True)
class ExecutionDiagnostic:
    category: str
    detail: str

    def to_json(self) -> str:
        return json.dumps(
            {"type": DIAGNOSTIC_TYPE, "category": self.category, "detail": self.detail},
            ensure_ascii=False,
        )


def parse_diagnostic_line(line: str) -> ExecutionDiagnostic | None:
    if not line or not isinstance(line, str):
        return None
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("type") != DIAGNOSTIC_TYPE:
        return None
    cat = data.get("category")
    detail = data.get("detail")
    if not isinstance(cat, str) or not cat.strip():
        return None
    return ExecutionDiagnostic(cat.strip(), str(detail or "")[:2000])


def classify_agent_failure(exc: BaseException) -> ExecutionDiagnostic:
    """Map an exception raised during agent/deep-agent setup or invoke."""
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
    except ImportError:  # pragma: no cover
        APIConnectionError = APITimeoutError = RateLimitError = ()

    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return ExecutionDiagnostic("llm_unavailable", str(exc))
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return ExecutionDiagnostic("llm_unavailable", str(exc))
    if isinstance(exc, RateLimitError):
        return ExecutionDiagnostic("llm_rate_limited", str(exc))
    if isinstance(exc, (TypeError, KeyError, AttributeError)):
        return ExecutionDiagnostic("internal_bug", type(exc).__name__)
    return ExecutionDiagnostic("agent_runtime_unavailable", type(exc).__name__)


def diagnostic_explains_empty_evidence(errors: list[str] | None) -> bool:
    for line in errors or []:
        diag = parse_diagnostic_line(line)
        if diag is not None and diag.category in EMPTY_EVIDENCE_CATEGORIES:
            return True
    return False


def append_diagnostic(errors: list[str], diagnostic: ExecutionDiagnostic) -> None:
    line = diagnostic.to_json()
    if line not in errors:
        errors.append(line)