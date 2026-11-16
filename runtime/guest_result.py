"""Guest-safe projection for API responses (JWT guest channel)."""

from __future__ import annotations

from typing import Any

from runtime.guest_output import guard_guest_reply, strip_runtime_markup

GUEST_API_CHANNEL = "guest_api"

# Keys operators may see on full execution results; omitted for guests.
_GUEST_STRIP_OUTPUT_KEYS = frozenset(
    {
        "plan_trace",
        "tool_failures",
        "report_markdown",
        "report_location",
        "report_storage",
        "validation_flags",
        "ungrounded_root_causes",
        "root_causes",
        "recommendations",
        "evidence",
        "summary",
        "errors",
        "visited_nodes",
        "flow_trace",
    }
)


def is_guest_api_execution(execution_input: dict | None) -> bool:
    return (execution_input or {}).get("channel") == GUEST_API_CHANNEL


def project_guest_result(
    output: dict | None,
    evidence: list | None,
    *,
    session_id: str | None = None,
    execution_input: dict | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Return a guest-safe (output, evidence) pair."""
    raw = dict(output or {})
    answer = (
        raw.get("answer")
        or raw.get("summary")
        or ""
    )
    answer = strip_runtime_markup(str(answer))
    cleaned, _ = guard_guest_reply(
        answer,
        context_blobs=[],
        session_id=session_id,
        input_data=execution_input,
    )
    safe_output: dict[str, Any] = {
        "answer": cleaned,
    }
    actions = raw.get("actions_taken")
    if isinstance(actions, list):
        safe_output["actions_taken"] = [str(a) for a in actions if a]
    safe_evidence: list[Any] = []
    if cleaned:
        safe_evidence.append(
            {
                "source": "guest",
                "kind": "guest_reply",
                "summary": cleaned,
                "stub": False,
            }
        )
    return safe_output, safe_evidence


def apply_guest_result_to_state(
    result: dict,
    evidence: list,
    *,
    execution_input: dict | None,
    session_id: str | None,
) -> tuple[dict, list]:
    """Persist guest-safe shapes when the run used the guest API channel."""
    if not is_guest_api_execution(execution_input):
        return result, evidence
    safe_out, safe_ev = project_guest_result(
        result,
        evidence,
        session_id=session_id,
        execution_input=execution_input,
    )
    merged = {k: v for k, v in result.items() if k not in _GUEST_STRIP_OUTPUT_KEYS}
    merged.update(safe_out)
    return merged, safe_ev