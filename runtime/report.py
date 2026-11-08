"""Markdown execution report — pure function, no infra.

The worker saves the report to MinIO (bucket `reports`) after a run; if
MinIO is unreachable the report text is stored inline in the
ExecutionResult instead. Either way the same Markdown is produced here.
"""

from __future__ import annotations


def build_markdown_report(
    execution_id: str, mission: dict, input_data: dict, final_state: dict
) -> str:
    """Render one execution as Markdown from its final Mission Graph state."""
    result = final_state.get("result") or {}
    evidence = final_state.get("evidence") or []
    errors = final_state.get("errors") or []
    notes = final_state.get("notes") or []
    status = final_state.get("status", "UNKNOWN")
    root_causes = result.get("root_causes") or []
    recommendations = result.get("recommendations") or []
    lines = [
        f"# Execution report — {execution_id}",
        "",
        f"- Mission: `{mission.get('mission_id', '')}` (v{mission.get('version', '')})",
        f"- Status: **{status}**",
        f"- Input: `{input_data}`",
        "",
        "## Root causes",
        "",
    ]
    if root_causes:
        lines.extend(f"- {cause}" for cause in root_causes)
    else:
        lines.append("- (none identified)")
    lines.extend(["", "## Evidence", ""])
    if evidence:
        for item in evidence:
            summary = item.get("summary", "")
            source = item.get("source", "")
            kind = item.get("kind", "")
            stub = " (stub data)" if item.get("stub") else ""
            lines.append(f"- [{source}/{kind}]{stub} {summary}")
    else:
        lines.append("- (no evidence collected)")
    lines.extend(["", "## Recommendations", ""])
    if recommendations:
        lines.extend(f"- {rec}" for rec in recommendations)
    else:
        lines.append("- (none)")
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    return "\n".join(lines)
