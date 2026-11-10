"""Markdown execution report — pure function, no infra.

The worker saves the report to MinIO (bucket `reports`) after a run; if
MinIO is unreachable the report text is stored inline in the
ExecutionResult instead. Either way the same Markdown is produced here.
"""

from __future__ import annotations

# Layer 5: at most this many evidence items render in the report.
MAX_EVIDENCE_ITEMS = 50


def build_markdown_report(
    execution_id: str, mission: dict, input_data: dict, final_state: dict
) -> str:
    """Render one execution as Markdown from its final Mission Graph state.

    Layer 5: only grounded root causes appear under "Root causes".
    Ungrounded causes and validation flags (injection signals, tools
    outside the allowed list) render under "Flagged (not in official
    findings)" so a reviewer sees them without them passing as findings.
    Evidence rendering is capped at MAX_EVIDENCE_ITEMS items.
    """
    result = final_state.get("result") or {}
    evidence = final_state.get("evidence") or []
    errors = final_state.get("errors") or []
    notes = final_state.get("notes") or []
    status = final_state.get("status", "UNKNOWN")
    root_causes = result.get("root_causes") or []
    recommendations = result.get("recommendations") or []
    ungrounded = result.get("ungrounded_root_causes") or []
    flags = result.get("validation_flags") or []
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
        shown = evidence[:MAX_EVIDENCE_ITEMS]
        for item in shown:
            summary = item.get("summary", "")
            source = item.get("source", "")
            kind = item.get("kind", "")
            stub = " (stub data)" if item.get("stub") else ""
            lines.append(f"- [{source}/{kind}]{stub} {summary}")
        if len(evidence) > MAX_EVIDENCE_ITEMS:
            lines.append(
                f"- … and {len(evidence) - MAX_EVIDENCE_ITEMS} more "
                "evidence item(s) not shown"
            )
    else:
        lines.append("- (no evidence collected)")
    lines.extend(["", "## Recommendations", ""])
    if recommendations:
        lines.extend(f"- {rec}" for rec in recommendations)
    else:
        lines.append("- (none)")
    if ungrounded or flags:
        lines.extend(["", "## Flagged (not in official findings)", ""])
        for cause in ungrounded:
            lines.append(f"- Ungrounded root cause (no supporting evidence): {cause}")
        for flag in flags:
            lines.append(f"- Validation flag: {flag}")
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    return "\n".join(lines)
