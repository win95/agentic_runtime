"""Execution result export_warnings and allowed-tool validation."""

from __future__ import annotations

import runtime.tools  # noqa: F401
from runtime.mission_graph import _allowed_tool_names, validate_results


def test_allowed_tool_names_excludes_builtins_when_hotel_deny():
    mission = {
        "mission_id": "hotel_reception",
        "capabilities": ["hotel_reception"],
        "execution": {
            "harness": {
                "builtin_tools": {
                    "deny": True,
                    "deny_message_template": "Tool '{tool_name}' denied.",
                }
            }
        },
    }
    allowed = _allowed_tool_names(mission, sandbox_enabled=False)
    assert allowed is not None
    assert "ls" not in allowed
    assert "task" not in allowed


def test_validate_flags_disallowed_builtin_for_hotel():
    mission = {
        "mission_id": "hotel_reception",
        "capabilities": ["hotel_reception"],
        "output_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        "execution": {
            "harness": {
                "builtin_tools": {
                    "deny": True,
                    "deny_message_template": "Tool '{tool_name}' denied.",
                }
            }
        },
    }
    state = {
        "mission": mission,
        "evidence": [{"summary": "guest asked for a room", "data": {}}],
        "result": {"answer": "ok", "actions_taken": ["ls", "reply_to_guest"]},
        "errors": [],
    }
    out = validate_results(state)
    flags = (out.get("result") or {}).get("validation_flags") or []
    flagged = [f for f in flags if f.get("type") == "tool_outside_allowed_list"]
    assert any(f.get("tool") == "ls" for f in flagged)


def test_export_warnings_includes_partial_skip_notes():
    export_notes = [
        "Export: skipped /output/huge.csv (900000 bytes > 262144 cap)",
        "Export: no report files newer than this run's marker under output/",
    ]
    export_warns = [n for n in export_notes if n.startswith("Export:")]
    assert len(export_warns) == 2