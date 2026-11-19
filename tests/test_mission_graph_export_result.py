"""Execution result export_warnings and allowed-tool validation."""

from __future__ import annotations

import runtime.tools  # noqa: F401
from runtime.mission_graph import (
    _allowed_tool_names,
    build_export_result_warnings,
    validate_results,
)


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


def test_build_export_result_warnings_partial_and_empty():
    notes = [
        "Export: skipped /output/huge.csv (900000 bytes > 262144 cap)",
        "other note",
    ]
    assert build_export_result_warnings([{"name": "a.csv"}], notes) == [
        "Export: skipped /output/huge.csv (900000 bytes > 262144 cap)",
    ]
    assert build_export_result_warnings([], notes) == [
        "Export: skipped /output/huge.csv (900000 bytes > 262144 cap)",
    ]
    empty = build_export_result_warnings([], [])
    assert empty == ["Export: output schema requires files but none were exported"]
    assert build_export_result_warnings([{"name": "ok"}], []) is None