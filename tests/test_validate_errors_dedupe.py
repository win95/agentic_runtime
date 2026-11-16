"""Validation errors — dedupe and no spurious empty-evidence when diagnostics exist."""

from runtime.mission_graph import validate_results


def test_no_evidence_suppressed_when_deep_agent_unavailable():
    state = {
        "errors": [
            "Deep agent unavailable: Connection error.",
            "Deep agent unavailable: Connection error.",
        ],
        "evidence": [],
        "result": {},
        "mission": {"output_schema": {"properties": {}}},
    }
    out = validate_results(state)
    assert "No evidence collected" not in out["errors"]
    assert out["errors"].count("Deep agent unavailable: Connection error.") == 1