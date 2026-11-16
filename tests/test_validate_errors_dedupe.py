"""Validation errors — dedupe and no spurious empty-evidence when diagnostics exist."""

from runtime.execution_diagnostics import ExecutionDiagnostic
from runtime.mission_graph import validate_results


def test_no_evidence_suppressed_when_llm_unavailable_diagnostic():
    diag = ExecutionDiagnostic("llm_unavailable", "Connection error.").to_json()
    state = {
        "errors": [diag, diag],
        "evidence": [],
        "result": {},
        "mission": {"output_schema": {"properties": {}}},
    }
    out = validate_results(state)
    assert "No evidence collected" not in out["errors"]
    assert out["errors"].count(diag) == 1