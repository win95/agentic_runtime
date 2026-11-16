"""Guest API result projection — no internal trace in guest responses."""

from runtime.guest_result import (
    is_guest_api_execution,
    project_guest_result,
)


def test_is_guest_api_channel():
    assert is_guest_api_execution({"channel": "guest_api", "message": "hi"})
    assert not is_guest_api_execution({"channel": "api"})
    assert not is_guest_api_execution(None)


def test_project_guest_result_strips_internal_fields():
    output, evidence = project_guest_result(
        {
            "answer": (
                '<data source="trace">{"step":"done"}</data>\n\n'
                "Xin chào khách."
            ),
            "plan_trace": {"reception": {"visited": ["policy-lookup"]}},
            "tool_failures": ["x"],
        },
        [
            {
                "source": "deep_agent",
                "kind": "summary",
                "summary": '<data source="execution_input">{"guest_id":"g1"}',
            }
        ],
        execution_input={"guest_id": "g1", "channel": "guest_api"},
    )
    assert "plan_trace" not in output
    assert "tool_failures" not in output
    assert "<data" not in output["answer"]
    assert "Xin chào khách" in output["answer"]
    assert len(evidence) == 1
    assert evidence[0]["kind"] == "guest_reply"
    assert "<data" not in evidence[0]["summary"]