"""Observability failures must not fail guest executions."""

from runtime.observability_resilience import (
    filter_observability_errors,
    invoke_with_observability_fallback,
    is_observability_transport_error,
)


def test_langsmith_429_is_observability_only():
    exc = Exception(
        "Client error '429 Too Many Requests' for url "
        "'https://api.smith.langchain.com/runs/multipart'"
    )
    assert is_observability_transport_error(exc)


def test_invoke_retries_without_callbacks():
    calls = []

    class Stub:
        def invoke(self, payload, config=None):
            calls.append(dict(config or {}))
            if len(calls) == 1:
                raise Exception("429 for https://api.smith.langchain.com/runs/multipart")
            return {"ok": True}

    out = invoke_with_observability_fallback(Stub().invoke, {"x": 1}, config={"callbacks": [object()]})
    assert out == {"ok": True}
    assert calls[1].get("callbacks") == []


def test_filter_observability_errors():
    errs = [
        "Router declared no flow (no FLOW: line); ending turn",
        "429 for api.smith.langchain.com/runs/multipart",
    ]
    assert filter_observability_errors(errs) == [errs[0]]