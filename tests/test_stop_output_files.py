"""stop_completion nudge when harness requires sandbox output files."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage

from runtime.hooks import builtin
from runtime.hooks.events import HookContext, HookEvent
from runtime.hooks.middleware import HookMiddleware
from runtime.hooks.registry import HookRegistry
from runtime.sandbox.export import has_exportable_files_since_marker


def _stop_ctx(**data) -> HookContext:
    return HookContext(data=dict(data))


def test_stop_completion_nudges_missing_files_with_tools_called():
    decision = builtin.stop_completion(
        _stop_ctx(
            tool_calls_made=3,
            continues=0,
            max_continues=3,
            output_files_required=True,
            output_files_status="missing",
        )
    )
    assert decision is not None and decision.action == "continue"
    assert "/output" in decision.message


def test_stop_completion_no_nudge_when_files_present():
    assert (
        builtin.stop_completion(
            _stop_ctx(
                tool_calls_made=1,
                continues=0,
                max_continues=3,
                output_files_required=True,
                output_files_status="present",
            )
        )
        is None
    )


def test_stop_completion_no_nudge_when_not_required():
    assert (
        builtin.stop_completion(
            _stop_ctx(
                tool_calls_made=0,
                continues=0,
                max_continues=3,
                output_files_required=False,
                output_files_status="missing",
            )
        )
        is not None
    )
    assert (
        builtin.stop_completion(
            _stop_ctx(
                tool_calls_made=2,
                continues=0,
                max_continues=3,
                output_files_required=False,
                output_files_status="missing",
            )
        )
        is None
    )


def test_stop_completion_no_nudge_when_backend_skipped():
    assert (
        builtin.stop_completion(
            _stop_ctx(
                tool_calls_made=2,
                continues=0,
                max_continues=3,
                output_files_required=True,
                output_files_status="skipped",
            )
        )
        is None
    )


def test_stop_completion_no_nudge_when_continues_exhausted():
    assert (
        builtin.stop_completion(
            _stop_ctx(
                tool_calls_made=2,
                continues=3,
                max_continues=3,
                output_files_required=True,
                output_files_status="missing",
            )
        )
        is None
    )


def test_stop_completion_check_failure_fail_open():
    assert (
        builtin.stop_completion(
            _stop_ctx(
                tool_calls_made=2,
                continues=0,
                max_continues=3,
                output_files_required=True,
                output_files_status="unknown",
            )
        )
        is None
    )


def test_has_exportable_files_since_run_fail_open_on_list_error():
    class _BrokenLs:
        def ls(self, _path):
            raise RuntimeError("down")

    backend = _BrokenLs()
    present, err = has_exportable_files_since_marker(backend, 1.0)
    assert present is None
    assert err


class _StopCheckBackend:
    def __init__(self, *, present: bool):
        self._present = present
        self._clock_skew = 0.0

    def has_exportable_files_since_run(self, run_started_at, clock_skew=0.0):  # noqa: ARG002
        return (self._present, None)


def _middleware_with_backend(backend) -> HookMiddleware:
    registry = HookRegistry()
    registry.register(HookEvent.STOP, builtin.stop_completion, name="stop_completion")
    mw = HookMiddleware(
        registry,
        execution_id="e1",
        mission_id="agent_analytis_mission",
        max_continues=3,
        output_files_required=True,
        sandbox_backend=backend,
        export_run_started_at=1000.0 if backend else None,
    )
    mw.shared_counters.increment_tool_calls()
    mw.shared_counters.increment_tool_calls()
    return mw


def test_middleware_after_model_nudges_missing_output_files():
    mw = _middleware_with_backend(_StopCheckBackend(present=False))
    out = mw.after_model({"messages": [AIMessage(content="Báo cáo sẽ được tạo")]}, None)
    assert out is not None and out.get("jump_to") == "model"
    assert "/output" in out["messages"][0].content
    assert mw.continues == 1


def test_middleware_after_model_no_nudge_when_files_present():
    mw = _middleware_with_backend(_StopCheckBackend(present=True))
    out = mw.after_model({"messages": [AIMessage(content="done")]}, None)
    assert out is None


def test_middleware_after_model_no_nudge_without_sandbox():
    mw = _middleware_with_backend(None)
    out = mw.after_model({"messages": [AIMessage(content="done")]}, None)
    assert out is None


def test_middleware_after_model_fail_open_on_check_error():
    class _FailBackend:
        def has_exportable_files_since_run(self, *_a, **_k):
            return None, "list failed"

    mw = _middleware_with_backend(_FailBackend())
    out = mw.after_model({"messages": [AIMessage(content="done")]}, None)
    assert out is None


def test_agent_analytis_harness_output_files_required():
    from pathlib import Path

    from runtime.harness.loader import load_mission_harness_from_file

    path = Path(__file__).resolve().parents[1] / "seeds/missions/agent_analytis_mission.yaml"
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    harness = load_mission_harness_from_file(path)
    assert data["harness"]["output_files"]["required"] is True
    assert harness.output_files.required is True


def test_hotel_reception_harness_no_output_files_required():
    from pathlib import Path

    from runtime.harness.loader import load_mission_harness_from_file

    path = (
        Path(__file__).resolve().parents[1] / "seeds/missions/hotel_reception.yaml"
    )
    harness = load_mission_harness_from_file(path)
    assert harness.output_files.required is False