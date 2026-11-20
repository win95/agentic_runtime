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


def test_has_exportable_files_since_marker_fail_open_on_execute_missing():
    backend = SimpleNamespace()
    present, err = has_exportable_files_since_marker(backend, "/tmp/.run_x")
    assert present is None
    assert err


class _MarkerBackend:
    def __init__(self, *, find_output: str = "", fail_find: bool = False):
        self._find_output = find_output
        self._fail_find = fail_find

    def execute(self, command: str):
        if "find /output" in command:
            if self._fail_find:
                raise RuntimeError("sandbox down")
            return SimpleNamespace(output=self._find_output, exit_code=0)
        if "readlink" in command or "realpath" in command:
            before_redirect = command.split("2>/dev/null")[0].strip()
            path = before_redirect.rsplit(maxsplit=1)[-1].strip("'\"")
            canon = path if path.startswith("/") else f"/{path}"
            return SimpleNamespace(output=f"{canon}\n", exit_code=0)
        return SimpleNamespace(output="", exit_code=1)


def _middleware_with_backend(backend, *, find_output: str = "") -> HookMiddleware:
    if backend is None:
        b = None
        marker = None
    else:
        b = backend if isinstance(backend, _MarkerBackend) else backend
        if not isinstance(b, _MarkerBackend):
            b = _MarkerBackend(find_output=find_output)
        else:
            b = backend
        marker = "/tmp/.run_e1"
    registry = HookRegistry()
    registry.register(HookEvent.STOP, builtin.stop_completion, name="stop_completion")
    mw = HookMiddleware(
        registry,
        execution_id="e1",
        mission_id="agent_analytis_mission",
        max_continues=3,
        output_files_required=True,
        sandbox_backend=b,
        export_marker_path=marker,
    )
    mw.shared_counters.increment_tool_calls()
    mw.shared_counters.increment_tool_calls()
    return mw


def test_middleware_after_model_nudges_missing_output_files():
    mw = _middleware_with_backend(_MarkerBackend(find_output=""))
    out = mw.after_model({"messages": [AIMessage(content="Báo cáo sẽ được tạo")]}, None)
    assert out is not None and out.get("jump_to") == "model"
    assert "/output" in out["messages"][0].content
    assert mw.continues == 1


def test_middleware_after_model_no_nudge_when_files_present():
    mw = _middleware_with_backend(_MarkerBackend(find_output="/output/report.md\n"))
    out = mw.after_model({"messages": [AIMessage(content="done")]}, None)
    assert out is None


def test_middleware_after_model_no_nudge_without_sandbox():
    mw = _middleware_with_backend(None)
    out = mw.after_model({"messages": [AIMessage(content="done")]}, None)
    assert out is None


def test_middleware_after_model_fail_open_on_check_error():
    mw = _middleware_with_backend(_MarkerBackend(fail_find=True))
    out = mw.after_model({"messages": [AIMessage(content="done")]}, None)
    assert out is None


def test_agent_analytis_harness_output_files_required():
    from runtime.harness.loader import load_mission_harness_from_file
    from pathlib import Path

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