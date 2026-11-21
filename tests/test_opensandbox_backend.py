"""OpenSandboxBackend shell output parsing."""

from __future__ import annotations

from types import SimpleNamespace

from runtime.sandbox.opensandbox_backend import OpenSandboxBackend, format_command_output


def test_format_command_output_prefers_logs_stdout_stderr():
    execution = SimpleNamespace(
        logs=SimpleNamespace(
            stdout=[SimpleNamespace(text="hi"), "report.html"],
            stderr=[SimpleNamespace(text="warn-line")],
        ),
        result=[],
        exit_code=0,
        error=None,
    )
    assert format_command_output(execution) == "hi\nreport.html\nwarn-line"


def test_format_command_output_falls_back_to_result():
    execution = SimpleNamespace(
        logs=None,
        result=[SimpleNamespace(text="legacy-only\n")],
        exit_code=0,
        error=None,
    )
    assert format_command_output(execution) == "legacy-only\n"


def test_execute_uses_logs_stdout():
    class _Commands:
        def run(self, command, opts=None):  # noqa: ARG002
            return SimpleNamespace(
                logs=SimpleNamespace(stdout=["echoed"], stderr=[]),
                result=[],
                exit_code=0,
                error=None,
            )

    backend = OpenSandboxBackend(SimpleNamespace(commands=_Commands()), "sb-1")
    resp = backend.execute("echo hi")
    assert resp.output == "echoed"
    assert resp.exit_code == 0