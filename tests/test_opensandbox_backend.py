"""OpenSandboxBackend and command output parsing."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from runtime.sandbox.command_output import join_stream_chunks, split_command_streams
from runtime.sandbox.opensandbox_backend import (
    OpenSandboxBackend,
    format_command_output,
)


def test_join_stream_preserves_blank_and_partial_lines():
    stream = ["line1\n", "\n", "part"]
    assert join_stream_chunks(stream) == "line1\n\npart"


def test_split_stderr_kept_separate_until_combine():
    execution = SimpleNamespace(
        logs=SimpleNamespace(stdout=["ok"], stderr=["warn"]),
        exit_code=1,
        result=[],
        error=None,
    )
    out, err = split_command_streams(execution)
    assert out == "ok"
    assert err == "warn"
    assert format_command_output(execution) == "ok\nwarn"


def test_format_command_output_success_no_stderr():
    execution = SimpleNamespace(
        logs=SimpleNamespace(stdout=["hi", "report.html"], stderr=[]),
        result=[],
        exit_code=0,
        error=None,
    )
    assert format_command_output(execution) == "hireport.html"


class _FakeCommands:
    def __init__(self, executions: list):
        self._executions = list(executions)
        self.last_opts = None
        self.calls = 0

    def run(self, command, opts=None):
        self.calls += 1
        self.last_opts = opts
        if self._executions:
            item = self._executions.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        raise RuntimeError("no more executions")


class _FakeFiles:
    def __init__(self, entries=None):
        self.entries = entries or []

    def list_directory(self, entry):  # noqa: ARG002
        return self.entries

    def read_file(self, path, *, offset=1, limit=2000):  # noqa: ARG002
        return "line1\nline2\n"

    def search(self, entry):  # noqa: ARG002
        return [SimpleNamespace(path="/workspace/foo.py")]

    def read_bytes(self, path):  # noqa: ARG002
        return b"data"


def _backend(commands, files=None, **kwargs) -> OpenSandboxBackend:
    sandbox = SimpleNamespace(commands=commands, files=files or _FakeFiles())
    return OpenSandboxBackend(sandbox, "sb-1", **kwargs)


def test_execute_always_sets_working_directory_and_timeout():
    cmd = _FakeCommands(
        [
            SimpleNamespace(
                logs=SimpleNamespace(stdout=["done"], stderr=[]),
                exit_code=0,
                result=[],
                error=None,
            )
        ]
    )
    backend = _backend(cmd, command_timeout_seconds=90)
    resp = backend.execute("echo hi", timeout=30)
    assert resp.exit_code == 0
    assert cmd.last_opts.working_directory == "/workspace"
    assert cmd.last_opts.timeout.total_seconds() == 30


def test_execute_timeout_exit_code_none():
    cmd = _FakeCommands(
        [
            SimpleNamespace(
                logs=SimpleNamespace(stdout=[], stderr=[]),
                exit_code=None,
                result=[],
                error=None,
            )
        ]
    )
    resp = _backend(cmd).execute("grep -r x .")
    assert resp.exit_code == 124
    assert "timed out" in resp.output.lower()


def test_sandbox_not_found_recreate_once_then_unavailable():
    recreated = SimpleNamespace(
        logs=SimpleNamespace(stdout=["ok"], stderr=[]),
        exit_code=0,
        result=[],
        error=None,
    )
    cmd = _FakeCommands(
        [
            RuntimeError("DOCKER::SANDBOX_NOT_FOUND"),
            recreated,
            RuntimeError("DOCKER::SANDBOX_NOT_FOUND"),
        ]
    )

    def recreate():
        new_cmd = _FakeCommands(
            [
                SimpleNamespace(
                    logs=SimpleNamespace(stdout=["ok"], stderr=[]),
                    exit_code=0,
                    result=[],
                    error=None,
                )
            ]
        )
        return (SimpleNamespace(commands=new_cmd, files=_FakeFiles()), "sb-2")

    backend = _backend(cmd, recreate_callback=recreate)
    first = backend.execute("echo 1")
    assert first.exit_code == 0
    assert backend.id == "sb-2"
    backend._sandbox.commands._executions.append(RuntimeError("DOCKER::SANDBOX_NOT_FOUND"))
    second = backend.execute("echo 2")
    assert second.exit_code == 127
    assert "unavailable" in second.output.lower()


def test_ls_uses_list_directory():
    entry = SimpleNamespace(
        path="/output/a.html",
        entry_type="file",
        size=10,
        modified_at=datetime.now(timezone.utc),
    )
    files = _FakeFiles(entries=[entry])
    backend = _backend(_FakeCommands([]), files=files)
    result = backend.ls("/output")
    assert result.error is None
    assert result.entries[0]["path"] == "/output/a.html"


def test_export_listing_filters_by_mtime():
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new = datetime(2030, 1, 1, tzinfo=timezone.utc)
    entries = [
        SimpleNamespace(
            path="/output/old.html",
            entry_type="file",
            size=3,
            modified_at=old,
        ),
        SimpleNamespace(
            path="/output/report.html",
            entry_type="file",
            size=8,
            modified_at=new,
        ),
    ]
    backend = _backend(_FakeCommands([]), files=_FakeFiles(entries=entries))
    run_start = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
    items, _notes = backend.list_output_files_for_export(run_start, clock_skew=0.0)
    assert [i["path"] for i in items] == ["/output/report.html"]


def test_has_exportable_files_since_run():
    new = datetime(2030, 1, 1)
    entries = [
        SimpleNamespace(
            path="/output/report.html",
            entry_type="file",
            size=8,
            modified_at=new,
        )
    ]
    backend = _backend(_FakeCommands([]), files=_FakeFiles(entries=entries))
    ok, err = backend.has_exportable_files_since_run(0.0, 0.0)
    assert ok is True
    assert err is None