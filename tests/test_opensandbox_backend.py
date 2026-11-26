"""OpenSandboxBackend and command output parsing."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from runtime.sandbox.command_output import join_stream_chunks, split_command_streams
from runtime.sandbox.opensandbox_backend import (
    BOOTSTRAP_WORKSPACE_SHELL,
    OpenSandboxBackend,
    bootstrap_sandbox_workspace,
    format_command_output,
)
from runtime.sandbox.paths import normalize_output_export_path


def test_join_stream_joins_sdk_line_chunks_with_newlines():
    assert join_stream_chunks(["1", "2", "3"]) == "1\n2\n3"
    assert join_stream_chunks(["line1", "", "part"]) == "line1\n\npart"


def test_join_stream_single_chunk_no_trailing_newline():
    assert join_stream_chunks(["x"]) == "x"
    assert join_stream_chunks([]) == ""


def test_join_stream_seq_lines_realistic_length():
    chunks = [str(i) for i in range(1, 6)]
    joined = join_stream_chunks(chunks)
    assert joined == "1\n2\n3\n4\n5"
    assert len(joined) == 9


def test_split_stderr_appended_on_success():
    execution = SimpleNamespace(
        logs=SimpleNamespace(stdout=["ok"], stderr=["warn"]),
        exit_code=0,
        result=[],
        error=None,
    )
    out, err = split_command_streams(execution)
    assert out == "ok"
    assert err == "warn"
    assert format_command_output(execution) == "ok\nwarn"


def test_split_stderr_on_failure():
    execution = SimpleNamespace(
        logs=SimpleNamespace(stdout=["ok"], stderr=["warn"]),
        exit_code=1,
        result=[],
        error=None,
    )
    assert format_command_output(execution) == "ok\nwarn"


def test_format_command_output_success_multiline_stdout():
    execution = SimpleNamespace(
        logs=SimpleNamespace(stdout=["hi", "report.html"], stderr=[]),
        result=[],
        exit_code=0,
        error=None,
    )
    assert format_command_output(execution) == "hi\nreport.html"


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
    def __init__(self, entries=None, *, reject_symlink=False):
        self.entries = entries or []
        self.reject_symlink = reject_symlink
        self.created: list = []
        self.list_paths: list[str] = []
        self.read_paths: list[str] = []
        self.write_paths: list[str] = []
        self.search_paths: list[str] = []

    def create_directories(self, entries):
        self.created.extend(entries)

    def list_directory(self, entry):
        path = getattr(entry, "path", "")
        self.list_paths.append(path)
        if self.reject_symlink and path == "/workspace/output":
            raise RuntimeError(
                "path is a symbolic link, refusing to traverse: /workspace/output"
            )
        return self.entries

    def read_file(self, path, *, offset=1, limit=2000):  # noqa: ARG002
        self.read_paths.append(path)
        if offset > 2:
            return ""
        return "line1\nline2\n"

    def search(self, entry):
        self.search_paths.append(entry.path)
        return [SimpleNamespace(path="/workspace/foo.py")]

    def read_bytes(self, path):
        self.read_paths.append(path)
        return b"data"

    def write_files(self, entries):
        for ent in entries:
            self.write_paths.append(ent.path)


def _backend(commands, files=None, **kwargs) -> OpenSandboxBackend:
    sandbox = SimpleNamespace(commands=commands, files=files or _FakeFiles())
    return OpenSandboxBackend(
        sandbox,
        "sb-1",
        command_timeout_max_seconds=kwargs.pop("command_timeout_max_seconds", 600),
        **kwargs,
    )


def _ok_exec(stdout=None):
    return SimpleNamespace(
        logs=SimpleNamespace(stdout=stdout or ["done"], stderr=[]),
        exit_code=0,
        result=[],
        error=None,
    )


def test_execute_always_sets_working_directory_and_timeout():
    cmd = _FakeCommands([_ok_exec()])
    backend = _backend(cmd, command_timeout_seconds=90, command_timeout_max_seconds=600)
    resp = backend.execute("echo hi", timeout=30)
    assert resp.exit_code == 0
    assert cmd.last_opts.working_directory == "/workspace"
    assert cmd.last_opts.timeout.total_seconds() == 30


def test_execute_model_timeout_capped_at_harness_max():
    cmd = _FakeCommands([_ok_exec()])
    backend = _backend(cmd, command_timeout_seconds=90, command_timeout_max_seconds=120)
    backend.execute("sleep 10", timeout=999)
    assert cmd.last_opts.timeout.total_seconds() == 120


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


def test_working_directory_error_does_not_mark_unavailable():
    cmd = _FakeCommands(
        [RuntimeError("working directory does not exist: /workspace")]
    )
    backend = _backend(cmd)
    resp = backend.execute("true")
    assert resp.exit_code == 127
    assert not backend._unavailable


def test_bootstrap_uses_root_working_directory():
    cmd = _FakeCommands([_ok_exec()])
    files = _FakeFiles()
    backend = _backend(cmd, files=files)
    bootstrap_sandbox_workspace(backend)
    assert files.created
    assert cmd.last_opts.working_directory == "/"


def test_sandbox_not_found_recreate_once_then_unavailable():
    cmd = _FakeCommands([RuntimeError("DOCKER::SANDBOX_NOT_FOUND")])

    def recreate():
        new_cmd = _FakeCommands([_ok_exec(), _ok_exec()])
        return (SimpleNamespace(commands=new_cmd, files=_FakeFiles()), "sb-2")

    backend = _backend(cmd, recreate_callback=recreate)
    first = backend.execute("echo 1")
    assert first.exit_code == 0
    assert backend.id == "sb-2"
    backend._sandbox.commands._executions.append(RuntimeError("DOCKER::SANDBOX_NOT_FOUND"))
    second = backend.execute("echo 2")
    assert second.exit_code == 127
    assert "unavailable" in second.output.lower()


def test_bootstrap_shell_moves_legacy_workspace_output_dir():
    assert "mv " in BOOTSTRAP_WORKSPACE_SHELL
    assert "[ -d /workspace/output ]" in BOOTSTRAP_WORKSPACE_SHELL


def test_ls_workspace_output_hits_real_output_dir():
    files = _FakeFiles(reject_symlink=True, entries=[])
    backend = _backend(_FakeCommands([]), files=files)
    backend.ls("/workspace/output")
    assert files.list_paths == ["/output"]


def test_write_relative_output_resolves_to_real_path():
    files = _FakeFiles()
    backend = _backend(_FakeCommands([_ok_exec()]), files=files)
    backend.upload_files([("output/a.html", b"<html/>")])
    assert files.write_paths == ["/output/a.html"]


def test_read_workspace_output_file_resolves():
    files = _FakeFiles()
    backend = _backend(_FakeCommands([]), files=files)
    backend.read("/workspace/output/a.html")
    assert files.read_paths == ["/output/a.html"]


def test_export_mtime_compares_utc_not_strip_tz():
    from datetime import timedelta as td

    # Run starts 2025-06-01 10:00 UTC
    run_start = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    # File at 2025-06-01 17:05 +07:00 == 10:05 UTC (after run start)
    mod = datetime(2025, 6, 1, 17, 5, 0, tzinfo=timezone(td(hours=7)))
    entries = [
        SimpleNamespace(
            path="/output/report.html",
            entry_type="file",
            size=8,
            modified_at=mod,
        ),
    ]
    backend = _backend(_FakeCommands([]), files=_FakeFiles(entries=entries))
    items, _ = backend.list_output_files_for_export(run_start, clock_skew=0.0)
    assert [i["path"] for i in items] == ["/output/report.html"]


def test_ls_uses_list_directory_real_output_dir():
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


def test_export_listing_rejects_symlink_workspace_path():
    assert normalize_output_export_path("/workspace/output/x.html") == "/output/x.html"


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


def test_glob_defaults_to_workspace():
    files = _FakeFiles()
    cmd = _FakeCommands([])
    backend = _backend(cmd, files=files)

    class _CaptureFiles(_FakeFiles):
        def search(self, entry):
            self.last = entry
            return []

    cap_files = _CaptureFiles()
    backend._sandbox.files = cap_files
    backend.glob("*.py", path=None)
    assert cap_files.last.path == "/workspace"


def test_read_eof_no_phantom_line():
    backend = _backend(_FakeCommands([]))
    result = backend.read("/workspace/foo.py", offset=5, limit=10)
    assert result.error is None
    assert result.start_line is None
    content = getattr(result.file_data, "content", None) or result.file_data.get("content")
    assert content == ""