"""Sandbox file export — no live sandbox."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from runtime.sandbox.export import (
    MAX_FILE_BYTES,
    export_output_files,
    run_start_timestamp,
)


class _FakeBackend:
    def __init__(self, *, entries=None, files=None, sizes=None, execute_output=None):
        self._entries = entries or {}
        self._files = files or {}
        self._sizes = sizes or {}
        self._execute_output = execute_output or {}
        self.download_calls: list[list[str]] = []

    def ls(self, path):
        return SimpleNamespace(
            entries=list(self._entries.get(path, [])),
            error=None,
        )

    def execute(self, command: str):
        for prefix, out in self._execute_output.items():
            if prefix in command:
                return SimpleNamespace(output=out, exit_code=0)
        path = command.split()[-1].strip("'\"")
        size = self._sizes.get(path)
        if size is None:
            return SimpleNamespace(output="-1\n", exit_code=1)
        return SimpleNamespace(output=f"{size}\n", exit_code=0)

    def download_files(self, paths):
        self.download_calls.append(list(paths))
        out = []
        for path in paths:
            data = self._files.get(path)
            if data is None:
                out.append(SimpleNamespace(path=path, error="missing", content=None))
            else:
                out.append(SimpleNamespace(path=path, error=None, content=data))
        return out


def test_export_none_backend_warns():
    files, notes = export_output_files(None)
    assert files == []
    assert notes and "unavailable" in notes[0].lower()


def test_export_lists_both_output_paths():
    backend = _FakeBackend(
        entries={
            "/output": [{"path": "/output/a.md", "is_dir": False}],
            "output": [{"path": "output/b.csv", "is_dir": False}],
        },
        files={
            "/output/a.md": b"# A",
            "output/b.csv": b"x,y",
        },
        sizes={"/output/a.md": 3, "output/b.csv": 3},
    )
    files, notes = export_output_files(backend)
    names = {f["name"] for f in files}
    assert names == {"a.md", "b.csv"}
    assert not notes or all("skipped" not in n for n in notes)


def test_export_skips_oversize_before_download():
    huge = MAX_FILE_BYTES + 1
    backend = _FakeBackend(
        entries={"/output": [{"path": "/output/huge.md", "is_dir": False}]},
        sizes={"/output/huge.md": huge},
        files={"/output/huge.md": b"x" * huge},
    )
    files, notes = export_output_files(backend)
    assert files == []
    assert not backend.download_calls
    assert any("skipped" in n and "huge.md" in n for n in notes)


def test_export_skips_stale_mtime():
    old = time.time() - 3600
    run_start = run_start_timestamp()
    backend = _FakeBackend(
        entries={
            "/output": [
                {
                    "path": "/output/old.md",
                    "is_dir": False,
                    "modified_at": old,
                }
            ]
        },
    )
    files, notes = export_output_files(backend, run_started_at=run_start)
    assert files == []
    assert any("before this run" in n for n in notes)