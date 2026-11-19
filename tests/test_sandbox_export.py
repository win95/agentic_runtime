"""Sandbox file export — no live sandbox."""

from __future__ import annotations

from types import SimpleNamespace

from runtime.sandbox.export import (
    MAX_FILE_BYTES,
    export_output_files,
    touch_export_run_marker,
)


class _FakeBackend:
    """Mirrors BaseSandbox.ls entries: path + is_dir only (no mtime)."""

    def __init__(self, *, entries=None, files=None, sizes=None, execute_output=None):
        self._entries = entries or {}
        self._files = files or {}
        self._sizes = sizes or {}
        self._execute_output = execute_output or {}
        self.download_calls: list[list[str]] = []
        self.execute_calls: list[str] = []

    def ls(self, path):
        return SimpleNamespace(
            entries=list(self._entries.get(path, [])),
            error=None,
        )

    def execute(self, command: str):
        self.execute_calls.append(command)
        for prefix, out in self._execute_output.items():
            if prefix in command:
                return SimpleNamespace(output=out, exit_code=0)
        if "stat -L" in command:
            before_redirect = command.split("2>/dev/null")[0].strip()
            path = before_redirect.rsplit(maxsplit=1)[-1].strip("'\"")
            size = self._sizes.get(path)
            if size is None and not path.startswith("/"):
                size = self._sizes.get(f"/{path}")
            if size is None:
                return SimpleNamespace(output="", exit_code=1)
            return SimpleNamespace(output=f"{size}\n", exit_code=0)
        if command.startswith("touch "):
            return SimpleNamespace(output="", exit_code=0)
        return SimpleNamespace(output="", exit_code=1)

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


def test_export_lists_both_output_paths_without_duplicate_list_error():
    backend = _FakeBackend(
        entries={
            "/output": [{"path": "/output/a.md", "is_dir": False}],
            "output": [],
        },
        files={"/output/a.md": b"# A"},
        sizes={"/output/a.md": 3},
    )
    files, notes = export_output_files(backend)
    assert len(files) == 1
    assert not any("could not list '/output'" in n for n in notes)


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


def test_export_skips_unknown_size_without_download():
    backend = _FakeBackend(
        entries={"/output": [{"path": "/output/unk.md", "is_dir": False}]},
        sizes={},
        files={"/output/unk.md": b"data"},
    )
    files, notes = export_output_files(backend)
    assert files == []
    assert not backend.download_calls
    assert any("could not determine file size" in n for n in notes)


def test_export_marker_filters_stale_files():
    marker = "/tmp/.run_export_exec-1"
    backend = _FakeBackend(
        execute_output={
            "find /output output": "/output/new.md\n",
        },
        files={"/output/new.md": b"fresh"},
        sizes={"/output/new.md": 5},
    )
    files, notes = export_output_files(backend, run_marker_path=marker)
    assert len(files) == 1
    assert files[0]["name"] == "new.md"
    assert any("find /output" in c for c in backend.execute_calls)


def test_touch_export_run_marker():
    backend = _FakeBackend()
    path = touch_export_run_marker(backend, "exec-abc")
    assert path and path.endswith("exec-abc")
    assert any(c.startswith("touch ") for c in backend.execute_calls)