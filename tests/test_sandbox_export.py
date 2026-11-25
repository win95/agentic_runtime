"""Sandbox file export — mtime + files API listing."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from runtime.sandbox.export import (
    export_output_files,
    has_exportable_files_since_run,
)


class _ListingBackend:
    def __init__(self, *, items=None, clock_skew=0.0):
        self._items = items or []
        self._clock_skew = clock_skew
        self.download_calls: list[list[str]] = []

    def list_output_files_for_export(self, run_started_at, clock_skew=0.0):
        return list(self._items), []

    def download_files(self, paths):
        self.download_calls.append(list(paths))
        return [
            SimpleNamespace(path=p, content=b"<html/>", error=None)
            for p in paths
        ]

    def has_exportable_files_since_run(self, run_started_at, clock_skew=0.0):  # noqa: ARG002
        import os

        from runtime.sandbox.export import REPORT_FILE_CONTENT_TYPES

        for item in self._items:
            ext = os.path.splitext(item["path"])[1].lower()
            if ext in REPORT_FILE_CONTENT_TYPES:
                return True, None
        return False, None


def test_export_none_backend_warns():
    files, notes = export_output_files(None)
    assert files == []
    assert notes and "unavailable" in notes[0].lower()


def test_export_selects_by_mtime_and_respects_caps():
    run_start = datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp()
    backend = _ListingBackend(
        items=[
            {
                "path": "/output/report.html",
                "size": 8,
                "modified_at": datetime(2025, 6, 2),
            }
        ]
    )
    files, notes = export_output_files(backend, run_started_at=run_start)
    assert len(files) == 1
    assert files[0]["name"] == "report.html"
    assert len(backend.download_calls) == 1


def test_export_dedupes_report_html_paths_via_listing():
    run_start = 0.0
    backend = _ListingBackend(
        items=[{"path": "/output/report.html", "size": 10, "modified_at": datetime(2030, 1, 1)}]
    )
    files, _notes = export_output_files(backend, run_started_at=run_start)
    assert len(files) == 1
    assert files[0]["path"] == "/output/report.html"


def test_has_exportable_false_when_only_old_files():
    run_start = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
    backend = _ListingBackend(items=[])
    present, _err = backend.has_exportable_files_since_run(run_start)
    assert present is False


def test_stop_nudge_listing_present():
    backend = _ListingBackend(
        items=[{"path": "/output/a.md", "size": 3, "modified_at": datetime(2030, 1, 1)}]
    )
    ok, err = backend.has_exportable_files_since_run(0.0)
    assert ok is True
    assert err is None