"""Export the agent's report files out of its session sandbox."""

from __future__ import annotations

import logging
import os

OUTPUT_DIR = "/output"
MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 1024 * 1024

REPORT_FILE_CONTENT_TYPES = {
    ".html": "text/html",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
}


def content_type_for_export_filename(filename: str) -> str:
    """MIME type for a sandbox report filename (export + artifact upload)."""
    ext = os.path.splitext(filename)[1].lower()
    return REPORT_FILE_CONTENT_TYPES.get(ext, "application/octet-stream")

logger = logging.getLogger(__name__)


def _is_toplevel_output_path(path: str) -> bool:
    norm = path if path.startswith("/") else f"/{path}"
    parts = norm.strip("/").split("/")
    return len(parts) == 2 and parts[0] == "output"


def _list_output_via_ls(
    backend, run_started_at: float | None
) -> tuple[list[dict], list[str]]:
    """Fallback when backend has no files API listing."""
    notes: list[str] = []
    items: list[dict] = []
    ls = getattr(backend, "ls", None)
    if not ls:
        return [], ["Export: sandbox backend cannot list output/"]
    for directory in (OUTPUT_DIR, "output", "/workspace/output"):
        try:
            listing = ls(directory)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Export: could not list '{directory}' ({exc})")
            continue
        if getattr(listing, "error", None):
            notes.append(f"Export: could not list '{directory}' ({listing.error})")
            continue
        for entry in getattr(listing, "entries", None) or []:
            if entry.get("is_dir"):
                continue
            path = str(entry.get("path") or "")
            if not path or not _is_toplevel_output_path(path):
                continue
            norm = path if path.startswith("/") else f"/{path}"
            size = entry.get("size")
            items.append({"path": norm, "size": size})
    if run_started_at is not None:
        notes.append(
            "Export: run start filter unavailable without OpenSandbox files API; "
            "listing all top-level output files"
        )
    if not items and not notes:
        notes.append("Export: no report files found under output/")
    return items, notes


def _list_candidates(
    backend, run_started_at: float | None, clock_skew: float
) -> tuple[list[dict], list[str]]:
    list_fn = getattr(backend, "list_output_files_for_export", None)
    if list_fn is not None and run_started_at is not None:
        return list_fn(run_started_at, clock_skew)
    if list_fn is not None and run_started_at is None:
        return list_fn(0.0, clock_skew)
    return _list_output_via_ls(backend, run_started_at)


def has_exportable_files_since_run(
    backend,
    run_started_at: float,
    clock_skew: float = 0.0,
) -> tuple[bool | None, str | None]:
    """Whether any allowed report file exists newer than this run's start."""
    check = getattr(backend, "has_exportable_files_since_run", None)
    if check is not None:
        try:
            return check(run_started_at, clock_skew)
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)
    try:
        items, notes = _list_candidates(backend, run_started_at, clock_skew)
        for note in notes:
            if "could not list" in note.lower():
                return None, note
        for item in items:
            ext = os.path.splitext(item["path"])[1].lower()
            if ext in REPORT_FILE_CONTENT_TYPES:
                return True, None
        return False, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# Back-compat alias for tests still naming marker API
has_exportable_files_since_marker = has_exportable_files_since_run


def export_output_files(
    backend,
    *,
    directory: str | None = None,
    run_started_at: float | None = None,
    clock_skew: float = 0.0,
    run_marker_path: str | None = None,  # noqa: ARG001 — legacy, unused
    marker_create_failed: bool = False,  # noqa: ARG001 — legacy
):
    """Download report files from sandbox ``output`` (mtime >= run start when known)."""
    _ = directory
    notes: list[str] = []
    if backend is None:
        return [], notes + [
            "Export: sandbox backend unavailable — no files exported "
            "(in-memory fallback cannot export; enable SANDBOX_* or see notes)"
        ]
    clock_skew = float(
        clock_skew
        if clock_skew
        else getattr(backend, "_clock_skew", 0.0)
    )
    if run_started_at is None:
        run_started_at = getattr(backend, "_run_started_at", None)

    items, list_notes = _list_candidates(backend, run_started_at, clock_skew)
    notes.extend(list_notes)
    filtered: list[dict] = []
    skipped_ext: list[str] = []
    for item in items:
        path = item["path"]
        ext = os.path.splitext(path)[1].lower()
        if ext in REPORT_FILE_CONTENT_TYPES:
            filtered.append(item)
        else:
            skipped_ext.append(path)
    if skipped_ext:
        notes.append(
            "Export: skipped non-text files in 'output/': "
            + ", ".join(sorted(skipped_ext))
        )
    if not filtered:
        return [], notes

    files: list[dict] = []
    total = 0
    download = getattr(backend, "download_files", None)
    if not download:
        return [], notes + ["Export: sandbox backend cannot download files"]
    for item in sorted(filtered, key=lambda x: x["path"]):
        path = item["path"]
        size = item.get("size")
        if size is None:
            notes.append(f"Export: skipped {path} (could not determine file size)")
            continue
        if int(size) > MAX_FILE_BYTES:
            notes.append(
                f"Export: skipped {path} ({size} bytes > {MAX_FILE_BYTES} cap)"
            )
            continue
        if total + int(size) > MAX_TOTAL_BYTES:
            notes.append(
                f"Export: skipped {path} (total would exceed {MAX_TOTAL_BYTES} cap)"
            )
            continue
        try:
            responses = download([path])
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Export: could not download {path} ({exc})")
            continue
        if not responses:
            notes.append(f"Export: could not download {path} (empty response)")
            continue
        resp = responses[0]
        if getattr(resp, "error", None):
            notes.append(f"Export: could not download {path} ({resp.error})")
            continue
        content = getattr(resp, "content", None) or b""
        if len(content) > MAX_FILE_BYTES:
            notes.append(
                f"Export: skipped {path} "
                f"({len(content)} bytes > {MAX_FILE_BYTES} cap)"
            )
            continue
        if total + len(content) > MAX_TOTAL_BYTES:
            notes.append(
                f"Export: skipped {path} (total would exceed {MAX_TOTAL_BYTES} cap)"
            )
            continue
        total += len(content)
        ext = os.path.splitext(path)[1].lower()
        files.append(
            {
                "name": os.path.basename(path),
                "path": path,
                "content_type": REPORT_FILE_CONTENT_TYPES[ext],
                "content": content.decode("utf-8", errors="replace"),
            }
        )
    return files, notes