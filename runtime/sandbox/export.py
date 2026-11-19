"""Export the agent's report files out of its session sandbox.

Direct return (tran 2026-11-19): after a run, the files the agent wrote
into the sandbox ``output/`` dir are downloaded and inlined into the
execution result (``result["files"]``) as text, so the user gets them
through the existing ``GET /executions/{id}/result`` API — no separate
store, no download endpoint. See docs/sandbox.md.

Rules:
- Only the ``output/`` dir is exported (the agent is told to write its
  reports there; see skills/agent-analytis/SKILL.md). Both ``output/``
  (relative) and ``/output/`` (deepagents absolute paths) are scanned.
- Only text report formats (.html, .csv, .md, .txt, .json); anything
  else is skipped with a note.
- Per-file cap 256 KiB, total cap 1 MiB — a report that big does not
  belong inlined in a JSON result row. Skipped files are named in the
  notes, never silently dropped. File size is checked before download.
- When ``run_started_at`` is set, only files modified at or after that
  time are exported (avoids stale re-export across turns in one session).
- Never raises: any failure returns ([], [note]) and the run proceeds.
  The export runs only when the mission declares ``files`` in its
  output schema (runtime/mission_graph.py fills it from here).
"""

from __future__ import annotations

import os
import shlex
import time
from datetime import datetime

OUTPUT_DIRS = ("/output", "output")
MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 1024 * 1024

_CONTENT_TYPES = {
    ".html": "text/html",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
}


def _parse_mtime(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None
    return None


def _file_size_bytes(backend, path: str) -> int | None:
    """Return file size without downloading content, or None when unknown."""
    execute = getattr(backend, "execute", None)
    if execute is None:
        return None
    try:
        quoted = shlex.quote(path)
        resp = execute(f"stat -c %s {quoted} 2>/dev/null || stat -f %z {quoted}")
        output = (getattr(resp, "output", None) or "").strip().split()
        if not output:
            return None
        size = int(output[0])
        return size if size >= 0 else None
    except Exception:  # noqa: BLE001 — size probe is best-effort
        return None


def _list_output_dir(backend, directory: str) -> tuple[list[dict], str | None]:
    try:
        listing = backend.ls(directory)
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)
    if getattr(listing, "error", None):
        return [], str(listing.error)
    return list(getattr(listing, "entries", None) or []), None


def _collect_entries(backend, *, run_started_at: float | None) -> tuple[list[dict], list[str]]:
    """Merge file entries from every output dir variant; dedupe by path."""
    notes: list[str] = []
    by_path: dict[str, dict] = {}
    listed_any = False
    for directory in OUTPUT_DIRS:
        entries, err = _list_output_dir(backend, directory)
        if err:
            notes.append(f"Export: could not list '{directory}' in sandbox ({err})")
            continue
        if entries:
            listed_any = True
        for entry in entries:
            if entry.get("is_dir"):
                continue
            path = str(entry.get("path") or "")
            if not path:
                continue
            norm = path if path.startswith("/") else f"/{path}"
            if norm in by_path:
                continue
            mtime = _parse_mtime(entry.get("modified_at") or entry.get("mtime"))
            if run_started_at is not None and mtime is not None and mtime < run_started_at:
                notes.append(
                    f"Export: skipped {path} (modified before this run started)"
                )
                continue
            by_path[norm] = {**entry, "path": path}
    if not by_path and not listed_any and not notes:
        notes.append("Export: no report files found under output/")
    return list(by_path.values()), notes


def export_output_files(
    backend,
    *,
    directory: str | None = None,
    run_started_at: float | None = None,
):
    """Download the report files from sandbox ``output`` dirs.

    Returns (files, notes): ``files`` is a list of {"name", "path",
    "content_type", "content"} dicts with content decoded as UTF-8
    text; ``notes`` names anything skipped or failed. ``backend`` None
    (in-memory fallback) yields a clear warning note (not silent success).
    ``directory`` is accepted for backwards compatibility but ignored —
    both ``output`` and ``/output`` are scanned.
    """
    _ = directory
    if backend is None:
        return [], [
            "Export: sandbox backend unavailable — no files exported "
            "(in-memory fallback cannot export; enable SANDBOX_* or see notes)"
        ]
    entries, list_notes = _collect_entries(backend, run_started_at=run_started_at)
    notes = list(list_notes)
    paths: list[str] = []
    skipped_ext: list[str] = []
    for entry in entries:
        path = str(entry.get("path") or "")
        ext = os.path.splitext(path)[1].lower()
        if ext in _CONTENT_TYPES:
            paths.append(path)
        else:
            skipped_ext.append(path)
    if skipped_ext:
        notes.append(
            "Export: skipped non-text files in 'output/': "
            + ", ".join(sorted(skipped_ext))
        )
    if not paths:
        if not notes:
            notes.append("Export: no report files found in output/")
        return [], notes

    files: list[dict] = []
    total = 0
    for path in sorted(paths):
        size = _file_size_bytes(backend, path)
        if size is None and run_started_at is not None:
            size = _file_size_bytes(backend, path if path.startswith("/") else f"/{path}")
        if size is not None and size > MAX_FILE_BYTES:
            notes.append(
                f"Export: skipped {path} ({size} bytes > {MAX_FILE_BYTES} cap)"
            )
            continue
        if size is not None and total + size > MAX_TOTAL_BYTES:
            notes.append(
                f"Export: skipped {path} (total would exceed {MAX_TOTAL_BYTES} cap)"
            )
            continue
        try:
            responses = backend.download_files([path])
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
                "content_type": _CONTENT_TYPES[ext],
                "content": content.decode("utf-8", errors="replace"),
            }
        )
    return files, notes


def run_start_timestamp() -> float:
    """Wall-clock seconds (UTC) at run start for mtime filtering."""
    return time.time()