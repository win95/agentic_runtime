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
- When ``run_marker_path`` is set (touched at run start in the sandbox),
  only files newer than that marker are exported (``find … -newer``),
  so prior-turn files in the same session are not re-inlined.
- Never raises: any failure returns ([], [note]) and the run proceeds.
  The export runs only when the mission declares ``files`` in its
  output schema (runtime/mission_graph.py fills it from here).
"""

from __future__ import annotations

import os
import re
import shlex

OUTPUT_DIRS = ("/output", "output")
RUN_MARKER_DIR = "/tmp"
MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 1024 * 1024

_CONTENT_TYPES = {
    ".html": "text/html",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
}


def export_run_marker_path(execution_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", execution_id or "unknown")[:80]
    return f"{RUN_MARKER_DIR}/.run_export_{safe}"


def touch_export_run_marker(backend, execution_id: str) -> str | None:
    """Create an in-sandbox marker file for this run (best-effort)."""
    execute = getattr(backend, "execute", None)
    if not execute or not (execution_id or "").strip():
        return None
    marker = export_run_marker_path(execution_id)
    try:
        resp = execute(f"touch {shlex.quote(marker)}")
        if int(getattr(resp, "exit_code", 0)) != 0:
            return None
        return marker
    except Exception:  # noqa: BLE001 — marker is best-effort
        return None


def _list_output_dir(backend, directory: str) -> tuple[list[dict], str | None]:
    try:
        listing = backend.ls(directory)
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)
    if getattr(listing, "error", None):
        return [], str(listing.error)
    return list(getattr(listing, "entries", None) or []), None


def _canonical_sandbox_path(backend, path: str) -> str:
    """Resolve to an absolute path (best-effort), assuming sandbox cwd is ``/``."""
    normalized = path if path.startswith("/") else f"/{path}"
    execute = getattr(backend, "execute", None)
    if not execute:
        return normalized
    try:
        quoted = shlex.quote(path)
        resp = execute(
            f"cd / && readlink -f {quoted} 2>/dev/null || realpath {quoted} 2>/dev/null"
        )
        lines = [
            line.strip()
            for line in (getattr(resp, "output", None) or "").splitlines()
            if line.strip()
        ]
        if lines:
            return lines[0]
    except Exception:  # noqa: BLE001
        pass
    return normalized


def _is_toplevel_output_file(canonical_path: str) -> bool:
    """True when ``canonical_path`` is a file directly under ``/output/``."""
    parts = canonical_path.strip("/").split("/")
    return len(parts) == 2 and parts[0] == "output"


def _dedupe_output_paths(backend, raw_paths: list[str]) -> list[str]:
    """Normalize via realpath and keep one path per inode (``/output`` vs ``output``)."""
    by_canonical: dict[str, str] = {}
    for path in raw_paths:
        canonical = _canonical_sandbox_path(backend, path)
        if not _is_toplevel_output_file(canonical):
            continue
        if canonical not in by_canonical:
            # Prefer the absolute ``/output/...`` form for download/stat.
            prefer = canonical if canonical.startswith("/output/") else path
            by_canonical[canonical] = prefer
    return sorted(by_canonical.values())


def _paths_from_ls(backend) -> tuple[list[str], list[str]]:
    """List output files via ``ls`` (no mtime) — mirrors BaseSandbox.ls entries."""
    notes: list[str] = []
    by_norm: dict[str, str] = {}
    failed_dirs: list[tuple[str, str]] = []
    succeeded_dirs: list[str] = []
    for directory in OUTPUT_DIRS:
        entries, err = _list_output_dir(backend, directory)
        if err:
            failed_dirs.append((directory, err))
            continue
        succeeded_dirs.append(directory)
        for entry in entries:
            if entry.get("is_dir"):
                continue
            path = str(entry.get("path") or "")
            if not path:
                continue
            canonical = _canonical_sandbox_path(backend, path)
            if not _is_toplevel_output_file(canonical):
                continue
            if canonical in by_norm:
                continue
            by_norm[canonical] = canonical if canonical.startswith("/") else path
    if failed_dirs:
        if not succeeded_dirs:
            for directory, err in failed_dirs:
                notes.append(f"Export: could not list '{directory}' in sandbox ({err})")
        elif len(failed_dirs) == 1:
            directory, err = failed_dirs[0]
            notes.append(
                f"Export: could not list '{directory}' in sandbox ({err}); "
                "using files from the other output path"
            )
    paths = list(by_norm.values())
    if not paths and not notes:
        notes.append("Export: no report files found under output/")
    return paths, notes


def _paths_newer_than_marker(backend, marker: str) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    execute = getattr(backend, "execute", None)
    if not execute:
        return [], [
            "Export: cannot apply run marker — sandbox backend has no execute"
        ]
    cmd = (
        f"cd / && (find /output -maxdepth 1 -type f -newer {shlex.quote(marker)} "
        f"2>/dev/null; find output -maxdepth 1 -type f -newer {shlex.quote(marker)} "
        f"2>/dev/null) | sort -u"
    )
    try:
        resp = execute(cmd)
        output = (getattr(resp, "output", None) or "").strip()
    except Exception as exc:  # noqa: BLE001
        return [], [f"Export: find newer-than-marker failed ({exc})"]
    raw = [line.strip() for line in output.splitlines() if line.strip()]
    paths = _dedupe_output_paths(backend, raw)
    if not raw and not paths:
        notes.append(
            "Export: no report files newer than this run's marker under output/"
        )
    return paths, notes


def _file_size_bytes(backend, path: str) -> int | None:
    """Return file size without downloading content, or None when unknown."""
    execute = getattr(backend, "execute", None)
    if execute is None:
        return None
    try:
        quoted = shlex.quote(path)
        resp = execute(f"stat -L -c %s {quoted} 2>/dev/null")
        output = (getattr(resp, "output", None) or "").strip().split()
        if not output:
            return None
        size = int(output[0])
        return size if size >= 0 else None
    except (ValueError, Exception):  # noqa: BLE001
        return None


def export_output_files(
    backend,
    *,
    directory: str | None = None,
    run_marker_path: str | None = None,
    marker_create_failed: bool = False,
):
    """Download the report files from sandbox ``output`` dirs.

    Returns (files, notes): ``files`` is a list of {"name", "path",
    "content_type", "content"} dicts with content decoded as UTF-8
    text; ``notes`` names anything skipped or failed. ``backend`` None
    (in-memory fallback) yields a clear warning note (not silent success).
    """
    _ = directory
    notes: list[str] = []
    if marker_create_failed:
        notes.append(
            "Export: run marker could not be created — listing output without "
            "per-run file filter (stale session files may be included)"
        )
    if backend is None:
        return [], notes + [
            "Export: sandbox backend unavailable — no files exported "
            "(in-memory fallback cannot export; enable SANDBOX_* or see notes)"
        ]
    if run_marker_path:
        paths, list_notes = _paths_newer_than_marker(backend, run_marker_path)
    else:
        paths, list_notes = _paths_from_ls(backend)
    notes.extend(list_notes)
    filtered_paths: list[str] = []
    skipped_ext: list[str] = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext in _CONTENT_TYPES:
            filtered_paths.append(path)
        else:
            skipped_ext.append(path)
    if skipped_ext:
        notes.append(
            "Export: skipped non-text files in 'output/': "
            + ", ".join(sorted(skipped_ext))
        )
    if not filtered_paths:
        return [], notes

    files: list[dict] = []
    total = 0
    for path in sorted(filtered_paths):
        canonical = _canonical_sandbox_path(backend, path)
        download_path = canonical if canonical.startswith("/") else path
        size = _file_size_bytes(backend, download_path)
        if size is None:
            notes.append(f"Export: skipped {path} (could not determine file size)")
            continue
        if size > MAX_FILE_BYTES:
            notes.append(
                f"Export: skipped {path} ({size} bytes > {MAX_FILE_BYTES} cap)"
            )
            continue
        if total + size > MAX_TOTAL_BYTES:
            notes.append(
                f"Export: skipped {path} (total would exceed {MAX_TOTAL_BYTES} cap)"
            )
            continue
        try:
            responses = backend.download_files([download_path])
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
                "path": download_path,
                "content_type": _CONTENT_TYPES[ext],
                "content": content.decode("utf-8", errors="replace"),
            }
        )
    return files, notes