"""Export the agent's report files out of its session sandbox.

Direct return (tran 2026-11-19): after a run, the files the agent wrote
into the sandbox ``output/`` dir are downloaded and inlined into the
execution result (``result["files"]``) as text, so the user gets them
through the existing ``GET /executions/{id}/result`` API — no separate
store, no download endpoint. See docs/sandbox.md.

Rules:
- Only the ``output/`` dir is exported (the agent is told to write its
  reports there; see skills/agent-analytis/SKILL.md).
- Only text report formats (.html, .csv, .md, .txt, .json); anything
  else is skipped with a note.
- Per-file cap 256 KiB, total cap 1 MiB — a report that big does not
  belong inlined in a JSON result row. Skipped files are named in the
  notes, never silently dropped.
- Never raises: any failure returns ([], [note]) and the run proceeds.
  The export runs only when the mission declares ``files`` in its
  output schema (runtime/mission_graph.py fills it from here).
"""

import os

OUTPUT_DIR = "output"
MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 1024 * 1024

_CONTENT_TYPES = {
    ".html": "text/html",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
}


def export_output_files(backend, *, directory=OUTPUT_DIR):
    """Download the report files from ``directory`` of ``backend``.

    Returns (files, notes): ``files`` is a list of {"name", "path",
    "content_type", "content"} dicts with content decoded as UTF-8
    text; ``notes`` names anything skipped or failed. ``backend`` None
    (in-memory fallback) means ([], []).
    """
    if backend is None:
        return [], []
    try:
        listing = backend.ls(directory)
    except Exception as exc:  # noqa: BLE001 — export never breaks a run
        return [], [f"Export: could not list '{directory}' in sandbox ({exc})"]
    if getattr(listing, "error", None):
        return [], [
            f"Export: could not list '{directory}' in sandbox ({listing.error})"
        ]
    entries = getattr(listing, "entries", None) or []
    paths = [
        e["path"]
        for e in entries
        if not e.get("is_dir")
        and os.path.splitext(e["path"])[1].lower() in _CONTENT_TYPES
    ]
    skipped = [
        e["path"]
        for e in entries
        if not e.get("is_dir")
        and os.path.splitext(e["path"])[1].lower() not in _CONTENT_TYPES
    ]
    notes = []
    if skipped:
        notes.append(
            "Export: skipped non-text files in 'output/': "
            + ", ".join(sorted(skipped))
        )
    if not paths:
        if not skipped:
            notes.append(f"Export: no report files found in '{directory}/'")
        return [], notes
    try:
        responses = backend.download_files(paths)
    except Exception as exc:  # noqa: BLE001 — export never breaks a run
        return [], notes + [f"Export: download from sandbox failed ({exc})"]
    files = []
    total = 0
    for resp in responses:
        path = getattr(resp, "path", "?")
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
                f"Export: skipped {path} "
                f"(total would exceed {MAX_TOTAL_BYTES} cap)"
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
