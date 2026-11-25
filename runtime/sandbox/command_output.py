"""Parse OpenSandbox command execution output."""

from __future__ import annotations

from typing import Any


def log_item_text(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    text = getattr(item, "text", None)
    if text is not None:
        return str(text)
    return str(item)


def join_stream_chunks(items: list | None) -> str:
    """Join SDK log chunks (one line per chunk, newline stripped by SDK).

    Inserts ``\\n`` only *between* chunks. A single-chunk stream (e.g. ``printf``
    without a trailing newline) stays without an artificial trailing newline.
    Multi-line commands still rebuild full lines because each stdout line is its
    own chunk.
    """
    parts = [log_item_text(item) for item in (items or [])]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    out = parts[0]
    for part in parts[1:]:
        if part == "":
            if not out.endswith("\n"):
                out += "\n"
            out += "\n"
            continue
        if out and not out.endswith("\n"):
            out += "\n"
        out += part
    return out


def split_command_streams(execution) -> tuple[str, str]:
    """Return ``(stdout, stderr)`` from an execution object."""
    logs = getattr(execution, "logs", None)
    stdout = join_stream_chunks(getattr(logs, "stdout", None) if logs else None)
    stderr = join_stream_chunks(getattr(logs, "stderr", None) if logs else None)
    if not stdout and not stderr:
        stdout = join_stream_chunks(getattr(execution, "result", None) or [])
    return stdout, stderr


def combine_stdout_stderr(stdout: str, stderr: str, exit_code: int) -> str:
    """Merge streams for ``execute`` tool text.

    On success, stdout stays primary (e.g. JSON from grep helpers); stderr is
    appended when present so warnings remain visible.
    """
    _ = exit_code
    if not stderr:
        return stdout
    if not stdout:
        return stderr
    if stdout.endswith("\n"):
        return f"{stdout}{stderr}"
    return f"{stdout}\n{stderr}"