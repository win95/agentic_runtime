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
    """Join SDK log chunks (one stripped line per chunk) with newlines."""
    parts = [log_item_text(item) for item in (items or [])]
    if not parts:
        return ""
    text = "\n".join(parts)
    return f"{text}\n"


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
    appended when present so warnings remain visible. Stdout from the SDK already
    ends with a trailing newline, so we do not insert an extra blank line.
    """
    _ = exit_code
    if not stderr:
        return stdout
    if not stdout:
        return stderr
    if stdout.endswith("\n"):
        return f"{stdout}{stderr}"
    return f"{stdout}\n{stderr}"