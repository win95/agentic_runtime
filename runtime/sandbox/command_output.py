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
    """Join log chunks without inserting line breaks (preserves partial lines)."""
    return "".join(log_item_text(item) for item in (items or []))


def split_command_streams(execution) -> tuple[str, str]:
    """Return ``(stdout, stderr)`` from an execution object."""
    logs = getattr(execution, "logs", None)
    stdout = join_stream_chunks(getattr(logs, "stdout", None) if logs else None)
    stderr = join_stream_chunks(getattr(logs, "stderr", None) if logs else None)
    if not stdout and not stderr:
        stdout = "".join(
            log_item_text(item) for item in (getattr(execution, "result", None) or [])
        )
    return stdout, stderr


def combine_stdout_stderr(stdout: str, stderr: str, exit_code: int) -> str:
    """Prefer stdout alone on success; append stderr on failure (keeps JSON parse clean)."""
    if exit_code == 0:
        return stdout
    if stderr and stdout:
        return f"{stdout}\n{stderr}"
    return stdout or stderr