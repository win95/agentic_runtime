"""Deep Agents backend backed by a self-hosted OpenSandbox sandbox.

``OpenSandboxBackend`` subclasses deepagents' ``BaseSandbox``: the
only methods a subclass must provide are ``execute``,
``upload_files``, ``download_files`` and the ``id`` property — the
file tools (ls/read/write/edit/glob/grep) are implemented by
``BaseSandbox`` itself on top of ``execute`` (its read helper runs a
small Python script inside the sandbox, so the sandbox image MUST
contain python3 — see ``settings.sandbox_image``).

The handle is a ``SandboxSync`` (the SDK's synchronous client). The
backend never holds credentials: the handle is authenticated by the
connection config built in ``runtime.sandbox.provider``.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

logger = logging.getLogger(__name__)


def _log_item_text(item) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    text = getattr(item, "text", None)
    if text is not None:
        return str(text)
    return str(item)


def format_command_output(execution) -> str:
    """Shell output from OpenSandbox execution (1.2+ logs or legacy result)."""
    logs = getattr(execution, "logs", None)
    stdout = getattr(logs, "stdout", None) if logs is not None else None
    stderr = getattr(logs, "stderr", None) if logs is not None else None
    out_lines = [_log_item_text(item) for item in (stdout or [])]
    err_lines = [_log_item_text(item) for item in (stderr or [])]
    out_lines = [line for line in out_lines if line]
    err_lines = [line for line in err_lines if line]
    if out_lines or err_lines:
        chunks: list[str] = []
        if out_lines:
            chunks.append("\n".join(out_lines))
        if err_lines:
            chunks.append("\n".join(err_lines))
        return "\n".join(chunks)
    return "".join(_log_item_text(item) for item in (getattr(execution, "result", None) or []))


class OpenSandboxBackend(BaseSandbox):
    """A ``BaseSandbox`` whose shell and files live in one OpenSandbox
    sandbox. ``sandbox_id`` is the server-side id (the same value the
    session mapping stores in Postgres)."""

    def __init__(self, sandbox, sandbox_id: str):
        self._sandbox = sandbox
        self._sandbox_id = sandbox_id

    @property
    def id(self) -> str:
        return self._sandbox_id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run ``command`` in the sandbox and return combined output."""
        from opensandbox.models.execd import RunCommandOpts

        opts = (
            RunCommandOpts(timeout=timedelta(seconds=timeout))
            if timeout is not None
            else None
        )
        execution = self._sandbox.commands.run(command, opts=opts)
        output = format_command_output(execution)
        if not output and execution.error:
            output = str(execution.error)
        return ExecuteResponse(
            output=output, exit_code=execution.exit_code, truncated=False
        )

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload (path, bytes) pairs; one response per file, in order."""
        from opensandbox.models.filesystem import WriteEntry

        try:
            self._sandbox.files.write_files(
                [WriteEntry(path=path, data=content) for path, content in files]
            )
        except Exception as exc:  # noqa: BLE001 — per-file error, not a crash
            logger.warning("Sandbox upload failed: %s", exc)
            return [
                FileUploadResponse(path=path, error=str(exc)[:200])
                for path, _ in files
            ]
        return [FileUploadResponse(path=path) for path, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download each path as bytes; a missing/unreadable path
        returns an error response for that path only."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self._sandbox.files.read_bytes(path)
            except Exception as exc:  # noqa: BLE001 — per-path error
                responses.append(
                    FileDownloadResponse(path=path, error=str(exc)[:200])
                )
            else:
                responses.append(
                    FileDownloadResponse(path=path, content=content)
                )
        return responses
