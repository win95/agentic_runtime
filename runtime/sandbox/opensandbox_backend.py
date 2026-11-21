"""Deep Agents backend backed by a self-hosted OpenSandbox sandbox."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    LsResult,
    ReadResult,
)
from deepagents.backends.sandbox import BaseSandbox, normalize_read_bounds

from runtime.sandbox.command_output import combine_stdout_stderr, split_command_streams

logger = logging.getLogger(__name__)

WORKDIR = "/workspace"
OUTPUT_DIRS = ("/output", f"{WORKDIR}/output")
MTIME_SKEW_TOLERANCE = timedelta(seconds=5)


def _is_toplevel_output_path(path: str) -> bool:
    norm = path if path.startswith("/") else f"/{path}"
    parts = norm.strip("/").split("/")
    return len(parts) == 2 and parts[0] == "output"

RecreateFn = Callable[[], tuple[Any, str] | None]


def is_sandbox_not_found(exc: BaseException) -> bool:
    text = str(exc).upper()
    return "SANDBOX_NOT_FOUND" in text or "DOCKER::SANDBOX_NOT_FOUND" in text


def is_transport_failure(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"RemoteProtocolError", "ReadError", "ConnectError", "ReadTimeout"}:
        return True
    mod = type(exc).__module__ or ""
    return "httpx" in mod or "httpcore" in mod


class OpenSandboxBackend(BaseSandbox):
    """``BaseSandbox`` over OpenSandbox 1.2+ files + commands APIs."""

    enable_capture_offload = True

    def __init__(
        self,
        sandbox,
        sandbox_id: str,
        *,
        command_timeout_seconds: int = 90,
        recreate_callback: RecreateFn | None = None,
    ):
        self._sandbox = sandbox
        self._sandbox_id = sandbox_id
        self._default_timeout = max(1, int(command_timeout_seconds or 90))
        self._recreate_callback = recreate_callback
        self._unavailable = False
        self._recreate_attempted = False
        self._run_started_at: float | None = None
        self._clock_skew: float = 0.0

    @property
    def id(self) -> str:
        return self._sandbox_id

    def begin_run_export(self, host_started_at: float) -> None:
        """Record run start for mtime-based export (host monotonic clock)."""
        self._run_started_at = float(host_started_at)
        self._clock_skew = self._probe_clock_skew(host_started_at)

    def _probe_clock_skew(self, host_started_at: float) -> float:
        resp = self._run_command("date +%s", timeout_seconds=5, allow_recreate=False)
        if resp.exit_code != 0:
            return 0.0
        try:
            container = float((resp.output or "").strip().split()[0])
            return container - host_started_at
        except (ValueError, IndexError):
            return 0.0

    def _effective_timeout(self, timeout: int | None) -> int:
        requested = self._default_timeout if timeout is None else int(timeout)
        return max(1, min(requested, self._default_timeout))

    def _run_command(
        self,
        command: str,
        *,
        timeout_seconds: int | None = None,
        allow_recreate: bool = True,
    ) -> ExecuteResponse:
        if self._unavailable:
            return ExecuteResponse(
                output="Sandbox unavailable — session sandbox was removed or could not be recreated",
                exit_code=127,
                truncated=False,
            )
        from opensandbox.models.execd import RunCommandOpts

        effective = self._effective_timeout(timeout_seconds)
        opts = RunCommandOpts(
            timeout=timedelta(seconds=effective),
            working_directory=WORKDIR,
        )
        try:
            execution = self._sandbox.commands.run(command, opts=opts)
        except Exception as exc:  # noqa: BLE001
            if allow_recreate and not self._recreate_attempted and is_sandbox_not_found(exc):
                if self._try_recreate():
                    return self._run_command(
                        command,
                        timeout_seconds=timeout_seconds,
                        allow_recreate=False,
                    )
            if is_sandbox_not_found(exc):
                self._unavailable = True
                return ExecuteResponse(
                    output=(
                        "Sandbox unavailable — session sandbox was removed "
                        "or could not be recreated"
                    ),
                    exit_code=127,
                    truncated=False,
                )
            if allow_recreate and not self._recreate_attempted and is_transport_failure(exc):
                logger.warning("Sandbox transport error on execute: %s", exc)
            self._unavailable = True
            return ExecuteResponse(
                output=f"Sandbox command failed: {exc}",
                exit_code=127,
                truncated=False,
            )

        stdout, stderr = split_command_streams(execution)
        exit_code = execution.exit_code
        if exit_code is None:
            msg = (
                f"command timed out after {effective}s; narrow the path or scope"
            )
            logger.info("Sandbox command timeout (exit_code=None): %s", command[:120])
            return ExecuteResponse(output=msg, exit_code=124, truncated=True)

        if not stdout and not stderr and execution.error:
            detail = str(execution.error)
            if "timeout" in detail.lower() or "timed out" in detail.lower():
                return ExecuteResponse(
                    output=(
                        f"command timed out after {effective}s; narrow the path or scope"
                    ),
                    exit_code=124,
                    truncated=True,
                )
            return ExecuteResponse(output=detail, exit_code=exit_code or 1, truncated=False)

        output = combine_stdout_stderr(stdout, stderr, int(exit_code or 0))
        if int(exit_code or 0) != 0 and not output.strip():
            output = f"command exited {exit_code}"
        return ExecuteResponse(
            output=output,
            exit_code=int(exit_code or 0),
            truncated=False,
        )

    def _try_recreate(self) -> bool:
        self._recreate_attempted = True
        if self._recreate_callback is None:
            self._unavailable = True
            return False
        try:
            result = self._recreate_callback()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sandbox recreate callback failed: %s", exc)
            self._unavailable = True
            return False
        if not result:
            self._unavailable = True
            return False
        handle, sandbox_id = result
        self._sandbox = handle
        self._sandbox_id = sandbox_id
        self._unavailable = False
        logger.info("Sandbox expired, recreated as %s", sandbox_id)
        return True

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._run_command(command, timeout_seconds=timeout)

    def ls(self, path: str) -> LsResult:
        if self._unavailable:
            return LsResult(error="Sandbox unavailable")
        from opensandbox.models.filesystem import DirectoryListEntry

        try:
            entries = self._sandbox.files.list_directory(
                DirectoryListEntry(path=path, depth=1)
            )
        except Exception as exc:  # noqa: BLE001
            if not self._recreate_attempted and is_sandbox_not_found(exc):
                if self._try_recreate():
                    return self.ls(path)
            return LsResult(error=f"Path '{path}': {exc}")
        file_infos = []
        for entry in entries:
            entry_type = (getattr(entry, "entry_type", None) or "").lower()
            is_dir = entry_type in {"dir", "directory", "d"}
            if not is_dir and entry.size == 0 and entry_type not in {"file", "f", ""}:
                is_dir = entry_type == ""
            file_infos.append({"path": entry.path, "is_dir": is_dir})
        return LsResult(entries=file_infos)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        if self._unavailable:
            return ReadResult(error=f"File '{file_path}': sandbox unavailable")
        offset, limit = normalize_read_bounds(offset, limit)
        if limit <= 0:
            return ReadResult(
                no_lines_requested=True,
                file_data=FileData(content="", encoding="utf-8"),
            )
        sdk_offset = offset + 1
        try:
            content = self._sandbox.files.read_file(
                file_path,
                offset=sdk_offset,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            if not self._recreate_attempted and is_sandbox_not_found(exc):
                if self._try_recreate():
                    return self.read(file_path, offset=offset, limit=limit)
            return ReadResult(error=f"File '{file_path}': {exc}")
        lines = content.split("\n")
        if content.endswith("\n") and lines:
            lines = lines[:-1]
        n = len(lines)
        start = offset + 1
        end = offset + n
        next_off = end if n >= limit else None
        total = end if next_off is None else None
        return ReadResult(
            file_data=FileData(content=content, encoding="utf-8"),
            start_line=start,
            end_line=end,
            next_offset=next_off,
            total_lines=total,
        )

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        if self._unavailable:
            return GlobResult(matches=None, error="Sandbox unavailable")
        from deepagents.backends.sandbox import _glob_search_root
        from opensandbox.models.filesystem import SearchEntry

        search_path = _glob_search_root(path) or WORKDIR
        try:
            matches = self._sandbox.files.search(
                SearchEntry(path=search_path, pattern=pattern)
            )
        except Exception as exc:  # noqa: BLE001
            if not self._recreate_attempted and is_sandbox_not_found(exc):
                if self._try_recreate():
                    return self.glob(pattern, path=path)
            return GlobResult(matches=None, error=str(exc))
        return GlobResult(matches=[{"path": m.path, "is_dir": False} for m in matches])

    def list_output_files_for_export(
        self, run_started_at: float, clock_skew: float = 0.0
    ) -> tuple[list[dict], list[str]]:
        """Top-level output files with ``modified_at`` after run start (mtime filter)."""
        notes: list[str] = []
        cutoff = datetime.fromtimestamp(
            float(run_started_at) + float(clock_skew), tz=timezone.utc
        ).replace(tzinfo=None) - MTIME_SKEW_TOLERANCE
        by_path: dict[str, dict] = {}
        for directory in OUTPUT_DIRS:
            try:
                from opensandbox.models.filesystem import DirectoryListEntry

                entries = self._sandbox.files.list_directory(
                    DirectoryListEntry(path=directory, depth=1)
                )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Export: could not list '{directory}' ({exc})")
                continue
            for entry in entries:
                entry_type = (getattr(entry, "entry_type", None) or "").lower()
                if entry_type in {"dir", "directory", "d"}:
                    continue
                path = entry.path
                if not _is_toplevel_output_path(path):
                    continue
                mod = entry.modified_at
                if isinstance(mod, datetime) and mod.tzinfo is not None:
                    mod = mod.replace(tzinfo=None)
                if mod < cutoff:
                    continue
                norm = path if path.startswith("/") else f"/{path}"
                if norm not in by_path:
                    by_path[norm] = {
                        "path": norm,
                        "size": int(entry.size),
                        "modified_at": mod,
                    }
        items = sorted(by_path.values(), key=lambda x: x["path"])
        if not items and not notes:
            notes.append(
                "Export: no report files newer than this run's start under output/"
            )
        return items, notes

    def has_exportable_files_since_run(
        self, run_started_at: float, clock_skew: float = 0.0
    ) -> tuple[bool | None, str | None]:
        try:
            items, notes = self.list_output_files_for_export(run_started_at, clock_skew)
            for note in notes:
                if "could not list" in note.lower():
                    return None, note
            import os

            from runtime.sandbox.export import _CONTENT_TYPES

            for item in items:
                ext = os.path.splitext(item["path"])[1].lower()
                if ext in _CONTENT_TYPES:
                    return True, None
            return False, None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        from opensandbox.models.filesystem import WriteEntry

        if self._unavailable:
            return [
                FileUploadResponse(path=p, error="sandbox unavailable")
                for p, _ in files
            ]
        try:
            self._sandbox.files.write_files(
                [WriteEntry(path=path, data=content) for path, content in files]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sandbox upload failed: %s", exc)
            return [
                FileUploadResponse(path=path, error=str(exc)[:200])
                for path, _ in files
            ]
        return [FileUploadResponse(path=path) for path, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        if self._unavailable:
            return [
                FileDownloadResponse(path=p, error="sandbox unavailable")
                for p in paths
            ]
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self._sandbox.files.read_bytes(path)
            except Exception as exc:  # noqa: BLE001
                responses.append(
                    FileDownloadResponse(path=path, error=str(exc)[:200])
                )
            else:
                responses.append(FileDownloadResponse(path=path, content=content))
        return responses


def bootstrap_sandbox_workspace(backend: OpenSandboxBackend) -> None:
    """Ensure workspace layout and ``/output`` symlink for deepagents paths."""
    backend.execute(
        "mkdir -p /workspace/output && ln -sfn /workspace/output /output",
        timeout=30,
    )


def format_command_output(execution) -> str:
    """Backward-compatible single-string output (used in tests)."""
    stdout, stderr = split_command_streams(execution)
    code = int(getattr(execution, "exit_code", 0) or 0)
    return combine_stdout_stderr(stdout, stderr, code)