"""Resolve agent-facing sandbox paths to OpenSandbox Files API paths."""

from __future__ import annotations

WORKDIR = "/workspace"
OUTPUT_REAL = "/output"
WORKSPACE_OUTPUT = f"{WORKDIR}/output"


def _collapse_slashes(path: str) -> str:
    while "//" in path:
        path = path.replace("//", "/")
    return path


def resolve_files_api_path(path: str, *, cwd: str = WORKDIR) -> str:
    """Map cwd-relative and ``/workspace/output`` aliases to real API paths.

    The Files API refuses symlink traversal, so ``/workspace/output`` (symlink
    to ``/output``) must be rewritten before any list/read/write/search call.
  Relative paths are resolved against ``cwd`` (default ``/workspace``).
    """
    raw = (path or "").strip()
    if not raw or raw == ".":
        return cwd
    if raw.startswith("/"):
        abs_path = _collapse_slashes(raw)
    else:
        base = cwd.rstrip("/")
        abs_path = _collapse_slashes(f"{base}/{raw}")
    if abs_path == WORKSPACE_OUTPUT or abs_path.startswith(f"{WORKSPACE_OUTPUT}/"):
        suffix = abs_path[len(WORKSPACE_OUTPUT) :]
        return f"{OUTPUT_REAL}{suffix}" if suffix else OUTPUT_REAL
    return abs_path


def normalize_output_export_path(path: str) -> str | None:
    """Top-level report file under canonical ``/output/<name>``."""
    canonical = resolve_files_api_path(path)
    parts = canonical.strip("/").split("/")
    if len(parts) == 2 and parts[0] == "output":
        return f"/{parts[0]}/{parts[1]}"
    return None