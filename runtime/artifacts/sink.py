"""Persist sandbox ``output.files`` to configured artifact storage (worker-only)."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from infra.minio_client import artifact_location, presigned_get_url, upload_bytes
from runtime.harness.schema import OutputArtifactsHarness
from runtime.harness.validation import output_schema_declares_files
from runtime.sandbox.export import _CONTENT_TYPES

logger = logging.getLogger(__name__)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_PATH_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _safe_path_segment(value: str | None, *, fallback: str = "unknown") -> str:
    raw = (value or "").strip() or fallback
    cleaned = _PATH_SEGMENT_RE.sub("_", raw).strip("._")
    return (cleaned[:120] or fallback)


def safe_basename(name: str) -> str | None:
    """Return a safe file basename or None when the name is rejected."""
    raw = (name or "").strip()
    if not raw:
        return None
    if ".." in raw or "/" in raw or "\\" in raw:
        return None
    if _CONTROL_CHAR_RE.search(raw):
        return None
    base = os.path.basename(raw)
    if not base or base in (".", ".."):
        return None
    if _CONTROL_CHAR_RE.search(base):
        return None
    return base


def artifact_object_key(
    *,
    agent_id: str | None,
    session_id: str | None,
    execution_id: str,
    basename: str,
) -> str:
    agent = _safe_path_segment(agent_id, fallback="no_agent")
    session = _safe_path_segment(session_id, fallback="no_session")
    execution = _safe_path_segment(execution_id, fallback="unknown")
    return (
        f"artifacts/agents/{agent}/sessions/{session}/executions/{execution}/{basename}"
    )


def content_type_for_name(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


def _persist_one_file_minio(
    entry: dict[str, Any],
    *,
    object_key: str,
    policy: OutputArtifactsHarness,
    notes: list[str],
) -> dict[str, Any]:
    content = entry.get("content")
    if content is None:
        notes.append(f"Artifacts: skipped {entry.get('name')!r} (no inline content)")
        out = dict(entry)
        out["storage"] = "inline"
        return out
    text = content if isinstance(content, str) else str(content)
    try:
        body = text.encode("utf-8")
        name = str(entry.get("name") or object_key.rsplit("/", 1)[-1])
        ctype = str(entry.get("content_type") or "") or content_type_for_name(name)
        upload_bytes(object_key, body, content_type=ctype)
        url = presigned_get_url(object_key)
        out = dict(entry)
        out["storage"] = "minio"
        out["object_key"] = object_key
        out["location"] = artifact_location(object_key)
        out["url"] = url
        if policy.strip_inline:
            out.pop("content", None)
        return out
    except Exception as exc:  # noqa: BLE001 — best-effort per file
        logger.warning("Artifact upload failed for %s: %s", object_key, exc)
        if not policy.inline_fallback:
            notes.append(f"Artifacts: upload failed for {entry.get('name')!r} ({exc})")
            out = dict(entry)
            out["storage"] = "inline"
            return out
        notes.append(
            f"Artifacts: kept inline content for {entry.get('name')!r} (upload failed: {exc})"
        )
        out = dict(entry)
        out["storage"] = "inline"
        return out


def persist_output_files_to_minio(
    files: list[dict[str, Any]],
    *,
    policy: OutputArtifactsHarness,
    agent_id: str | None,
    session_id: str | None,
    execution_id: str,
    notes: list[str],
) -> list[dict[str, Any]]:
    """Upload each file entry; never raises."""
    updated: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            updated.append(entry)
            continue
        raw_name = str(entry.get("name") or "").strip() or str(entry.get("path") or "")
        name = safe_basename(raw_name)
        if name is None:
            notes.append(f"Artifacts: rejected unsafe file name {entry.get('name')!r}")
            out = dict(entry)
            out["storage"] = "inline"
            updated.append(out)
            continue
        key = artifact_object_key(
            agent_id=agent_id,
            session_id=session_id,
            execution_id=execution_id,
            basename=name,
        )
        updated.append(
            _persist_one_file_minio(
                entry,
                object_key=key,
                policy=policy,
                notes=notes,
            )
        )
    return updated


def _artifact_storage_summary(files: list[Any]) -> str | None:
    storages: set[str] = set()
    for entry in files:
        if isinstance(entry, dict) and entry.get("storage") in ("minio", "inline"):
            storages.add(str(entry["storage"]))
    if not storages:
        return None
    if storages == {"minio"}:
        return "minio"
    if storages == {"inline"}:
        return "inline"
    return "mixed"


def apply_output_artifact_storage(
    output: dict[str, Any],
    *,
    mission_dict: dict[str, Any],
    execution_id: str,
    agent_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    """Rewrite ``output.files`` when harness + schema opt in to MinIO storage."""
    from runtime.harness.registry import effective_harness

    output_schema = mission_dict.get("output_schema") or {}
    if not output_schema_declares_files(output_schema):
        return output
    execution_cfg = mission_dict.get("execution") or {}
    harness = effective_harness(
        mission_dict.get("mission_id") or mission_dict.get("id") or "",
        execution_cfg,
    )
    policy = harness.output_artifacts
    if policy.sink != "minio":
        return output
    files = output.get("files")
    if not isinstance(files, list) or not files:
        return output

    notes = list(output.get("notes") or [])
    new_files = persist_output_files_to_minio(
        files,
        policy=policy,
        agent_id=agent_id or execution_cfg.get("agent_id"),
        session_id=session_id,
        execution_id=execution_id,
        notes=notes,
    )
    result = {**output, "files": new_files}
    summary = _artifact_storage_summary(new_files)
    if summary:
        result["artifact_storage"] = summary
    if notes != list(output.get("notes") or []):
        result["notes"] = notes
    return result