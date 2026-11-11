"""Agent versioning (Option C) — one immutable snapshot per agent config.

An agent's behaviour is decided by four things: the system prompt, the
model/temperature, the allowed tool list, and the tools themselves
(name, description, schema, permission). An :class:`AgentVersion`
pins all four in one row, so a run is reproducible from
(prompt version + code + toolset) instead of from the prompt alone.

Rules:
- Rows are never edited in place. Any change creates a new version.
- Only one version per agent is active; the runtime uses the active
  one, and rollback is activating an older version.
- Creating a version whose snapshot is identical to the active one is
  a no-op (returns the active version, ``created=False``).
- The tool manifest covers the built-in tools (fault + admin). MCP
  tools are environment-dependent (they need a configured MCP server)
  and are NOT in the manifest; the runtime still filters them by the
  version's allowed_tools.
- ``git_sha`` and ``prompt_registry_version`` are best-effort metadata
  (NULL outside a git checkout / without a reachable MLflow server).
  The pinned ``prompt_text`` is what the runtime uses, so a version
  stays reproducible even when MLflow is down.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

from sqlalchemy.orm import Session

from app import models

logger = logging.getLogger(__name__)


class AgentNotFoundError(ValueError):
    pass


class AgentVersionNotFoundError(ValueError):
    pass


def _ensure_tools_registered() -> None:
    # Importing the tool modules registers them. Idempotent.
    import runtime.tools.admin_tools  # noqa: F401
    import runtime.tools.fault_tools  # noqa: F401


def _tool_input_schema(entry) -> dict:
    """JSON schema for a tool's args, when obtainable (best-effort)."""
    try:
        args_schema = getattr(entry.tool, "args_schema", None)
        if args_schema is not None:
            return args_schema.model_json_schema()
    except Exception:  # noqa: BLE001 — schema is metadata, not load-bearing
        pass
    return {}


def compute_tool_manifest() -> tuple[list[dict], str]:
    """The built-in toolset as a sorted manifest + its SHA-256.

    One entry per tool: name, description, capability, permission,
    input schema, implementation, timeout, retry count, read_only.
    The hash is over canonical JSON (sorted keys, stable separators),
    so the same toolset always hashes the same.
    """
    _ensure_tools_registered()
    from runtime.tools.registry import registry

    manifest = []
    for entry in registry.list():
        manifest.append({
            "name": entry.name,
            "description": entry.description,
            "capability": entry.capability,
            "permission": entry.permission,
            "input_schema": _tool_input_schema(entry),
            "implementation": entry.implementation,
            "timeout_seconds": entry.timeout_seconds,
            "retry_count": entry.retry_count,
            "read_only": entry.read_only,
        })
    manifest.sort(key=lambda e: e["name"])
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return manifest, digest


def _git_sha() -> str | None:
    """Current HEAD SHA of the repo checkout (None outside git)."""
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, timeout=5,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:  # noqa: BLE001 — metadata only
        return None


def _mlflow_prompt_version(prompt_name: str) -> int | None:
    """Version number of prompts:/<name>@production (None when unknown)."""
    try:
        from infra.mlflow_setup import is_mlflow_tracing_enabled

        if not is_mlflow_tracing_enabled():
            return None
        import mlflow.genai as genai

        version = genai.load_prompt(f"prompts:/{prompt_name}@production")
        return int(version.version) if version is not None else None
    except Exception:  # noqa: BLE001 — metadata only, never blocks versioning
        return None


def snapshot_agent_config(agent: models.Agent, *,
                          prompt_text: str | None = None,
                          record_mlflow_version: bool = False) -> dict:
    """Capture everything that decides this agent's behaviour, as a dict."""
    manifest, digest = compute_tool_manifest()
    return {
        "prompt_text": prompt_text if prompt_text is not None else agent.system_prompt,
        "prompt_registry_version": (
            _mlflow_prompt_version(agent.id) if record_mlflow_version else None
        ),
        "provider": agent.provider,
        "model": agent.model,
        "temperature": agent.temperature,
        "allowed_tools": list(agent.allowed_tools or []),
        "session_history": (
            dict(agent.session_history) if agent.session_history else None
        ),
        "tool_manifest": manifest,
        "manifest_sha256": digest,
        "git_sha": _git_sha(),
    }


def _snapshots_equal(a: dict, b: models.AgentVersion) -> bool:
    return (
        a["prompt_text"] == b.prompt_text
        and a["provider"] == b.provider
        and a["model"] == b.model
        and a["temperature"] == b.temperature
        and sorted(a["allowed_tools"]) == sorted(b.allowed_tools or [])
        and (a.get("session_history") or None) == (b.session_history or None)
        and a["manifest_sha256"] == b.manifest_sha256
    )


def get_active_version(db: Session, agent_id: str) -> models.AgentVersion | None:
    return (
        db.query(models.AgentVersion)
        .filter_by(agent_id=agent_id, is_active=True)
        .first()
    )


def get_version(db: Session, agent_id: str, version: int) -> models.AgentVersion | None:
    return (
        db.query(models.AgentVersion)
        .filter_by(agent_id=agent_id, version=version)
        .first()
    )


def list_versions(db: Session, agent_id: str) -> list[models.AgentVersion]:
    return (
        db.query(models.AgentVersion)
        .filter_by(agent_id=agent_id)
        .order_by(models.AgentVersion.version.desc())
        .all()
    )


def create_version(db: Session, agent_id: str, *, source: str = "seed",
                   activate: bool = True, prompt_text: str | None = None,
                   record_mlflow_version: bool = False
                   ) -> tuple[models.AgentVersion, bool]:
    """Snapshot the agent's current config as a new version.

    Returns ``(version, created)`` — ``created`` is False when the
    snapshot is identical to the active version (idempotent no-op).
    """
    agent = db.get(models.Agent, agent_id)
    if agent is None:
        raise AgentNotFoundError(f"Agent {agent_id!r} not found")
    snapshot = snapshot_agent_config(
        agent, prompt_text=prompt_text,
        record_mlflow_version=record_mlflow_version,
    )
    active = get_active_version(db, agent_id)
    if active is not None and _snapshots_equal(snapshot, active):
        return active, False
    max_version = (
        db.query(models.AgentVersion.version)
        .filter_by(agent_id=agent_id)
        .order_by(models.AgentVersion.version.desc())
        .first()
    )
    number = (max_version[0] + 1) if max_version else 1
    if activate and active is not None:
        active.is_active = False
    row = models.AgentVersion(
        id=f"{agent_id}@v{number}",
        agent_id=agent_id,
        version=number,
        source=source,
        is_active=activate,
        **snapshot,
    )
    db.add(row)
    db.commit()
    logger.info("Agent %s: created version v%d (source=%s, active=%s)",
                agent_id, number, source, activate)
    return row, True


def ensure_version_for_agent(db: Session, agent_id: str, *,
                             source: str = "seed") -> tuple[models.AgentVersion, bool]:
    """Create a version only when the agent's config changed since the
    active one (or when no version exists yet). Used by the seed loader
    so re-seeding an unchanged agent creates nothing."""
    return create_version(db, agent_id, source=source, activate=True)


def activate_version(db: Session, agent_id: str, version: int) -> models.AgentVersion:
    """Make an existing version the active one (deploy / rollback)."""
    row = get_version(db, agent_id, version)
    if row is None:
        raise AgentVersionNotFoundError(
            f"Agent {agent_id!r} has no version v{version}"
        )
    active = get_active_version(db, agent_id)
    if active is not None and active.id != row.id:
        active.is_active = False
    row.is_active = True
    db.commit()
    logger.info("Agent %s: activated version v%d", agent_id, version)
    return row
