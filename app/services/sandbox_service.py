"""Session -> OpenSandbox mapping: one sandbox per session.

The service owns the ``session_sandboxes`` rows and nothing else —
it never runs the agent. Every function here is best-effort by
design: a sandbox failure returns ``None``/``False`` and the caller
falls back to the in-memory backend. A sandbox problem must never
fail an execution (docs/sandbox.md).

Lifecycle:
- ``resolve_session_sandbox`` — called from ``deep_agent_node`` when
  a mission opts in (``execution.sandbox: true``) and the run has a
  session_id. Re-attaches to the session's sandbox when the mapping
  is active; creates one (and the row) on first use; replaces the row
  when the old sandbox is unreachable server-side.
- ``destroy_session_sandbox`` — explicit teardown of one session.
- ``reap_idle_sandboxes`` — beat task: destroys sandboxes whose
  session has been idle longer than ``sandbox_idle_ttl_seconds``.

The ``client`` parameter is the injectable seam (a
``runtime.sandbox.provider.OpenSandboxClient`` in production, a fake
in tests). When omitted, one is built from settings.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.config import settings as _default_settings
from app.models import SessionSandbox
from runtime.sandbox.opensandbox_backend import OpenSandboxBackend

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _touch(row: SessionSandbox) -> None:
    row.last_used_at = _now()


def _handle_id(handle) -> str | None:
    # The SDK stores the server-side id as ``.id`` on the handle.
    return getattr(handle, "id", None) or getattr(handle, "sandbox_id", None)


def resolve_session_sandbox(
    db: Session,
    *,
    session_id: str,
    agent_id: str | None,
    mission_id: str | None,
    settings=None,
    client=None,
) -> OpenSandboxBackend | None:
    """Backend for ``session_id``'s sandbox, or None when sandboxes
    are disabled / unavailable (caller falls back to StateBackend).

    Disabled means: the master switch is off, or no API key is set
    (the server refuses to start without one, so there is nothing to
    talk to). A create/connect failure is logged and returns None —
    the run proceeds without a sandbox.
    """
    cfg = settings if settings is not None else _default_settings
    if not cfg.sandbox_enabled or not cfg.sandbox_api_key:
        return None
    from runtime.sandbox.provider import default_client

    client = client if client is not None else default_client(cfg)

    row = db.get(SessionSandbox, session_id)
    if row is not None and row.status == "active":
        try:
            handle = client.connect(row.sandbox_id)
        except Exception as exc:  # noqa: BLE001 — dead sandbox: replace it
            logger.warning(
                "Sandbox %s for session %s unreachable (%s); creating a new one",
                row.sandbox_id, session_id, exc,
            )
        else:
            _touch(row)
            db.commit()
            return OpenSandboxBackend(handle, row.sandbox_id)

    # No usable sandbox: create one (fresh row, or replace the dead one).
    try:
        handle = client.create(cfg.sandbox_image)
    except Exception as exc:  # noqa: BLE001 — server down: no sandbox
        logger.warning("Could not create sandbox for session %s: %s", session_id, exc)
        return None
    sandbox_id = _handle_id(handle)
    if not sandbox_id:
        logger.warning("Sandbox create returned a handle with no id")
        return None
    if row is None:
        row = SessionSandbox(
            session_id=session_id,
            sandbox_id=sandbox_id,
            agent_id=agent_id,
            mission_id=mission_id,
            status="active",
            image=cfg.sandbox_image,
        )
        db.add(row)
    else:
        row.sandbox_id = sandbox_id
        row.agent_id = agent_id
        row.mission_id = mission_id
        row.status = "active"
        row.image = cfg.sandbox_image
    _touch(row)
    db.commit()
    return OpenSandboxBackend(handle, sandbox_id)


def destroy_session_sandbox(
    db: Session,
    session_id: str,
    *,
    settings=None,
    client=None,
) -> bool:
    """Destroy ``session_id``'s sandbox and mark the row destroyed.

    Returns True when the sandbox is gone (or was already gone
    server-side — the row is still marked destroyed). Returns False
    when there is nothing to destroy, or the destroy itself failed
    (the row stays active so the next reap retries).
    """
    cfg = settings if settings is not None else _default_settings
    row = db.get(SessionSandbox, session_id)
    if row is None or row.status != "active":
        return False
    from runtime.sandbox.provider import default_client

    client = client if client is not None else default_client(cfg)
    try:
        handle = client.connect(row.sandbox_id)
    except Exception as exc:  # noqa: BLE001 — already gone server-side
        logger.warning(
            "Sandbox %s for session %s already unreachable (%s); marking destroyed",
            row.sandbox_id, session_id, exc,
        )
    else:
        try:
            client.destroy(handle)
        except Exception as exc:  # noqa: BLE001 — keep active, retry later
            logger.warning(
                "Could not destroy sandbox %s for session %s: %s",
                row.sandbox_id, session_id, exc,
            )
            return False
    row.status = "destroyed"
    _touch(row)
    db.commit()
    return True


def reap_idle_sandboxes(
    db: Session,
    *,
    settings=None,
    client=None,
) -> dict:
    """Destroy sandboxes idle longer than ``sandbox_idle_ttl_seconds``.

    Returns {"checked", "destroyed", "failed"} (failed = sandbox ids
    whose destroy failed; they stay active for the next reap). Files
    inside a reaped sandbox are gone with it — anything the agent
    must keep beyond the session belongs in the execution result or
    the host-side report store, not in the sandbox (docs/sandbox.md).
    """
    cfg = settings if settings is not None else _default_settings
    cutoff = _now() - timedelta(seconds=cfg.sandbox_idle_ttl_seconds)
    rows = (
        db.query(SessionSandbox)
        .filter(
            SessionSandbox.status == "active",
            SessionSandbox.last_used_at < cutoff,
        )
        .all()
    )
    destroyed = 0
    failed: list[str] = []
    for row in rows:
        if destroy_session_sandbox(
            db, row.session_id, settings=cfg, client=client
        ):
            destroyed += 1
        else:
            failed.append(row.sandbox_id)
    return {"checked": len(rows), "destroyed": destroyed, "failed": failed}
