"""Slack notifications for executions — Incoming Webhook, stdlib only.

Scope (tran's choice, 2026-11-11): notify ONLY on failure + admin
actions, never on every execution. "Admin action" means an execution
of mission ``system_admin`` whose output reports a non-empty
``actions_taken`` list — a read-only admin query (empty / missing
``actions_taken``) does not notify.

The worker calls :func:`notify_execution` best-effort after the final
status is persisted; a notification failure (no webhook configured,
network error, non-2xx) is logged and never fails the execution.
With no ``SLACK_WEBHOOK_URL`` configured this module is a silent
no-op. No webhook URL is ever hardcoded here.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable

logger = logging.getLogger(__name__)

ADMIN_MISSION_ID = "system_admin"
MAX_ERROR_CHARS = 500
MAX_ACTION_CHARS = 200
MAX_ACTIONS = 5

# Injectable sender type: (webhook_url, payload, timeout) -> bool.
Sender = Callable[[str, dict, int], bool]


def _as_dict(output: Any) -> dict:
    return output if isinstance(output, dict) else {}


def should_notify(
    *,
    mission_id: str | None,
    status: str | None,
    output: Any = None,
    enabled: bool = True,
    webhook_url: str = "",
) -> bool:
    """True iff this execution deserves a Slack message. Never raises."""
    try:
        if not enabled:
            return False
        if not webhook_url or not str(webhook_url).strip():
            return False
        if status == "FAILED":
            return True
        if mission_id == ADMIN_MISSION_ID and status == "SUCCEEDED":
            actions = _as_dict(output).get("actions_taken")
            return isinstance(actions, list) and len(actions) > 0
        return False
    except Exception:  # noqa: BLE001 — scope check must never raise
        return False


def build_message(
    *,
    execution_id: str,
    mission_id: str | None,
    agent_id: str | None,
    status: str | None,
    output: Any = None,
    error: str | None = None,
) -> dict:
    """Build the ``{"text": ...}`` webhook payload (plain text, Vietnamese).

    One short message: status line, mission, execution id, and either
    the error (truncated to 500 chars) or the admin actions summary
    (max 5 items, each truncated to 200 chars). Never raises.
    """
    try:
        out = _as_dict(output)
        agent = agent_id or "—"
        mission = mission_id or "—"
        lines = []
        if status == "FAILED":
            lines.append(f"Execution FAILED — mission {mission}")
        elif mission_id == ADMIN_MISSION_ID:
            lines.append(f"Hành động admin — mission {mission}")
        else:
            lines.append(f"Execution {status or 'UNKNOWN'} — mission {mission}")
        lines.append(f"Execution: {execution_id}")
        lines.append(f"Agent: {agent}")
        if status == "FAILED":
            err = (error or "").strip() or "(không có thông tin lỗi)"
            lines.append(f"Lỗi: {err[:MAX_ERROR_CHARS]}")
        else:
            actions = out.get("actions_taken")
            if isinstance(actions, list) and actions:
                lines.append("Hành động đã thực hiện:")
                for action in actions[:MAX_ACTIONS]:
                    lines.append(f"- {str(action)[:MAX_ACTION_CHARS]}")
                if len(actions) > MAX_ACTIONS:
                    lines.append(f"- … và {len(actions) - MAX_ACTIONS} hành động nữa")
        return {"text": "\n".join(lines)}
    except Exception:  # noqa: BLE001 — message building must never raise
        return {"text": f"Execution {status or 'UNKNOWN'} — {execution_id}"}


def _urllib_post(webhook_url: str, payload: dict, timeout: int = 5) -> bool:
    """POST JSON to the webhook URL. 2xx => True, anything else => False."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return 200 <= response.status < 300


def post_notification(
    webhook_url: str,
    payload: dict,
    timeout: int = 5,
    _post: Sender | None = None,
) -> bool:
    """Send one payload to the webhook. Returns False (never raises) on
    empty URL/payload, any exception, a falsy sender result, or a
    non-2xx response from the default urllib sender."""
    try:
        if not webhook_url or not str(webhook_url).strip():
            return False
        if not payload:
            return False
        sender = _post or _urllib_post
        return bool(sender(webhook_url, payload, timeout))
    except Exception as exc:  # noqa: BLE001 — posting must never raise
        logger.warning("Slack notification failed: %s", exc)
        return False


def notify_execution(
    *,
    execution_id: str,
    mission_id: str | None,
    agent_id: str | None,
    status: str | None,
    output: Any = None,
    error: str | None = None,
    enabled: bool = True,
    webhook_url: str = "",
    timeout: int = 5,
    _post: Sender | None = None,
) -> bool:
    """Scope check + build + post. True only when a message was posted;
    False when skipped or when posting failed. Never raises."""
    try:
        if not should_notify(
            mission_id=mission_id,
            status=status,
            output=output,
            enabled=enabled,
            webhook_url=webhook_url,
        ):
            return False
        payload = build_message(
            execution_id=execution_id,
            mission_id=mission_id,
            agent_id=agent_id,
            status=status,
            output=output,
            error=error,
        )
        return post_notification(
            webhook_url, payload, timeout=timeout, _post=_post
        )
    except Exception as exc:  # noqa: BLE001 — notification never fails a run
        logger.warning("Slack notification failed: %s", exc)
        return False
