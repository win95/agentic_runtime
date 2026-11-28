"""Worker-side failure notifier for Lark executions (D-LARK-16).

When a Lark-origin execution ends FAILED, the user otherwise gets
silence — the agent only replies when the run succeeds. The worker
sends one short error message via the Lark client (the same client
the tools use — NOT a tool call, the run is already over).

Rules:
- Only for ``mission_id == "agent_lark_me_mission"`` with a
  ``chat_id`` in the execution input (other missions never get a Lark
  message, even if their input happens to carry a chat_id) AND only
  when the mission's harness opts in
  (``notifications.lark_failure_notice`` — the worker passes the
  flag from ``effective_harness``, like the Slack admin-action flag).
- Only for status FAILED (not CANCELLED; INTERRUPTED is HITL, not an
  error).
- Best-effort, never raises, never blocks the worker; a failed send
  is only logged.
- No retry of the execution, no error details to the user.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

LARK_MISSION_ID = "agent_lark_me_mission"
_FAILURE_TEXT = "Bot gặp lỗi khi xử lý tin của bạn, vui lòng thử lại sau."


def notify_lark_execution_failed(
    *,
    mission_id: str,
    execution_input: dict,
    status: str,
    lark_failure_notice: bool = False,
) -> None:
    """Send one short error message when a Lark run FAILED. Never raises."""
    try:
        if (
            status != "FAILED"
            or not lark_failure_notice
            or mission_id != LARK_MISSION_ID
        ):
            return
        chat_id = (execution_input or {}).get("chat_id")
        if not isinstance(chat_id, str) or not chat_id:
            return
        from runtime.integrations import lark_client

        root_id = (execution_input or {}).get("root_id")
        message_id = (execution_input or {}).get("message_id")
        target = root_id if isinstance(root_id, str) and root_id else None
        if target is None and isinstance(message_id, str) and message_id:
            target = message_id
        if target is not None:
            result = lark_client.reply_message(
                message_id=target, text=_FAILURE_TEXT, reply_in_thread=True
            )
        else:
            result = lark_client.send_message(
                receive_id=chat_id, receive_id_type="chat_id", text=_FAILURE_TEXT
            )
        if isinstance(result, dict) and "error" in result:
            logger.warning(
                "Lark failure notice failed for chat %s: %s", chat_id, result["error"]
            )
    except Exception as exc:  # noqa: BLE001 — notify never blocks
        logger.warning("Lark failure notice raised: %s", exc)
