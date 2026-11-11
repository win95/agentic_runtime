"""Slack notifications for executions (Incoming Webhook, stdlib only).

See ``runtime/notifications/slack.py`` and ``docs/slack-notifications.md``.
Scope: only failures + admin actions (mission ``system_admin`` with a
non-empty ``actions_taken``), never every execution.
"""

from __future__ import annotations

from runtime.notifications.slack import (
    build_message,
    notify_execution,
    post_notification,
    should_notify,
)

__all__ = [
    "build_message",
    "notify_execution",
    "post_notification",
    "should_notify",
]
