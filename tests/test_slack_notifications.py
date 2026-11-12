"""Slack notifier — scope matrix, message content, failure safety.

No network, no real webhook: posting is always through an injected
fake sender. Scope (tran's choice): only failures + admin actions
(mission system_admin with a non-empty actions_taken), never every
execution.
"""

from __future__ import annotations

from runtime.notifications.slack import (
    notify_execution,
    post_notification,
    should_notify,
)

WEBHOOK = "https://hooks.slack.com/services/T000/B000/xxxx"


class FakeSender:
    """Records calls; configurable to raise or to report failure."""

    def __init__(self, result: bool = True, exc: Exception | None = None):
        self.calls: list[tuple] = []
        self.result = result
        self.exc = exc

    def __call__(self, webhook_url, payload, timeout=5):
        self.calls.append((webhook_url, payload, timeout))
        if self.exc is not None:
            raise self.exc
        return self.result


def _notify(status, mission_id="fault_analysis", output=None, **kw):
    sender = kw.pop("_sender", None)
    return notify_execution(
        execution_id="exec_1",
        mission_id=mission_id,
        agent_id="fault_investigator",
        status=status,
        output=output or {},
        error=kw.pop("error", None),
        enabled=kw.pop("enabled", True),
        webhook_url=kw.pop("webhook_url", WEBHOOK),
        _post=sender,
        **kw,
    )


# --- scope matrix ---


def test_failed_fault_analysis_notifies():
    sender = FakeSender()
    assert _notify("FAILED", _sender=sender) is True
    assert len(sender.calls) == 1
    assert sender.calls[0][0] == WEBHOOK


def test_succeeded_fault_analysis_does_not_notify():
    sender = FakeSender()
    assert _notify("SUCCEEDED", _sender=sender) is False
    assert sender.calls == []


def test_succeeded_system_admin_with_actions_notifies():
    sender = FakeSender()
    assert (
        _notify(
            "SUCCEEDED",
            mission_id="system_admin",
            output={"actions_taken": ["save_mission"]},
            _sender=sender,
        )
        is True
    )
    assert len(sender.calls) == 1


def test_succeeded_system_admin_empty_actions_does_not_notify():
    sender = FakeSender()
    assert (
        _notify(
            "SUCCEEDED",
            mission_id="system_admin",
            output={"actions_taken": []},
            _sender=sender,
        )
        is False
    )
    assert sender.calls == []


def test_succeeded_system_admin_missing_actions_does_not_notify():
    sender = FakeSender()
    assert _notify("SUCCEEDED", mission_id="system_admin", _sender=sender) is False
    assert sender.calls == []


def test_cancelled_does_not_notify():
    sender = FakeSender()
    assert _notify("CANCELLED", _sender=sender) is False
    assert sender.calls == []


def test_disabled_does_not_notify():
    sender = FakeSender()
    assert _notify("FAILED", enabled=False, _sender=sender) is False
    assert sender.calls == []


def test_empty_webhook_does_not_notify():
    sender = FakeSender()
    assert _notify("FAILED", webhook_url="", _sender=sender) is False
    assert _notify("FAILED", webhook_url="   ", _sender=sender) is False
    assert sender.calls == []


def test_should_notify_never_raises_on_weird_output():
    assert should_notify(
        mission_id="system_admin", status="SUCCEEDED", output="not-a-dict",
        webhook_url=WEBHOOK,
    ) is False
    assert should_notify(
        mission_id="system_admin", status="SUCCEEDED", output=None,
        webhook_url=WEBHOOK,
    ) is False
    assert should_notify(
        mission_id="system_admin", status="SUCCEEDED",
        output={"actions_taken": "save_mission"}, webhook_url=WEBHOOK,
    ) is False
    assert should_notify(mission_id=None, status=None, webhook_url=WEBHOOK) is False


# --- message content ---


# --- posting failures never raise ---


def test_post_notification_empty_url_or_payload_returns_false():
    sender = FakeSender()
    assert post_notification("", {"text": "hi"}, _post=sender) is False
    assert post_notification(WEBHOOK, {}, _post=sender) is False
    assert sender.calls == []


def test_post_notification_sender_raising_returns_false():
    sender = FakeSender(exc=RuntimeError("boom"))
    assert post_notification(WEBHOOK, {"text": "hi"}, _post=sender) is False
