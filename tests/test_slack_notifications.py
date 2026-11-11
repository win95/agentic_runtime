"""Slack notifier — scope matrix, message content, failure safety.

No network, no real webhook: posting is always through an injected
fake sender. Scope (tran's choice): only failures + admin actions
(mission system_admin with a non-empty actions_taken), never every
execution.
"""

from __future__ import annotations

from runtime.notifications.slack import (
    build_message,
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
            output={"actions_taken": ["create_mission"]},
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
        output={"actions_taken": "create_mission"}, webhook_url=WEBHOOK,
    ) is False
    assert should_notify(mission_id=None, status=None, webhook_url=WEBHOOK) is False


# --- message content ---


def test_message_contains_execution_id_mission_and_truncated_error():
    long_error = "x" * 600
    payload = build_message(
        execution_id="exec_42",
        mission_id="fault_analysis",
        agent_id="fault_investigator",
        status="FAILED",
        output={},
        error=long_error,
    )
    text = payload["text"]
    assert "exec_42" in text
    assert "fault_analysis" in text
    assert "x" * 500 in text
    assert "x" * 600 not in text  # truncated to 500 chars


def test_message_admin_actions_capped_and_truncated():
    actions = [f"action_{i}_" + "y" * 300 for i in range(7)]
    payload = build_message(
        execution_id="exec_7",
        mission_id="system_admin",
        agent_id="admin_bot",
        status="SUCCEEDED",
        output={"actions_taken": actions},
    )
    text = payload["text"]
    assert "exec_7" in text
    assert "system_admin" in text
    assert "action_0_" in text
    assert "action_4_" in text
    assert "action_5_" not in text  # max 5 items shown
    # Each item is truncated to 200 chars total: "action_0_" (9 chars)
    # + 191 y's; the full 300-y string never appears.
    assert "y" * 191 in text
    assert "y" * 200 not in text
    assert "y" * 300 not in text
    assert "2 hành động nữa" in text


def test_message_failed_without_error_has_placeholder():
    payload = build_message(
        execution_id="exec_9",
        mission_id="fault_analysis",
        agent_id=None,
        status="FAILED",
        output={},
        error=None,
    )
    assert "không có thông tin lỗi" in payload["text"]
    assert "exec_9" in payload["text"]


# --- posting failures never raise ---


def test_post_failure_sender_raising_returns_false():
    sender = FakeSender(exc=RuntimeError("boom"))
    assert _notify("FAILED", _sender=sender) is False


def test_post_failure_sender_false_returns_false():
    sender = FakeSender(result=False)
    assert _notify("FAILED", _sender=sender) is False


def test_post_notification_empty_url_or_payload_returns_false():
    sender = FakeSender()
    assert post_notification("", {"text": "hi"}, _post=sender) is False
    assert post_notification(WEBHOOK, {}, _post=sender) is False
    assert sender.calls == []


def test_post_notification_sender_raising_returns_false():
    sender = FakeSender(exc=RuntimeError("boom"))
    assert post_notification(WEBHOOK, {"text": "hi"}, _post=sender) is False
