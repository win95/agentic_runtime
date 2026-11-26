"""Worker failure notifier (D-LARK-16) — one short Lark message on FAILED."""

from runtime.integrations import lark_client, lark_notify


def test_notify_sends_reply_in_thread(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        lark_client,
        "reply_message",
        lambda **kw: calls.update(kw) or {"message_id": "om_n"},
    )
    lark_notify.notify_lark_execution_failed(
        mission_id="agent_lark_me_mission",
        execution_input={
            "chat_id": "oc_1",
            "message_id": "om_in",
            "root_id": "om_root",
        },
        status="FAILED",
    )
    assert calls["message_id"] == "om_root"
    assert "lỗi" in calls["text"]


def test_notify_falls_back_to_send_without_thread(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        lark_client,
        "send_message",
        lambda **kw: calls.update(kw) or {"message_id": "om_n"},
    )
    lark_notify.notify_lark_execution_failed(
        mission_id="agent_lark_me_mission",
        execution_input={"chat_id": "oc_1"},
        status="FAILED",
    )
    assert calls["receive_id"] == "oc_1"


def test_notify_ignores_other_missions_and_statuses(monkeypatch):
    called = []
    monkeypatch.setattr(
        lark_client, "send_message", lambda **kw: called.append(kw)
    )
    monkeypatch.setattr(
        lark_client, "reply_message", lambda **kw: called.append(kw)
    )
    lark_notify.notify_lark_execution_failed(
        mission_id="hotel_reception",
        execution_input={"chat_id": "oc_1"},
        status="FAILED",
    )
    lark_notify.notify_lark_execution_failed(
        mission_id="agent_lark_me_mission",
        execution_input={"chat_id": "oc_1"},
        status="SUCCEEDED",
    )
    lark_notify.notify_lark_execution_failed(
        mission_id="agent_lark_me_mission",
        execution_input={"chat_id": "oc_1"},
        status="CANCELLED",
    )
    assert called == []


def test_notify_never_raises(monkeypatch):
    def boom(**kw):
        raise RuntimeError("lark down")

    monkeypatch.setattr(lark_client, "send_message", boom)
    # Must not raise even when the client raises.
    lark_notify.notify_lark_execution_failed(
        mission_id="agent_lark_me_mission",
        execution_input={"chat_id": "oc_1"},
        status="FAILED",
    )
