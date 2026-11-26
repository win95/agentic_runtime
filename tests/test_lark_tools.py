"""Lark tools — registration + behaviour with a stubbed client.

The Lark client is stubbed (no credentials, no network); the tests
cover the registry contract (capability, read-only flags, destructive
target) and the tools' error passthrough.
"""

import runtime.tools.lark_tools as lark_tools
from runtime.integrations import lark_client
from runtime.tools.registry import registry

ALL_LARK_TOOLS = {
    "lark_create_group",
    "lark_send_message",
    "lark_get_group",
    "lark_upload_image",
    "lark_add_members",
    "lark_reply_message",
    "lark_get_history",
    "lark_update_group",
}

READ_ONLY = {"lark_get_group", "lark_get_history"}


def test_all_eight_lark_tools_registered():
    entries = {
        e.name: e
        for e in registry._tools.values()
        if e.capability == "lark_messaging"
    }
    assert set(entries) == ALL_LARK_TOOLS
    for name, entry in entries.items():
        assert entry.read_only is (name in READ_ONLY), name


def test_update_group_carries_destructive_target():
    entry = registry.get("lark_update_group")
    assert entry is not None
    assert entry.destructive_target_args == ("chat_id",)


def test_send_message_returns_client_result(monkeypatch):
    monkeypatch.setattr(
        lark_client,
        "send_message",
        lambda **kw: {"message_id": "om_1", "chat_id": "oc_1", "root_id": "om_1"},
    )
    out = lark_tools.lark_send_message.invoke(
        {"receive_id": "oc_1", "text": "hello"}
    )
    assert out["message_id"] == "om_1"
    assert out["provenance"] == "lark:send_message"


def test_send_message_passes_client_error_through(monkeypatch):
    monkeypatch.setattr(
        lark_client, "send_message", lambda **kw: {"error": "lark boom"}
    )
    out = lark_tools.lark_send_message.invoke(
        {"receive_id": "oc_1", "text": "hello"}
    )
    assert out["error"] == "lark boom"


def test_send_message_rejects_overlong_text():
    out = lark_tools.lark_send_message.invoke(
        {"receive_id": "oc_1", "text": "x" * 4001}
    )
    assert "error" in out and "too long" in out["error"]


def test_get_history_token_guard_truncates_oldest_first(monkeypatch):
    # 100 messages x 500 chars (~135 tokens each) >> 4000 budget:
    # only the newest ~29 fit.
    messages = [
        {
            "message_id": f"om_{i}",
            "sender_id": "ou_1",
            "create_time": 1000 + i,
            "message_type": "text",
            "content": "x" * 500,
        }
        for i in range(100)
    ]
    monkeypatch.setattr(
        lark_client,
        "list_messages",
        lambda **kw: {"messages": messages, "count": 100, "truncated": False},
    )
    out = lark_tools.lark_get_history.invoke({"chat_id": "oc_1"})
    assert out["truncated"] is True
    assert "note" in out and "4000" in out["note"]
    # Newest messages kept, oldest dropped.
    assert out["messages"][0]["message_id"] != "om_0"
    assert out["messages"][-1]["message_id"] == "om_99"
    total = sum(len(m["content"]) // 4 + 10 for m in out["messages"])
    assert total <= 4000


def test_get_history_short_history_not_truncated(monkeypatch):
    messages = [
        {
            "message_id": "om_1",
            "sender_id": "ou_1",
            "create_time": 1000,
            "message_type": "text",
            "content": "hello",
        }
    ]
    monkeypatch.setattr(
        lark_client,
        "list_messages",
        lambda **kw: {"messages": messages, "count": 1, "truncated": False},
    )
    out = lark_tools.lark_get_history.invoke({"chat_id": "oc_1"})
    assert out["truncated"] is False and out["count"] == 1
    assert "note" not in out


def test_get_history_caps_limit_at_client(monkeypatch):
    seen: dict = {}

    def fake_list_messages(**kw):
        seen.update(kw)
        return {"messages": [], "count": 0, "truncated": False}

    monkeypatch.setattr(lark_client, "list_messages", fake_list_messages)
    lark_tools.lark_get_history.invoke({"chat_id": "oc_1", "limit": 500})
    assert seen["limit"] == 500  # the tool passes it on; the client caps at 100


def test_client_list_messages_caps_at_100(monkeypatch):
    # The cap lives in the client (D-LARK-20): a 500-limit request
    # still pages at most 100 messages.
    monkeypatch.setattr(lark_client, "get_client", lambda: _FakeClient(pages=3))
    out = lark_client.list_messages(chat_id="oc_1", limit=500)
    assert out["count"] == 100
    assert out["truncated"] is True


class _FakeMessage:
    def __init__(self, i):
        self.message_id = f"om_{i}"
        self.create_time = 1000 + i
        self.msg_type = "text"
        self.sender = type("S", (), {"id": "ou_1"})()
        self.body = type("B", (), {"content": "hi"})()


class _FakeListData:
    def __init__(self, items, has_more, page_token):
        self.items = items
        self.has_more = has_more
        self.page_token = page_token


class _FakeListResp:
    def __init__(self, data):
        self.code = 0
        self.msg = "ok"
        self.data = data


class _FakeMessageAPI:
    def __init__(self, pages):
        self.pages = pages
        self.calls = 0

    def list(self, req):
        self.calls += 1
        start = (self.calls - 1) * 50
        items = [_FakeMessage(i) for i in range(start, start + 50)]
        has_more = self.calls < self.pages
        return _FakeListResp(
            _FakeListData(items, has_more, f"tok{self.calls}" if has_more else None)
        )


class _FakeClient:
    def __init__(self, pages=3):
        self.im = type("IM", (), {})()
        self.im.v1 = type("V1", (), {})()
        self.im.v1.message = _FakeMessageAPI(pages)
