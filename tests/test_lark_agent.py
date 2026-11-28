"""LarkAgent scope — a Lark tool may only touch this run's own chat.

The agent instance is fresh per run; ``execution_input`` carries the
run's input (chat_id, sender, message/root ids). The wrapper refuses
with a structured error dict — the agent sees it as a tool result,
not a crash.
"""

import pytest

import runtime.tools.lark_calendar_tools  # noqa: F401
import runtime.tools.lark_tools  # noqa: F401 — registers the tools
from runtime.agents.registry import resolve_agent
from runtime.integrations import lark_client
from runtime.tools.registry import registry

INPUT = {
    "message": "ping",
    "chat_id": "oc_home",
    "chat_type": "p2p",
    "sender_id": {"union_id": "on_1", "user_id": "u_1", "open_id": "ou_1"},
    "message_id": "om_in",
    "root_id": "om_root",
}


@pytest.fixture
def agent():
    obj = resolve_agent("agent_lark_me", {})
    obj.execution_input = dict(INPUT)
    return obj


@pytest.fixture
def stub_client(monkeypatch):
    monkeypatch.setattr(
        lark_client,
        "send_message",
        lambda **kw: {"message_id": "om_sent", "chat_id": kw["receive_id"]},
    )
    monkeypatch.setattr(
        lark_client,
        "reply_message",
        lambda **kw: {"message_id": "om_reply", "root_id": "om_root"},
    )
    monkeypatch.setattr(
        lark_client,
        "create_group",
        lambda **kw: {"chat_id": "oc_new", "name": kw["name"]},
    )
    monkeypatch.setattr(
        lark_client,
        "get_group",
        lambda **kw: {"chat_id": kw["chat_id"], "name": "G"},
    )
    monkeypatch.setattr(
        lark_client,
        "update_group",
        lambda **kw: {"chat_id": kw["chat_id"], "ok": True},
    )
    monkeypatch.setattr(
        lark_client,
        "add_members",
        lambda **kw: {"chat_id": kw["chat_id"], "invalid_id_list": []},
    )
    monkeypatch.setattr(
        lark_client,
        "list_messages",
        lambda **kw: {"messages": [], "count": 0, "truncated": False},
    )


def _scoped(agent, name):
    entry = registry.get(name)
    assert entry is not None, name
    return agent.scope_tool(entry.tool, "lark-abc", "agent_lark_me_mission")


def test_non_lark_tool_passes_through_unchanged(agent):
    entry = registry.get("tavily_search")
    assert agent.scope_tool(entry.tool, "lark-abc") is entry.tool


def test_send_to_own_chat_allowed(agent, stub_client):
    tool = _scoped(agent, "lark_send_message")
    out = tool.invoke({"receive_id": "oc_home", "text": "hi"})
    assert out["message_id"] == "om_sent"


def test_send_to_other_chat_refused(agent, stub_client):
    tool = _scoped(agent, "lark_send_message")
    out = tool.invoke({"receive_id": "oc_elsewhere", "text": "hi"})
    assert "error" in out and "outside this run" in out["error"]


def test_dm_to_sender_allowed_other_user_refused(agent, stub_client):
    tool = _scoped(agent, "lark_send_message")
    out = tool.invoke(
        {"receive_id": "ou_1", "receive_id_type": "open_id", "text": "hi"}
    )
    assert "message_id" in out
    out = tool.invoke(
        {"receive_id": "ou_stranger", "receive_id_type": "open_id", "text": "hi"}
    )
    assert "error" in out


def test_created_group_becomes_allowed(agent, stub_client):
    create = _scoped(agent, "lark_create_group")
    out = create.invoke({"name": "incident"})
    assert out["chat_id"] == "oc_new"
    send = _scoped(agent, "lark_send_message")
    out = send.invoke({"receive_id": "oc_new", "text": "alert"})
    assert out["message_id"] == "om_sent"
    get = _scoped(agent, "lark_get_history")
    out = get.invoke({"chat_id": "oc_new"})
    assert out["count"] == 0


def test_update_group_only_for_created_groups(agent, stub_client):
    update = _scoped(agent, "lark_update_group")
    out = update.invoke({"chat_id": "oc_home", "name": "renamed"})
    assert "error" in out and "not created by this run" in out["error"]
    _scoped(agent, "lark_create_group").invoke({"name": "incident"})
    out = update.invoke({"chat_id": "oc_new", "name": "resolved"})
    assert out["ok"] is True


def test_reply_only_in_own_threads(agent, stub_client):
    reply = _scoped(agent, "lark_reply_message")
    out = reply.invoke({"message_id": "om_in", "text": "hi"})
    assert out["message_id"] == "om_reply"
    out = reply.invoke({"message_id": "om_root", "text": "hi"})
    assert out["message_id"] == "om_reply"
    out = reply.invoke({"message_id": "om_other", "text": "hi"})
    assert "error" in out


def test_calendar_freebusy_uses_sender_only(agent, monkeypatch):
    monkeypatch.setattr(
        lark_client,
        "get_freebusy",
        lambda **kw: {
            "busy_intervals": [],
            "count": 0,
            "user_open_id": kw["user_open_id"],
        },
    )
    entry = registry.get("lark_calendar_freebusy")
    tool = agent.scope_tool(entry.tool, "lark-abc", "agent_lark_me_mission")
    out = tool.invoke({"range": "today"})
    assert out["count"] == 0

    bare = registry.get("lark_calendar_freebusy").tool
    out = bare.invoke({"range": "today"})
    assert "error" in out


def test_calendar_freebusy_refused_without_sender(stub_client):
    obj = resolve_agent("agent_lark_me", {})
    obj.execution_input = {"chat_id": "oc_home"}
    entry = registry.get("lark_calendar_freebusy")
    tool = obj.scope_tool(entry.tool, "lark-abc")
    out = tool.invoke({"range": "today"})
    assert "error" in out and "open_id" in out["error"]


def test_empty_input_is_fail_closed(stub_client):
    obj = resolve_agent("agent_lark_me", {})
    obj.execution_input = {}
    entry = registry.get("lark_send_message")
    tool = obj.scope_tool(entry.tool, "lark-abc")
    out = tool.invoke({"receive_id": "oc_home", "text": "hi"})
    assert "error" in out
