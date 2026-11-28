"""Lark Task API client + tasklist-task tool (unit, stubbed SDK)."""

from __future__ import annotations

import json

import pytest

from runtime.integrations import lark_client
from runtime.integrations.lark_client import TASK_EMPTY_VISIBILITY_NOTE
import runtime.tools.lark_task_tools as lark_task_tools
from runtime.agents.registry import resolve_agent
from runtime.tools.registry import registry


class _FakeTask:
    def __init__(self, i: int):
        self.guid = f"task_{i}"
        self.summary = f"Task {i}"
        self.due = None
        self.status = "todo"
        self.completed_at = None
        self.members = []
        self.creator = None
        self.url = f"https://tasks/{i}"
        self.tasklists = []
        self.created_at = None
        self.updated_at = None


class _FakeListData:
    def __init__(self, items, has_more, page_token):
        self.items = items
        self.has_more = has_more
        self.page_token = page_token


class _FakeResp:
    def __init__(self, data):
        self.code = 0
        self.msg = "ok"
        self.data = data


class _FakeTasklistTasksAPI:
    def __init__(self, pages: int):
        self.pages = pages
        self.calls = 0
        self.last_req = None

    def tasks(self, req):
        self.calls += 1
        self.last_req = req
        start = (self.calls - 1) * 50
        items = [_FakeTask(i) for i in range(start, start + 50)]
        has_more = self.calls < self.pages
        return _FakeResp(
            _FakeListData(items, has_more, f"tok{self.calls}" if has_more else None)
        )


class _FakeClient:
    def __init__(self, pages: int = 1):
        self.task = type("T", (), {})()
        self.task.v2 = type("V2", (), {})()
        self.task.v2.tasklist = _FakeTasklistTasksAPI(pages)


def test_list_tasklist_tasks_pagination_and_cap(monkeypatch):
    fake = _FakeClient(pages=3)
    monkeypatch.setattr(lark_client, "get_client", lambda: fake)
    out = lark_client.list_tasklist_tasks(tasklist_guid="tl_1", limit=100)
    assert out["count"] == 100
    assert out["truncated"] is True
    assert out["tasklist_guid"] == "tl_1"
    assert fake.task.v2.tasklist.calls == 2


def test_list_tasklist_tasks_sdk_error(monkeypatch):
    class _ErrResp:
        code = 99991679
        msg = "scope required"
        data = None

    class _Api:
        def tasks(self, req):  # noqa: ARG002
            return _ErrResp()

    client = type("C", (), {})()
    client.task = type("T", (), {})()
    client.task.v2 = type("V2", (), {})()
    client.task.v2.tasklist = _Api()
    monkeypatch.setattr(lark_client, "get_client", lambda: client)
    out = lark_client.list_tasklist_tasks(tasklist_guid="tl_x")
    assert "error" in out
    assert "99991679" in out["error"] or "scope" in out["error"].lower()


def test_list_tasklists_empty_includes_visibility_note(monkeypatch):
    from lark_oapi.api.task.v2 import model

    class _TasklistAPI:
        def list(self, req):  # noqa: ARG002
            return _FakeResp(_FakeListData([], False, None))

    client = type("C", (), {})()
    client.task = type("T", (), {})()
    client.task.v2 = type("V2", (), {})()
    client.task.v2.tasklist = _TasklistAPI()
    monkeypatch.setattr(lark_client, "_task_model", lambda: (client, model))
    out = lark_client.list_tasklists(limit=10)
    assert out["count"] == 0
    assert out.get("visibility_note") == TASK_EMPTY_VISIBILITY_NOTE


def test_lark_list_tasklist_tasks_registered():
    entry = registry.get("lark_list_tasklist_tasks")
    assert entry is not None
    assert entry.capability == "lark_tasks"


def test_tasklist_guid_scope_requires_prior_list_tasklists(monkeypatch):
    monkeypatch.setattr(
        lark_client,
        "list_tasklist_tasks",
        lambda **kw: {"tasks": [], "count": 0, "tasklist_guid": kw["tasklist_guid"]},
    )
    monkeypatch.setattr(
        lark_client,
        "list_tasklists",
        lambda **kw: {"tasklists": [{"guid": "tl_ok"}], "count": 1, "truncated": False},
    )
    agent = resolve_agent("agent_lark_me", {})
    agent.execution_input = {"chat_id": "oc_1", "message_id": "om_1"}
    scoped_lists = agent.scope_tool(registry.get("lark_list_tasklists").tool, "")
    scoped_tasks = agent.scope_tool(
        registry.get("lark_list_tasklist_tasks").tool, ""
    )
    scoped_lists.invoke({"limit": 10})
    out = scoped_tasks.invoke({"tasklist_guid": "tl_ok"})
    assert "error" not in out
    blocked = scoped_tasks.invoke({"tasklist_guid": "tl_other"})
    assert "refused" in blocked.get("error", "")