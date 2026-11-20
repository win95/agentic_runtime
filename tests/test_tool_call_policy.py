"""Harness tool sequencing and web-result budgets."""

from __future__ import annotations

from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage

from runtime.context_budget import estimate_tokens
from runtime.deep_agent import build_deep_agent
from runtime.hooks import builtin
from runtime.hooks.events import HookEvent
from runtime.hooks.middleware import HookMiddleware, SharedRunCounters
from runtime.hooks.registry import HookRegistry
from runtime.harness.loader import parse_mission_harness
from runtime.tool_call_policy import (
    TRUNCATION_MARKER,
    apply_tool_result_budget,
    sequential_defer_message,
    truncate_text_to_token_budget,
)


def _req(name: str, call_id: str, ai: AIMessage) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": call_id},
        tool=None,
        state={"messages": [AIMessage(content="hi"), ai]},
        runtime=None,
    )


def test_sequential_defers_second_parallel_tavily_call():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "tavily_search", "args": {"q": "a"}, "id": "c1"},
            {"name": "tavily_extract", "args": {"url": "u"}, "id": "c2"},
        ],
    )
    seq = frozenset({"tavily_search", "tavily_extract"})
    assert sequential_defer_message(_req("tavily_search", "c1", ai), tool_name="tavily_search", tool_call_id="c1", sequential_tools=seq) is None
    defer = sequential_defer_message(
        _req("tavily_extract", "c2", ai),
        tool_name="tavily_extract",
        tool_call_id="c2",
        sequential_tools=seq,
    )
    assert isinstance(defer, ToolMessage)
    assert "one at a time" in str(defer.content)
    assert defer.tool_call_id == "c2"


def test_sequential_disabled_by_default_hotel():
    from pathlib import Path

    from runtime.harness.loader import load_mission_harness_from_file

    path = Path(__file__).resolve().parents[1] / "seeds/missions/hotel_reception.yaml"
    harness = load_mission_harness_from_file(path)
    assert harness.tool_calls.sequential == []
    assert harness.tool_budget.max_result_tokens == 0


def test_truncate_text_to_token_budget():
    text = "x" * 40000
    out = truncate_text_to_token_budget(text, 100)
    assert TRUNCATION_MARKER in out
    assert estimate_tokens(out) <= 100


def test_truncate_preserves_prefix_and_single_marker():
    prefix = "KEEP_ME_" + ("word " * 4000)
    suffix = "DROP_ME_" + ("tail " * 4000)
    text = prefix + suffix
    assert estimate_tokens(text) > 500
    out = truncate_text_to_token_budget(text, 500)
    assert out.startswith("KEEP_ME_")
    assert "DROP_ME_" not in out
    assert out.count(TRUNCATION_MARKER) == 1
    assert estimate_tokens(out) <= 500


def test_truncate_huge_budget_no_marker_storm():
    text = "a" * (17000 * 4)
    out = truncate_text_to_token_budget(text, 500)
    assert out.endswith(TRUNCATION_MARKER)
    assert out.count(TRUNCATION_MARKER) == 1
    assert out.startswith("a")
    assert estimate_tokens(out) <= 500


def test_wrap_tool_call_budget_truncation():
    registry = HookRegistry()
    registry.register(HookEvent.STOP, builtin.stop_completion, name="stop_completion")
    shared = SharedRunCounters()
    mw = HookMiddleware(
        registry,
        execution_id="e1",
        mission_id="agent_analytis_mission",
        tool_budget_tools=frozenset({"tavily_search"}),
        tool_budget_max_result_tokens=50,
        tool_budget_max_calls={},
        shared_counters=shared,
    )
    huge = "word " * 5000

    def handler(_request):
        return ToolMessage(content=huge, tool_call_id="t1")

    result = mw.wrap_tool_call(
        _req("tavily_search", "t1", AIMessage(content="", tool_calls=[])),
        handler,
    )
    assert isinstance(result, ToolMessage)
    assert TRUNCATION_MARKER in str(result.content)
    assert shared.tool_budget_tokens_used <= 50


def test_wrap_tool_call_cap_exhaustion():
    registry = HookRegistry()
    mw = HookMiddleware(
        registry,
        execution_id="e1",
        mission_id="m",
        tool_budget_tools=frozenset({"tavily_search"}),
        tool_budget_max_result_tokens=6000,
        tool_budget_max_calls={"tavily_search": 1},
        shared_counters=SharedRunCounters(),
    )
    ai = AIMessage(content="", tool_calls=[{"name": "tavily_search", "args": {}, "id": "a"}])

    def handler(_request):
        return ToolMessage(content="ok", tool_call_id="a")

    mw.wrap_tool_call(_req("tavily_search", "a", ai), handler)
    blocked = mw.wrap_tool_call(
        _req("tavily_search", "b", ai), handler,
    )
    assert isinstance(blocked, ToolMessage)
    assert "not executed" in str(blocked.content).lower()


def test_shared_budget_between_main_and_subagent():
    registry = HookRegistry()
    shared = SharedRunCounters()
    parent = HookMiddleware(
        registry,
        execution_id="e1",
        mission_id="m",
        tool_budget_tools=frozenset({"tavily_search"}),
        tool_budget_max_result_tokens=100,
        tool_budget_max_calls={},
        shared_counters=shared,
    )
    sub = HookMiddleware.for_subagent(parent)
    ai = AIMessage(content="", tool_calls=[{"name": "tavily_search", "args": {}, "id": "s1"}])

    def handler(_request):
        return ToolMessage(content="y" * 200, tool_call_id="s1")

    sub.wrap_tool_call(_req("tavily_search", "s1", ai), handler)
    assert shared.tool_budget_tokens_used > 0
    assert sub.shared_counters is parent.shared_counters


def test_middleware_parallel_only_first_sequential_runs():
    registry = HookRegistry()
    mw = HookMiddleware(
        registry,
        execution_id="e1",
        mission_id="m",
        sequential_tool_names=frozenset({"tavily_search", "tavily_extract"}),
    )
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "tavily_search", "args": {}, "id": "1"},
            {"name": "tavily_extract", "args": {}, "id": "2"},
        ],
    )
    ran: list[str] = []

    def handler(request):
        ran.append(str((request.tool_call or {}).get("name")))
        return ToolMessage(
            content="ok",
            tool_call_id=str((request.tool_call or {}).get("id")),
        )

    r1 = mw.wrap_tool_call(_req("tavily_search", "1", ai), handler)
    r2 = mw.wrap_tool_call(_req("tavily_extract", "2", ai), handler)
    assert ran == ["tavily_search"]
    assert isinstance(r2, ToolMessage)
    assert "one at a time" in str(r2.content)


def test_harness_tool_budget_rejects_negative_token_cap():
    import pytest

    with pytest.raises(ValueError):
        parse_mission_harness({"tool_budget": {"tools": ["t"], "max_result_tokens": -1}})


def test_call_cap_reserve_is_atomic_under_threads():
    import threading

    registry = HookRegistry()
    shared = SharedRunCounters()
    mw = HookMiddleware(
        registry,
        execution_id="e1",
        mission_id="m",
        tool_budget_tools=frozenset({"tavily_search"}),
        tool_budget_max_result_tokens=0,
        tool_budget_max_calls={"tavily_search": 2},
        shared_counters=shared,
    )
    lock = threading.Lock()
    results: list[str] = []

    def handler(request):
        with lock:
            results.append(str((request.tool_call or {}).get("id")))
        return ToolMessage(
            content="ok",
            tool_call_id=str((request.tool_call or {}).get("id")),
        )

    def run_one(cid: str) -> None:
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "tavily_search", "args": {}, "id": cid}],
        )
        mw.wrap_tool_call(_req("tavily_search", cid, ai), handler)

    threads = [threading.Thread(target=run_one, args=(cid,)) for cid in ("t1", "t2", "t3")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 2
    assert shared.tool_budget_calls["tavily_search"] == 2


def test_build_deep_agent_uses_real_create_deep_agent_with_sequential():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    class _BindableFakeModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

    mw = HookMiddleware(
        HookRegistry(),
        execution_id="e1",
        mission_id="agent_analytis_mission",
        sequential_tool_names=frozenset({"tavily_search", "tavily_extract"}),
    )
    model = _BindableFakeModel(messages=iter([AIMessage(content="ok")]))
    from unittest.mock import patch

    with patch("runtime.deep_agent.build_chat_model", return_value=model):
        agent = build_deep_agent(
            [],
            system_prompt="t",
            middleware=[mw],
        )
    assert agent is not None


def test_apply_tool_result_budget_charges_once():
    shared = SharedRunCounters()
    msg = ToolMessage(content="hello world", tool_call_id="x")
    out = apply_tool_result_budget(
        msg,
        shared_counters=shared,
        tool_name="tavily_search",
        budget_tools=frozenset({"tavily_search"}),
        max_result_tokens=6000,
        max_calls={},
    )
    assert out.content == "hello world"
    assert shared.tool_budget_tokens_used == estimate_tokens("hello world")
    assert shared.tool_budget_calls["tavily_search"] == 1