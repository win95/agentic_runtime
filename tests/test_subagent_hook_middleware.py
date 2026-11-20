"""Subagent must inherit HookMiddleware + skills (deepagents 0.7.x)."""

from __future__ import annotations

from unittest.mock import patch

from app.config import Settings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from runtime.deep_agent import _subagents_inherit_main_middleware, build_deep_agent
from runtime.hooks import HookMiddleware, default_registry
from runtime.hooks.events import HookDecision, HookEvent
from runtime.hooks.middleware import builtin_subagent_hook_names
from runtime.hooks.registry import HookRegistry


def test_subagent_excludes_output_files_stop_nudge_config():
    parent = HookMiddleware(
        default_registry(),
        execution_id="e1",
        mission_id="agent_analytis_mission",
        output_files_required=True,
        sandbox_backend=object(),
        export_marker_path="/tmp/.run_x",
    )
    sub = HookMiddleware.for_subagent(parent)
    assert sub.output_files_required is False
    assert sub.sandbox_backend is None
    assert sub.export_marker_path is None
    assert sub._output_files_stop_status() == "skipped"


def test_subagents_spec_includes_subagent_scoped_hook_middleware():
    mw = HookMiddleware(default_registry(), execution_id="e1", mission_id="m1")
    specs = _subagents_inherit_main_middleware([mw], skills=["/skills/"])
    assert specs is not None
    gp = specs[0]
    assert gp["name"] == "general-purpose"
    assert gp.get("skills") == ["/skills/"]
    sub_mw = gp["middleware"][0]
    assert isinstance(sub_mw, HookMiddleware)
    assert sub_mw.subagent_mode is True
    assert "stop_completion" not in sub_mw.registry.handler_names()


def test_subagents_spec_none_without_hook_middleware():
    assert _subagents_inherit_main_middleware([]) is None
    assert _subagents_inherit_main_middleware(None) is None


def test_create_deep_agent_receives_general_purpose_subagents():
    """build_deep_agent wires subagents= when HookMiddleware is present."""
    from runtime import deep_agent as da

    mw = HookMiddleware(default_registry(), execution_id="e1", mission_id="m1")
    seen: dict = {}
    cfg = Settings(
        llm_api_base="http://localhost:4000",
        llm_api_key="test-key",
        llm_model="test-model",
    )

    def fake_create(**kwargs):
        seen.update(kwargs)
        return object()

    with patch("deepagents.create_deep_agent", side_effect=fake_create):
        da.build_deep_agent(
            [],
            middleware=[mw],
            model_config=cfg,
            skills=["/skills/"],
        )

    subagents = seen.get("subagents") or []
    assert subagents and subagents[0]["name"] == "general-purpose"
    assert subagents[0].get("skills") == ["/skills/"]


def test_subagent_wrap_tool_call_runs_pre_tool_hooks():
    from types import SimpleNamespace

    registry = HookRegistry()

    def deny_ls(ctx):
        if ctx.tool_name == "ls":
            return HookDecision(
                action="deny", message="ls blocked for subagent test"
            )
        return None

    registry.register(HookEvent.PRE_TOOL_USE, deny_ls, name="enforce_layer")
    parent = HookMiddleware(registry, execution_id="e-deny", mission_id="m1", max_continues=0)
    sub = HookMiddleware.for_subagent(parent)
    request = SimpleNamespace(
        tool_call={"name": "ls", "args": {"path": "/"}, "id": "c1"}
    )

    result = sub.wrap_tool_call(
        request,
        lambda _req: ToolMessage(content="listed", tool_call_id="c1"),
    )
    assert isinstance(result, ToolMessage)
    assert "ls blocked" in str(result.content)
    assert any(e.get("tool") == "ls" for e in parent.events)


def test_compiled_create_deep_agent_accepts_gp_spec_with_skills_and_hooks():
    from deepagents import create_deep_agent

    class _BindableFakeModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

    parent = HookMiddleware(default_registry(), execution_id="e-g", mission_id="m1")
    specs = _subagents_inherit_main_middleware([parent], skills=["/skills/"])
    model = _BindableFakeModel(messages=iter([AIMessage(content="done")]))
    agent = create_deep_agent(
        model=model,
        tools=[],
        system_prompt="t",
        middleware=[parent],
        subagents=specs,
        skills=["/skills/"],
    )
    assert specs and specs[0].get("skills") == ["/skills/"]
    assert isinstance(specs[0]["middleware"][0], HookMiddleware)
    assert agent is not None


def test_builtin_subagent_hook_names_derived_from_tool_events():
    names = builtin_subagent_hook_names()
    assert "enforce_layer" in names
    assert "stop_completion" not in names
    assert "retrieval_context" not in names
    assert "validate_input" not in names


def test_subagent_before_model_trim_does_not_encode_ltm(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    encoded: list = []

    def _fake_encode(*args, **kwargs):  # noqa: ARG001
        encoded.append(args)
        return 0

    monkeypatch.setattr("runtime.hooks.middleware.encode_texts", _fake_encode)
    monkeypatch.setattr(
        "runtime.hooks.middleware.trim_history",
        lambda _messages, _budget: [AIMessage(content="drop-me")],
    )
    parent = HookMiddleware(
        default_registry(), execution_id="e1", mission_id="m1", session_id="s1"
    )
    sub = HookMiddleware.for_subagent(parent)
    sub.before_model(
        {"messages": [HumanMessage(content="hi"), AIMessage(content="tail")]},
        None,
    )
    assert encoded == []