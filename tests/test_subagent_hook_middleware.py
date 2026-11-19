"""Subagent must inherit HookMiddleware (deepagents 0.7.x)."""

from __future__ import annotations

from runtime.deep_agent import _subagents_inherit_main_middleware
from runtime.hooks import HookMiddleware, default_registry


def test_subagents_spec_includes_hook_middleware():
    mw = HookMiddleware(default_registry(), execution_id="e1", mission_id="m1")
    specs = _subagents_inherit_main_middleware([mw])
    assert specs is not None
    gp = specs[0]
    assert gp["name"] == "general-purpose"
    assert any(isinstance(m, HookMiddleware) for m in gp["middleware"])


def test_subagents_spec_none_without_hook_middleware():
    assert _subagents_inherit_main_middleware([]) is None
    assert _subagents_inherit_main_middleware(None) is None


def test_create_deep_agent_receives_general_purpose_subagents():
    """build_deep_agent wires subagents= when HookMiddleware is present."""
    from unittest.mock import patch

    from app.config import Settings
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
        da.build_deep_agent([], middleware=[mw], model_config=cfg)

    subagents = seen.get("subagents") or []
    assert subagents and subagents[0]["name"] == "general-purpose"