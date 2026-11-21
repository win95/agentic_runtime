"""BaseAgent core + capability registry (plan Phase 0–1).

Main-feature coverage only: definition/compile wiring, agent
resolution, capability registration, and the capability node routing
through the registry instead of the old hardcoded dicts.
"""

import runtime.deep_agent as deep_agent_module
import runtime.mission_graph as mission_graph_module
from runtime.agents.base import AgentDefinition, BaseAgent
from runtime.agents.registry import (
    canonical_capability_name,
    register_agent_class,
    register_capability,
    resolve_agent,
    resolve_capability,
)
from runtime.mission_graph import capability_node, run_mission


def test_definition_carries_mda_fields():
    agent = BaseAgent("fault_investigator", {"mission_id": "fault_analysis"})
    definition = agent.definition(
        tools=["t1"],
        instructions="prompt",
        skills=["/skills/"],
        model_config="cfg",
        middleware=["m"],
        backend="b",
    )
    assert isinstance(definition, AgentDefinition)
    assert definition.name == "fault_investigator"
    assert definition.instructions == "prompt"
    assert definition.tools == ["t1"]
    assert definition.skills == ["/skills/"]
    assert definition.model_config == "cfg"
    assert definition.middleware == ["m"]
    assert definition.backend == "b"
    assert agent.name == "fault_investigator"


def test_compile_passes_only_configured_kwargs():
    seen: dict = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return object()

    agent = BaseAgent("a", {})
    definition = agent.definition(tools=["t"], instructions="p")
    agent.compile(definition, build_fn=fake_build)
    # The always-passed kwargs (tools/system_prompt/skills/middleware/
    # backend) reach the builder; the configured-only ones do not.
    assert seen == {
        "tools": ["t"],
        "system_prompt": "p",
        "skills": None,
        "middleware": None,
        "backend": None,
    }

    seen.clear()
    definition.response_format = {"type": "object", "properties": {"a": {}}}
    definition.interrupt_on = {"delete_mission": True}
    definition.checkpointer = "cp"
    definition.subagents = ["sub"]  # declared, never passed to the builder
    agent.compile(definition, build_fn=fake_build)
    assert seen["response_format"] == {"type": "object", "properties": {"a": {}}}
    assert seen["interrupt_on"] == {"delete_mission": True}
    assert seen["checkpointer"] == "cp"
    assert "subagents" not in seen
    assert "context_schema" not in seen


def test_resolve_agent_defaults_to_base():
    agent = resolve_agent("no_such_agent", {"mission_id": "x"})
    assert isinstance(agent, BaseAgent)
    assert agent.agent_id == "no_such_agent"
    assert agent.name == "no_such_agent"


def test_registered_agent_class_resolves():
    class TestAgent(BaseAgent):
        pass

    register_agent_class("test_agent_phase0", TestAgent)
    agent = resolve_agent("test_agent_phase0", {"mission_id": "x"})
    assert isinstance(agent, TestAgent)


def test_capability_registry_aliases():
    calls: list = []

    @register_capability("test_cap", aliases=("test_alias_a", "test_alias_b"))
    def _builder():
        calls.append("built")
        return object()

    assert canonical_capability_name("test_cap") == "test_cap"
    assert canonical_capability_name("test_alias_a") == "test_cap"
    assert canonical_capability_name("test_alias_b") == "test_cap"
    assert canonical_capability_name("nope") is None
    assert resolve_capability("test_alias_a") is _builder
    assert resolve_capability("nope") is None
    resolve_capability("test_cap")()
    assert calls == ["built"]


def test_fault_capability_registered_with_aliases():
    # Importing mission_graph pulls in fault_analysis, whose decorator
    # registers the subgraph — the old CAPABILITY_BUILDERS dict is gone.
    assert not hasattr(mission_graph_module, "CAPABILITY_BUILDERS")
    assert canonical_capability_name("fault_analysis") == "fault_analysis"
    assert canonical_capability_name("sensor_data") == "fault_analysis"
    assert canonical_capability_name("asset_relations") == "fault_analysis"
    assert resolve_capability("fault_history") is not None


class _FakeSubgraph:
    runs: list = []

    def invoke(self, state):
        _FakeSubgraph.runs.append(state)
        return {
            "evidence": [
                {"source": "test_cap", "kind": "summary", "summary": "ok"}
            ],
            "result": {"answer": "fake"},
        }


def test_capability_node_uses_registry(monkeypatch):
    monkeypatch.setattr(_FakeSubgraph, "runs", [])
    register_capability("test_node_cap", _FakeSubgraph, aliases=("test_node_alias",))
    state = {
        "mission": {"capabilities": ["test_node_cap", "test_node_alias"]},
        "evidence": [],
    }
    updates = capability_node(state)
    # Both names route to the same subgraph — it runs once (dedup).
    assert len(_FakeSubgraph.runs) == 1
    assert updates["result"] == {"answer": "fake"}
    assert updates["evidence"][0]["source"] == "test_cap"
    assert updates["errors"] == []


def test_capability_node_unknown_capability_is_an_error():
    state = {"mission": {"capabilities": ["nope"]}, "evidence": []}
    updates = capability_node(state)
    assert updates["errors"] == ["Capability not implemented: nope"]


def test_node_compiles_through_agent(monkeypatch):
    seen: dict = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return object()

    def fake_run(agent, mission_arg, input_data, callbacks=None, files=None, **kwargs):
        return {
            "summary": "ok",
            "evidence": [
                {"source": "deep_agent", "kind": "summary", "summary": "x", "stub": False}
            ],
            "actions_taken": [],
            "errors": [],
        }

    # The doubles patch runtime.deep_agent (the node imports
    # build_deep_agent lazily at call time), and compile() receives
    # that same fake through build_fn.
    monkeypatch.setattr(deep_agent_module, "build_deep_agent", fake_build)
    monkeypatch.setattr(deep_agent_module, "run_deep_agent", fake_run)
    mission = {
        "mission_id": "fault_analysis",
        "agent_id": "fault_investigator",
        "objective": "Test BaseAgent wiring",
        "capabilities": [],
        "execution": {"mode": "autonomous"},
    }
    final = run_mission(mission, {}, "exec_agents_1")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    # The node reached the builder through BaseAgent.compile with the
    # same kwargs it used to pass directly.
    assert seen["system_prompt"].startswith("You are")
    assert "tools" in seen
    assert "skills" in seen
    assert "middleware" in seen
    assert "backend" in seen
    assert "model_config" not in seen  # no active AgentVersion here
    assert "interrupt_on" not in seen
    assert "response_format" not in seen
