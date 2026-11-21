"""BaseAgent — the shared core every agent inherits (plan §3.2, §8.3).

MDA-shaped: each agent class is one ``define_deep_agent`` definition
(``name`` / ``model`` / ``instructions`` / ``skills`` / ``tools`` /
``middleware`` / ``subagents``, plus ``interrupt_on`` /
``response_format`` / ``context_schema`` when configured — see
docs/agent-definition.md). ``BaseAgent`` is the definition resolver
plus the single compile point; it holds NO agent-specific logic.
Per-agent subclasses only declare what differs from the default and
override the methods listed in §3.3 of the plan — hotel scope, write
pipeline, guest channel, capability subgraph, and the like stay in
their own modules until their phase moves them here.

The agent's decisions still live in Skill + Rule (YAML / skill map).
This class only executes what YAML/skills cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentDefinition:
    """One agent definition, in MDA ``define_deep_agent`` vocabulary.

    ``name`` is the static agent id (``agent_id``). ``instructions`` is
    the system prompt text. ``context_schema`` is declared here but
    deliberately NOT passed to ``create_deep_agent`` — the invoke
    carries no context object today, so there is nothing for the
    graph to type-check (docs/agent-definition.md). ``subagents`` is
    likewise declared, not passed: ``build_deep_agent`` derives the
    subagent from middleware + skills itself.
    """

    name: str
    instructions: str | None = None
    model_config: Any = None
    skills: list | None = None
    tools: list = field(default_factory=list)
    middleware: list | None = None
    backend: Any = None
    subagents: list | None = None
    interrupt_on: Any = None
    checkpointer: Any = None
    response_format: Any = None
    context_schema: Any = None


class BaseAgent:
    """Thin wrapper + override points around the existing assembly chain.

    Phase 0: wraps exactly what ``deep_agent_node`` already does —
    ``definition()`` carries the values the node computed, ``compile()``
    is the one place that calls ``build_deep_agent``. No logic moves
    here yet; subclasses override one method at a time in later phases.
    """

    agent_id: str = ""

    def __init__(self, agent_id: str, mission: dict, agent_version: Any = None):
        self.agent_id = agent_id
        self.mission = mission
        self.agent_version = agent_version

    @property
    def name(self) -> str:
        """The MDA static name — the agent id."""
        return self.agent_id

    def definition(
        self,
        *,
        tools: list,
        instructions: str | None = None,
        skills: list | None = None,
        model_config: Any = None,
        middleware: list | None = None,
        backend: Any = None,
        subagents: list | None = None,
        interrupt_on: Any = None,
        checkpointer: Any = None,
        response_format: Any = None,
        context_schema: Any = None,
    ) -> AgentDefinition:
        """Build this run's definition from the node's computed values."""
        return AgentDefinition(
            name=self.agent_id,
            instructions=instructions,
            model_config=model_config,
            skills=skills,
            tools=tools,
            middleware=middleware,
            backend=backend,
            subagents=subagents,
            interrupt_on=interrupt_on,
            checkpointer=checkpointer,
            response_format=response_format,
            context_schema=context_schema,
        )

    def compile(self, definition: AgentDefinition, build_fn: Any = None):
        """Compile the definition into a deep agent — the single place
        that calls ``build_deep_agent``.

        Only configured kwargs are passed: test doubles for
        ``build_deep_agent`` have fixed signatures, so an explicit
        ``interrupt_on=None`` (or ``model_config=None`` on the
        no-version path) would break them with "unexpected keyword
        argument". ``build_fn`` defaults to the real builder; the
        Mission Graph passes its own module-level name, which is the
        name the test doubles patch.
        """
        if build_fn is None:
            from runtime.deep_agent import build_deep_agent

            build_fn = build_deep_agent
        # The always-passed kwargs mirror what deep_agent_node used to
        # build by hand: tools, system_prompt, skills, middleware and
        # backend are always present (even None — doubles and the
        # sandbox wiring tests read them by key). The rest are only
        # added when configured, so older build_deep_agent doubles
        # (fixed signatures) keep working on the common path.
        kwargs: dict[str, Any] = {
            "tools": definition.tools,
            "system_prompt": definition.instructions,
            "skills": definition.skills,
            "middleware": definition.middleware,
            "backend": definition.backend,
        }
        for key, value in (
            ("model_config", definition.model_config),
            ("interrupt_on", definition.interrupt_on),
            ("checkpointer", definition.checkpointer),
            ("response_format", definition.response_format),
        ):
            if value is not None:
                kwargs[key] = value
        return build_fn(**kwargs)

    def capability_builder(self, capability: str):
        """The subgraph builder for one capability name, or None."""
        from runtime.agents.registry import resolve_capability

        return resolve_capability(capability)
