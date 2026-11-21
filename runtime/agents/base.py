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

from runtime.grounding_config import GroundingConfig


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
    # Grounding sources for this agent's guest replies (Phase 4). The
    # default is empty — no quote/policy grounding — and only a
    # subclass that declares its sources (HotelAgent) gets it.
    grounding_config: GroundingConfig = GroundingConfig()

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

    # -- Override points (Phase 2 + 3) -----------------------------------
    # The default is "no policy": the agent does not scope tools and
    # its write pipelines are the generic engine as-is. Subclasses
    # override the pieces that are theirs (hotel scope, room rates,
    # guest-facing wording) — the core never names them.

    def scope_tool(self, tool, session_id: str, mission_id: str = ""):
        """Wrap one tool with this agent's data-scope policy.

        Default: no policy, the tool is returned unchanged. The
        registry calls this for every tool when a session is active
        (see ``runtime.tools.registry.get_langchain_tools``); an agent
        without a policy therefore changes nothing for other agents.
        """
        return tool

    def confirm_tool_name(self, pipeline_id: str) -> str:
        """The confirm tool's name for one pipeline — pipeline-id-based.

        The old hotel name ``confirm_guest_booking`` is gone: the name
        follows the pipeline id (``confirm_booking_create``), so the
        generic engine never carries a hotel noun.
        """
        return f"confirm_{pipeline_id}"

    confirm_tool_description = (
        "Record that the quoted write pipeline was explicitly confirmed "
        "in this conversation. Call once after confirmation, before the "
        "write step runs."
    )

    def confirm_tool(self, session_id: str, pipeline_id: str):
        """The session-scoped confirm tool for one pipeline.

        Builds the per-session tool and registers the registry stub
        under the same name, so the L2 call-time gate
        (``enforce_layer``) knows the tool — the stub is the only thing
        the registry ever holds; the real tool lives in the flow agent.
        """
        from runtime.write_pipeline_runner import (
            ensure_confirm_tool_registered,
            make_confirm_tool,
        )

        tool_name = self.confirm_tool_name(pipeline_id)
        ensure_confirm_tool_registered(tool_name, self.confirm_tool_description)
        return make_confirm_tool(
            session_id, pipeline_id, tool_name, self.confirm_tool_description
        )

    def write_pipeline(
        self,
        pipeline,
        *,
        session_id: str,
        execution_id: str,
        tools_by_name: dict,
        session_turns: int,
    ):
        """Run one pipeline's automated write (generic engine as-is)."""
        from runtime.write_pipeline_runner import try_automated_write

        return try_automated_write(
            pipeline,
            session_id=session_id,
            execution_id=execution_id,
            tools_by_name=tools_by_name,
            session_turns=session_turns,
            agent=self,
        )

    # -- Override points (Phase 4 + 5) -----------------------------------
    # Guest channel + prompt fallback. Both used to be hardcoded in
    # the shared modules (flow compiler, harness registry); the agent
    # now declares them and the modules only read the declaration.

    def guest_reply_tool(self):
        """The reply_to_guest tool for this agent's flows, or None.

        Gated by the mission harness flag
        ``guest_output.register_reply_tools`` — hotel_reception sets
        it; missions without the flag (analytis, fault, admin) never
        get the tool, so their flow agents answer in free text. The
        flag lives in the parsed harness
        (``execution.harness.guest_output``); a raw
        ``execution.guest_output`` block is accepted as a fallback
        for hand-built mission dicts.
        """
        execution = self.mission.get("execution") or {}
        guest_output = (execution.get("harness") or {}).get("guest_output") or {}
        if not guest_output:
            guest_output = execution.get("guest_output") or {}
        if not guest_output.get("register_reply_tools"):
            return None
        from runtime.guest_output import guest_reply_tool as _guest_reply_tool

        return _guest_reply_tool()

    def fallback_system_prompt(self) -> str:
        """Prompt when no AgentVersion exists — the seed YAML text.

        The old harness ``system_prompt_fallback`` Literal
        (default/admin/seed) is gone: each agent class declares its
        own fallback. The base reads the agent's seed YAML; AdminAgent
        returns its own text without touching the file. Empty when the
        file has no prompt — the caller falls back to the default.
        """
        if not self.agent_id:
            return ""
        from pathlib import Path

        import yaml

        path = (
            Path(__file__).resolve().parents[2]
            / "seeds"
            / "agents"
            / f"{self.agent_id}.yaml"
        )
        if not path.is_file():
            return ""
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 — a broken seed file is not fatal
            return ""
        prompt = data.get("system_prompt")
        return prompt if isinstance(prompt, str) and prompt.strip() else ""
