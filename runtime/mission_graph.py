"""Mission Graph — the top-level LangGraph for one workflow family (Layer 3).

Flow: START → understand → router → capability graph and/or deep agent →
validate → END.

Routing is driven by `mission.execution.mode`:
- `deterministic`: only the Capability Graph runs.
- `autonomous`: only the Deep Agent runs.
- `hybrid` (default): Capability Graph first; the Deep Agent runs only if
  the collected evidence is too thin (fewer than MIN_EVIDENCE items).

LangGraph is also allowed to call plain Python nodes and Capability Graphs
directly — a Deep Agent is not required for fixed workflows.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading

from langgraph.graph import END, START, StateGraph

from app.config import settings
from infra.mlflow_setup import setup_mlflow_tracing
from runtime.capabilities.fault_analysis import build_fault_analysis_graph
from runtime.security import detect_injection_signals
from runtime.state import MissionState

logger = logging.getLogger(__name__)

# MLflow tracing for this process (API worker, eval script, langgraph dev
# server — all import this module). Never raises; logs and continues
# without tracing when the server is down.
setup_mlflow_tracing()

# Below this much SUBSTANTIVE evidence after the Capability Graph, a
# hybrid mission escalates to the Deep Agent for further investigation.
# Substantive = real findings (fault / maintenance / document /
# relation). "error" items (a store was down) and "empty" items (a store
# was consulted and had nothing) do NOT count: previously every item
# counted, so a run with all four stores down still had 4 evidence
# items and never escalated — the Deep Agent never got the chance to
# investigate the outage itself.
MIN_EVIDENCE_FOR_HYBRID = 3

_SUBSTANTIVE_EVIDENCE_KINDS = frozenset(
    {"fault", "maintenance", "document", "relation"}
)


def _substantive_evidence_count(evidence: list | None) -> int:
    return sum(
        1 for item in (evidence or []) if (item or {}).get("kind") in _SUBSTANTIVE_EVIDENCE_KINDS
    )

CAPABILITY_BUILDERS = {
    "fault_analysis": build_fault_analysis_graph,
    # Capability names map to subgraph builders. Add new capabilities here
    # (e.g. "maintenance_analysis", "knowledge_retrieval") as they are built.
}


# --- Nodes ---
def understand_mission(state: MissionState) -> dict:
    """Validate the execution input and sketch the plan from the mission."""
    mission = state.get("mission") or {}
    input_data = state.get("input") or {}
    errors: list[str] = []

    required = (mission.get("input_schema") or {}).get("required", []) or []
    missing = [f for f in required if f not in input_data]
    if missing:
        errors.append(f"Missing required input fields: {missing}")

    mode = (mission.get("execution") or {}).get("mode", "hybrid")
    plan = [
        {"step": capability, "status": "pending"}
        for capability in (mission.get("capabilities") or [])
    ]
    return {
        "mode": mode,
        "plan": plan,
        "status": "RUNNING",
        "errors": errors,
    }


def capability_node(state: MissionState) -> dict:
    """Run the mission's Capability Graphs (deterministic subgraphs).

    Each named capability maps to a reusable LangGraph subgraph. Unknown
    capability names are recorded as errors instead of failing the run.
    """
    mission = state.get("mission") or {}
    capabilities = mission.get("capabilities") or []
    # Map business capability names to subgraph builders.
    # NOTE (F1 fix, 2026-11-11): capability names do double duty — they
    # route the capability node AND gate tools in registry.check(). The
    # fault tools get_sensor_data (sensor_data) and
    # query_asset_relationships (asset_relations) need their capability
    # declared on the mission or the agent never receives them, even
    # though the agent lists all six tools in allowed_tools. Both map
    # to the same fault_analysis subgraph (deduped by `ran` below), so
    # declaring them costs no extra subgraph run.
    capability_to_graph = {
        "fault_history": "fault_analysis",
        "maintenance_analysis": "fault_analysis",
        "knowledge_retrieval": "fault_analysis",
        "sensor_data": "fault_analysis",
        "asset_relations": "fault_analysis",
        "fault_analysis": "fault_analysis",
    }
    new_evidence: list = []
    errors: list[str] = []
    result_update: dict = {}
    ran: set[str] = set()
    for capability in capabilities:
        graph_name = capability_to_graph.get(capability)
        if graph_name is None or graph_name not in CAPABILITY_BUILDERS:
            errors.append(f"Capability not implemented: {capability}")
            continue
        if graph_name in ran:
            continue
        ran.add(graph_name)
        try:
            subgraph = CAPABILITY_BUILDERS[graph_name]()
            before = len(state.get("evidence") or [])
            out = subgraph.invoke(dict(state))
            after_evidence = out.get("evidence") or []
            new_evidence.extend(after_evidence[before:])
            if out.get("result"):
                result_update = out["result"]
        except Exception as exc:  # noqa: BLE001 — capability failure is state
            errors.append(f"Capability {graph_name} failed: {exc}")
    updates: dict = {"evidence": new_evidence, "errors": errors}
    if result_update:
        updates["result"] = result_update
    return updates


def _resolve_agent_version(agent_id: str) -> tuple[object | None, bool]:
    """Resolve the active AgentVersion + drift flag for one agent.

    Best-effort and never raising: any failure returns ``(None,
    False)`` and the caller falls back to the pre-versioning
    behaviour (env model + registry prompt, no allowed_tools gate).
    Shared by ``run_mission`` (every execution, so capability-only
    runs record their version too) and ``deep_agent_node``.
    """
    if not agent_id:
        return None, False
    agent_version = None
    try:
        from app.core.db import SessionLocal
        from app.services import agent_version_service

        _db = SessionLocal()
        try:
            agent_version = agent_version_service.get_active_version(_db, agent_id)
        finally:
            _db.close()
    except Exception as exc:  # noqa: BLE001 — versioning never breaks a run
        logger.debug("Could not resolve active agent version for %s: %s", agent_id, exc)
        return None, False
    if agent_version is None:
        return None, False
    drift = False
    try:
        from app.services import agent_version_service

        _, live_digest = agent_version_service.compute_tool_manifest()
        drift = live_digest != agent_version.manifest_sha256
        if drift:
            logger.warning(
                "Agent %s v%d: live tool manifest drifted from the pinned one",
                agent_id, agent_version.version,
            )
    except Exception as exc:  # noqa: BLE001 — drift is advisory
        logger.debug("Could not compute live tool manifest: %s", exc)
    return agent_version, drift


def _load_agent_session_limits(agent_id: str) -> dict | None:
    """Best-effort live-row session_history limits for one agent.

    Only used when the resolved AgentVersion carries no limits (older
    versions predate the field). Never raises: any failure means the
    caller falls back to the global caps.
    """
    if not agent_id:
        return None
    try:
        from app import models
        from app.core.db import SessionLocal

        _db = SessionLocal()
        try:
            agent = _db.get(models.Agent, agent_id)
            if agent is None or not agent.session_history:
                return None
            return dict(agent.session_history)
        finally:
            _db.close()
    except Exception as exc:  # noqa: BLE001 — limits never break a run
        logger.debug("Could not load agent %s session limits: %s", agent_id, exc)
        return None


def deep_agent_node(state: MissionState) -> dict:
    """Run the autonomous Deep Agent (Layer 4) with permission-filtered tools.

    If no LLM is configured (or the deep-agents package/model fails), the
    error is recorded in state and validation decides the outcome — the
    graph never crashes on a missing model.
    """
    mission = state.get("mission") or {}
    input_data = state.get("input") or {}
    # "input" + "execution_config" feed Layer 4 execution scoping in the
    # registry: tools taking asset_id are wrapped to refuse any asset
    # other than this execution's, unless the mission opts out with
    # execution_config.allow_cross_asset.
    context = {
        "mission_id": state.get("mission_id"),
        "execution_id": state.get("execution_id"),
        "tenant": "default",
        "capabilities": mission.get("capabilities") or [],
        "input": input_data,
        "execution_config": mission.get("execution") or {},
    }
    try:
        from runtime.deep_agent import (
            build_deep_agent,
            builtin_tools_exclusion,
            run_deep_agent,
        )
        from runtime.hooks import (
            HookContext,
            HookEvent,
            HookMiddleware,
            default_registry,
            select_hooks,
        )
        from runtime.middleware import TokenAuditCallbackHandler
        from runtime.tools.mcp_adapter import load_mcp_tools
        from runtime.tools.registry import registry

        # Hooks: one registry per run (built-ins, narrowed to the
        # mission's declared hooks when it names any — otherwise all
        # built-ins run). UserPromptSubmit fires before anything else:
        # a blocked input never reaches the agent, the tools, or the
        # version lookup. This is the net for executions that bypass
        # the API router's Layer 1 validation (schedules, admin
        # dispatch) — see runtime/hooks/builtin.py::validate_input.
        hook_registry = default_registry().with_only(select_hooks(mission))
        hook_base = HookContext(
            execution_id=state.get("execution_id") or "",
            mission_id=state.get("mission_id") or "",
            tenant=context.get("tenant") or "default",
            agent_id=mission.get("agent_id") or "",
            mission=mission,
            input=input_data,
        )
        blocked = [
            d
            for d in hook_registry.dispatch(HookEvent.USER_PROMPT_SUBMIT, hook_base)
            if d.action == "block"
        ]
        if blocked:
            return {"errors": [f"Input blocked by hook: {blocked[0].message}"]}

        # Option C: the active AgentVersion for this mission's agent.
        # run_mission already resolved it for every execution (see
        # run_mission) and put its number in the state — reuse that
        # version by number instead of re-resolving, and keep the
        # drift flag it computed. When the node is called directly
        # without a state version (tests, langgraph dev), resolve the
        # active version here instead. Any failure falls back to the
        # pre-versioning behaviour: env model + registry prompt, and no
        # allowed_tools gate. When a version is active, its allowed
        # tools gate the registry (registry.check enforces
        # context["allowed_tools"]) and its pinned prompt/model/
        # temperature run the agent; the tools themselves still come
        # from the live code registry.
        agent_version = None
        agent_version_drift = bool(state.get("agent_version_drift"))
        agent_id = mission.get("agent_id") or ""
        state_version = state.get("agent_version")
        if agent_id and state_version is not None:
            try:
                from app.core.db import SessionLocal
                from app.services import agent_version_service

                _db = SessionLocal()
                try:
                    agent_version = agent_version_service.get_version(
                        _db, agent_id, state_version
                    )
                finally:
                    _db.close()
            except Exception as exc:  # noqa: BLE001 — versioning never breaks a run
                logger.debug(
                    "Could not load agent version %s v%s: %s",
                    agent_id, state_version, exc,
                )
                agent_version = None
        if agent_version is None:
            agent_version, agent_version_drift = _resolve_agent_version(agent_id)
        if agent_version is not None:
            context["allowed_tools"] = list(agent_version.allowed_tools or [])

        # Layer 4: MCP tools go through the registry permission check
        # like built-in tools. load_mcp_tools() registers only the MCP
        # tools mapped to a capability (default deny for the rest);
        # get_langchain_tools() then filters everything — built-in and
        # MCP — by the mission's capabilities. Previously MCP tools were
        # appended unfiltered (audit finding, now closed).
        load_mcp_tools()
        tools = registry.get_langchain_tools(context)
        # Middleware: one handler instance per run, carrying this run's
        # execution context — it logs token usage + tool calls to the DB
        # (best-effort; a logging failure never fails the run). It is
        # passed through the invoke config so it sees the agent's LLM and
        # tool calls; the eval script's mock agent ignores the config, so
        # mock evals keep working with no rows written.
        handler = TokenAuditCallbackHandler(
            execution_id=state.get("execution_id") or "",
            mission_id=state.get("mission_id") or "",
            tenant=context.get("tenant") or "default",
            model=agent_version.model if agent_version is not None else settings.llm_model,
        )
        # Skills: deterministic selection (explicit mission declaration
        # wins, else keyword matching over mission + request — see
        # runtime/skills/). Only the selected skill files are handed to
        # the agent via invoke(files=...); the native deepagents
        # SkillsMiddleware discloses them, and the default in-memory
        # backend stays in use. No matching skill means no skills
        # argument at all — the run behaves exactly as before.
        from runtime.skills import SKILLS_SOURCE, select_skills, skill_files_for_run
        from runtime.skills.map_plan import plan_trace_note, render_plan

        selected_skills = select_skills(mission, input_data)
        skill_sources = [SKILLS_SOURCE] if selected_skills else None
        skill_files = skill_files_for_run(selected_skills) if selected_skills else None
        # Skill maps (Option C): skills that declare a ``map:`` become a
        # plan in the agent's instructions — rendered once here and
        # appended to whichever system prompt runs below (pinned version
        # prompt, admin prompt, or default). The plan is skill-derived
        # runtime context, like the skill files themselves, not part of
        # the pinned prompt text. Skills without a map are unaffected.
        mapped_skills = [
            s for s in (selected_skills or []) if getattr(s, "map", None) is not None
        ]
        # Phase 3 — the skill subgraph is the new architecture's core:
        # for pilot missions the map is COMPILED into a LangGraph (one
        # agent per map node, code-owned routing) instead of running one
        # free-form agent with the map as prose. The plan gate (Phase
        # 3a) is the enforcement mechanism for missions still on the
        # free-form path — a run uses ONE of the two, never both, so the
        # gate is built only when the subgraph does not run.
        from runtime.skill_subgraph import use_skill_subgraph

        use_subgraph = use_skill_subgraph(state.get("mission_id"), mapped_skills)
        plan_gate_mw = None
        if not use_subgraph:
            from runtime.plan_gate import plan_gate

            plan_gate_mw = plan_gate(
                [s.map for s in mapped_skills],
                state.get("mission_id"),
                execution_id=state.get("execution_id") or "",
            )
        plan_block = (
            "\n\n".join(
                render_plan(s.map, enforced=plan_gate_mw is not None)
                for s in mapped_skills
            )
            if mapped_skills
            else ""
        )
        # Option B session history (docs/sessions.md): prior turns of
        # this execution's session, rendered into the system prompt in
        # ALL branches below (pinned version prompt, admin prompt,
        # default) — the agent reads back what earlier runs did. The
        # block is measured against its own budget at render time
        # (charged to the history slice of the context budget, not the
        # system-prompt bucket), so the prefix check below measures the
        # prompt WITHOUT it (see prompt_for_budget).
        #
        # Per-agent limits: the pinned AgentVersion carries the agent's
        # session_history when it was snapshotted with one; older
        # versions predate the field, so fall back to the live agent
        # row, then to the global caps. The load-time cap in
        # session_service is the global safety net; re-capping here
        # enforces the agent's own (possibly tighter) limit, and the
        # omitted counts add up.
        from runtime.session_history import (
            cap_turns,
            render_session_history,
            resolve_session_limits,
        )

        _history = state.get("session_history") or {}
        _version_limits = (
            getattr(agent_version, "session_history", None)
            if agent_version is not None
            else None
        )
        _limits = resolve_session_limits(
            version_limits=_version_limits,
            agent_limits=(
                _load_agent_session_limits(agent_id) if not _version_limits else None
            ),
        )
        _shown_turns, _render_omitted = cap_turns(
            _history.get("turns") or [], limits=_limits
        )
        _shown_omitted = (_history.get("omitted") or 0) + _render_omitted
        history_block = render_session_history(
            _shown_turns,
            omitted=_shown_omitted,
            session_id=state.get("session_id"),
            limits=_limits,
            agent_id=agent_id or None,
        )
        # Hooks (in-agent half): one middleware instance per run, over
        # the same registry the node already dispatched UserPromptSubmit
        # on. It denies dangerous tool calls, injects feedback on tool
        # results, and forces extra turns when the agent wants to stop
        # with no evidence gathered (see runtime/hooks/middleware.py).
        # The stop cap comes from execution.stop_max_continues.
        hook_middleware = HookMiddleware(
            hook_registry,
            execution_id=state.get("execution_id") or "",
            mission_id=state.get("mission_id") or "",
            tenant=context.get("tenant") or "default",
            agent_id=mission.get("agent_id") or "",
            mission=mission,
            input_data=input_data,
            evidence_count=len(state.get("evidence") or []),
            max_continues=int(
                (mission.get("execution") or {}).get("stop_max_continues", 3)
            ),
        )
        # Part 1 (b): the per-run middleware list. For missions that
        # must not see any built-in tool (hotel only), append the
        # exclusion middleware — it hides the 9 deepagents built-ins
        # from the model request and rejects them at the call boundary.
        # Other missions keep the default list (fault_analysis needs
        # `execute` in its sandbox). See
        # runtime/deep_agent.py::builtin_tools_exclusion.
        middleware = [hook_middleware]
        exclusion = builtin_tools_exclusion(state.get("mission_id"))
        if exclusion is not None:
            middleware.append(exclusion)
        # Phase 3a: the plan gate enforces the skill map's tool order
        # for the pilot mission (hotel only). It is appended after the
        # exclusion so the exclusion hides the built-ins first; the gate
        # then sees only the tools the model may actually call.
        if plan_gate_mw is not None:
            middleware.append(plan_gate_mw)
        # Sandbox backend (docs/sandbox.md): when the mission opts in
        # (execution.sandbox: true) and this run has a session, the
        # agent's file tools + `execute` run inside that session's
        # OpenSandbox sandbox instead of the in-memory StateBackend.
        # Resolution is best-effort and never breaks a run: a missing
        # server, a dead sandbox, or a disabled switch falls back to
        # the default backend, with a note on the execution. The
        # backend handle lives only for this run — the sandbox itself
        # persists server-side and the next run re-attaches by id.
        sandbox_backend = None
        sandbox_note = None
        if (mission.get("execution") or {}).get("sandbox") is True and state.get("session_id"):
            try:
                from app.core.db import SessionLocal
                from app.services import sandbox_service

                _sdb = SessionLocal()
                try:
                    sandbox_backend = sandbox_service.resolve_session_sandbox(
                        _sdb,
                        session_id=state["session_id"],
                        agent_id=agent_id,
                        mission_id=state.get("mission_id") or "",
                    )
                finally:
                    _sdb.close()
            except Exception as exc:  # noqa: BLE001 — sandbox never breaks a run
                logger.warning("Sandbox backend unavailable: %s", exc)
                sandbox_backend = None
            if sandbox_backend is None:
                sandbox_note = "Sandbox unavailable — in-memory file backend"
            else:
                sandbox_note = f"Sandbox backend: {sandbox_backend.id}"
        # The fixed prefix measured against the system-prompt budget —
        # set in each branch below, BEFORE the history block is
        # appended (history is charged to the history bucket, not here).
        # The base prompt + model config are selected once, shared by
        # the free-form agent and the subgraph pilot (which takes the
        # base prompt WITHOUT the plan block — the map is compiled, not
        # prose — and reads the session history from the graph state).
        prompt_for_budget = ""
        version_cfg = None
        if agent_version is not None:
            # Option C: the pinned prompt + model/temperature run the agent.
            # The config (not a built model) is passed down so the model
            # is constructed inside build_deep_agent, where test doubles
            # replace it — the node itself never needs live LLM config
            # just to resolve a version.
            version_cfg = settings.model_copy(update={
                "llm_model": agent_version.model,
                "llm_temperature": agent_version.temperature,
            })
            base_prompt_text = agent_version.prompt_text
        elif (mission.get("agent_id") or "") == "admin_bot":
            from runtime.deep_agent import ADMIN_SYSTEM_PROMPT

            # No-version fallback: the code constant (a copy of the YAML
            # seed text) runs — the runtime never looks up MLflow for
            # prompts (docs/agent-versioning.md).
            base_prompt_text = ADMIN_SYSTEM_PROMPT
        else:
            from runtime.deep_agent import DEFAULT_SYSTEM_PROMPT

            # No-version fallback: the code constant (a copy of the YAML
            # seed text) runs — the runtime never looks up MLflow for
            # prompts (docs/agent-versioning.md).
            base_prompt_text = DEFAULT_SYSTEM_PROMPT
        # Tracing (docs/sessions.md): the execution/session identity rides
        # in the LangSmith config of the agent invoke (free-form) or the
        # subgraph invoke (pilot) — built once, used by either path.
        from runtime.tracing import build_run_metadata, build_run_tags

        trace_metadata = build_run_metadata(
            state.get("execution_id") or "",
            state.get("mission_id") or "",
            mission.get("agent_id") or None,
            state.get("session_id"),
        )
        trace_tags = build_run_tags(
            state.get("mission_id") or "",
            mission.get("agent_id") or None,
            state.get("session_id"),
        )
        if use_subgraph:
            # Phase 3: compile the first mapped skill's map into a
            # subgraph and run it instead of one free-form agent. The
            # node agents carry the hook middleware + built-in exclusion
            # (the plan gate is not attached — the subgraph IS the
            # enforcement). The context-budget prefix check is skipped
            # here: the per-node prompts are smaller than the free-form
            # prefix it measures, and the check is advisory.
            from runtime.skill_subgraph import build_skill_subgraph, run_skill_subgraph

            prompt_for_budget = base_prompt_text
            prefix_overruns = []
            subgraph = build_skill_subgraph(
                mapped_skills[0].map,
                tools,
                base_prompt_text,
                model_config=version_cfg,
                middleware=middleware,
                backend=sandbox_backend,
                skills=skill_sources,
                files=skill_files,
                callbacks=[handler],
                trace_metadata=trace_metadata,
                trace_tags=trace_tags,
                # Confirmation-turn rule: from the start node, the
                # guest's confirmation may resume straight at
                # create-booking when the session has prior turns
                # (see runtime/skill_subgraph.py).
                resume_nodes=frozenset({"create-booking"}),
            )
            outcome = run_skill_subgraph(
                subgraph, mission, input_data,
                session_history=_history, history_block=history_block,
                skill_maps=[s.map for s in mapped_skills],
                trace_metadata=trace_metadata,
                trace_tags=trace_tags,
            )
        else:
            if agent_version is not None:
                system_prompt_text = base_prompt_text
                if plan_block:
                    system_prompt_text = system_prompt_text + "\n\n" + plan_block
                prompt_for_budget = system_prompt_text
                if history_block:
                    system_prompt_text = system_prompt_text + "\n\n" + history_block
                agent = build_deep_agent(
                    tools,
                    system_prompt=system_prompt_text,
                    skills=skill_sources,
                    model_config=version_cfg,
                    middleware=middleware,
                    backend=sandbox_backend,
                )
            elif (mission.get("agent_id") or "") == "admin_bot":
                system_prompt_text = base_prompt_text
                if plan_block:
                    system_prompt_text = system_prompt_text + "\n\n" + plan_block
                prompt_for_budget = system_prompt_text
                if history_block:
                    system_prompt_text = system_prompt_text + "\n\n" + history_block
                agent = build_deep_agent(
                    tools,
                    system_prompt=system_prompt_text,
                    skills=skill_sources,
                    middleware=middleware,
                    backend=sandbox_backend,
                )
            else:
                system_prompt_text = base_prompt_text
                if plan_block:
                    system_prompt_text = system_prompt_text + "\n\n" + plan_block
                prompt_for_budget = system_prompt_text
                if history_block:
                    system_prompt_text = system_prompt_text + "\n\n" + history_block
                agent = build_deep_agent(
                    tools,
                    system_prompt=system_prompt_text,
                    skills=skill_sources,
                    middleware=middleware,
                    backend=sandbox_backend,
                )
            # Context budget (Phương án A): measure the fixed prefix —
            # system prompt, tool definitions, selected skill texts —
            # against its buckets. Advisory only: an overrun is logged and
            # noted on the execution; the run still proceeds (these are
            # pinned config, not run data). The session-history block is
            # measured against its own budget at render time and is NOT
            # part of this check (prompt_for_budget excludes it).
            from runtime.context_budget import check_prefix_budgets

            tools_text = "".join(
                getattr(t, "name", "")
                + (getattr(t, "description", "") or "")
                + json.dumps(getattr(t, "args", {}) or {}, default=str)
                for t in tools
            )
            skills_text = "".join(s.raw for s in (selected_skills or []))
            prefix_overruns = check_prefix_budgets(
                prompt_for_budget, tools_text, skills_text
            )
            for overrun in prefix_overruns:
                logger.warning("Context budget overrun: %s", overrun)
            run_kwargs: dict = {"callbacks": [handler], "files": skill_files}
            if mapped_skills:
                run_kwargs["skill_maps"] = [s.map for s in mapped_skills]
            run_kwargs["trace_metadata"] = trace_metadata
            run_kwargs["trace_tags"] = trace_tags
            # Doubles predating skill_maps / trace kwargs (older tests, eval
            # mocks) don't accept them — retry without each unknown kwarg;
            # the run proceeds, just without a plan trace / trace metadata.
            # Any other TypeError is a real bug and is re-raised. Each retry
            # strips at least one kwarg, so the loop always terminates.
            while True:
                try:
                    outcome = run_deep_agent(agent, mission, input_data, **run_kwargs)
                    break
                except TypeError as exc:
                    msg = str(exc)
                    if "unexpected keyword argument 'skill_maps'" in msg:
                        run_kwargs.pop("skill_maps", None)
                    elif "unexpected keyword argument 'trace_metadata'" in msg:
                        run_kwargs.pop("trace_metadata", None)
                        run_kwargs.pop("trace_tags", None)
                    elif "unexpected keyword argument 'trace_tags'" in msg:
                        run_kwargs.pop("trace_tags", None)
                    else:
                        raise
        notes = [f"Deep agent: {outcome.get('summary', '')}"] if outcome.get("summary") else []
        # Phase 3a: surface plan-gate denials in the execution notes, so
        # a run that had to be corrected is explainable from the result
        # alone (the denial messages themselves are in the trace).
        if plan_gate_mw is not None and plan_gate_mw.denied_count:
            notes.append(
                f"Plan gate: {plan_gate_mw.denied_count} out-of-plan tool "
                "call(s) denied"
            )
        # Sandbox backend (docs/sandbox.md): one line saying which
        # backend ran — the sandbox id, or that the run fell back to
        # the in-memory backend because no sandbox was available.
        if sandbox_note:
            notes.append(sandbox_note)
        # Option B session history: one line saying how many prior
        # turns the agent was shown (the turns themselves are in the
        # system prompt, not in the result). Counts are the turns
        # actually shown after the per-agent cap, not the loaded ones.
        if _shown_turns:
            _history_note = f"Session history: {len(_shown_turns)} prior turn(s)"
            if _shown_omitted:
                _history_note += f" ({_shown_omitted} earlier omitted)"
            notes.append(_history_note)
        # Plan trace (Option C): deterministic audit of the run against
        # each map — visited nodes, skipped nodes, tools outside the map.
        plan_trace = outcome.get("plan_trace") or {}
        for skill_name, trace in plan_trace.items():
            notes.append(plan_trace_note(skill_name, trace))
        # Surface hook interventions in the state: one compact line in
        # notes plus the full per-dispatch record in hook_events, so a
        # run that was denied/feedbacked/continued is explainable from
        # the execution result alone (no DB lookup needed).
        hook_events = hook_middleware.events
        denied = sum(
            1 for e in hook_events for a in e["decisions"] if a == "deny"
        )
        feedbacks = sum(
            1 for e in hook_events for a in e["decisions"] if a == "feedback"
        )
        continued = hook_middleware.continues
        if denied or feedbacks or continued:
            notes.append(
                "Hooks intervened: "
                + ", ".join(
                    part
                    for part in [
                        f"{denied} tool call(s) denied" if denied else "",
                        f"{feedbacks} feedback(s) injected" if feedbacks else "",
                        f"{continued} extra turn(s) forced" if continued else "",
                    ]
                    if part
                )
            )
        # Context budget notes: fixed-prefix overruns (advisory) and any
        # history trimming the middleware performed before model calls.
        if prefix_overruns:
            notes.append("Context budget: " + "; ".join(prefix_overruns))
        if hook_middleware.trimmed_messages:
            notes.append(
                f"Context budget: trimmed {hook_middleware.trimmed_messages} "
                f"message(s) (~{hook_middleware.trimmed_tokens} tokens) "
                "to fit the history budget"
            )
        # Fill the mission's declared output-schema keys from the agent
        # outcome so an autonomous run can validate (Option A, tran
        # 2026-11-11). The result MERGES with the capability subgraph's
        # result instead of overwriting it: on hybrid escalation the
        # capability path's root_causes/recommendations survive when
        # the agent provides none. Keys the outcome cannot fill
        # directly are mapped from the evidence, never invented:
        # root_causes come from substantive evidence items (same
        # fallback as the capability path), recommendations quote
        # actionable sentences from the agent's own summary, or a
        # single honest generic item when there are none.
        result: dict = dict(state.get("result") or {})
        required_keys = list((mission.get("output_schema") or {}).get("properties", {}).keys())
        summary = outcome.get("summary", "")
        if "answer" in required_keys:
            result["answer"] = summary
        if "summary" in required_keys:
            result["summary"] = summary
        if "actions_taken" in required_keys:
            result["actions_taken"] = outcome.get("actions_taken") or []
        if "evidence" in required_keys:
            result["evidence"] = list(state.get("evidence") or []) or (
                outcome.get("evidence") or []
            )
        all_evidence = list(state.get("evidence") or []) + list(
            outcome.get("evidence") or []
        )
        if "root_causes" in required_keys and not result.get("root_causes"):
            result["root_causes"] = _root_causes_from_evidence(all_evidence)
        if "recommendations" in required_keys and not result.get("recommendations"):
            result["recommendations"] = _recommendations_from_summary(summary)
        tool_failures = outcome.get("errors") or []
        if tool_failures:
            # Business-level tool refusals (a delete refused, a cancel
            # on a finished execution) are outcomes the agent handled
            # by design — they stay visible in the result even when
            # the run SUCCEEDs, instead of failing validation via
            # state errors. The structured list is authoritative over
            # the model's prose summary, which is not rewritten.
            result["tool_failures"] = list(tool_failures)
            notes.append("Tool failures reported: " + "; ".join(tool_failures))
        if plan_trace:
            # Additive only: validate_results checks the declared
            # output-schema keys and ignores extra keys.
            result["plan_trace"] = plan_trace
        return {
            "evidence": outcome.get("evidence") or [],
            "result": result,
            "notes": notes,
            "errors": [],
            "agent_version": agent_version.version if agent_version is not None else None,
            "agent_version_drift": agent_version_drift,
            "hook_events": hook_events,
        }
    except Exception as exc:  # noqa: BLE001 — model/package problems are state
        logger.warning("Deep agent unavailable: %s", exc)
        return {"errors": [f"Deep agent unavailable: {exc}"]}


def _root_causes_from_evidence(evidence: list | None) -> list[str]:
    """One root-cause line per substantive evidence item.

    The line is the item's own summary, which already cites its
    source (fault code/title, document title, relation). Only real
    findings count — "error"/"empty"/"summary" items are not causes.
    With no substantive evidence the honest fallback is the same one
    the capability path uses. Nothing is invented here.
    """
    causes = [
        str((item or {}).get("summary") or "").strip()
        for item in (evidence or [])
        if (item or {}).get("kind") in _SUBSTANTIVE_EVIDENCE_KINDS
        and str((item or {}).get("summary") or "").strip()
    ]
    return causes[:5] if causes else ["unknown — insufficient evidence"]


# Words that mark a summary sentence as an action the agent itself
# proposed (used only to quote the agent's own words — never to
# invent a specific action).
_ACTIONABLE_WORDS = (
    "recommend", "should", "must", "need to", "replace", "repair",
    "schedule", "inspect", "monitor", "investigate", "fix",
)


def _recommendations_from_summary(summary: str) -> list[str]:
    """Actionable sentences from the agent's own summary (max 3).

    A sentence only qualifies when it reads as an action (see
    ``_ACTIONABLE_WORDS``) — the returned text is quoted verbatim
    from the summary. With no actionable sentence, a single honest
    generic item is returned instead of an invented specific action.
    """
    actions: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", summary or ""):
        text = sentence.strip()
        if not text:
            continue
        if any(word in text.lower() for word in _ACTIONABLE_WORDS):
            actions.append(text)
        if len(actions) >= 3:
            break
    return actions or ["Investigate the listed causes with the available evidence"]


def _grounding_tokens(text: str) -> set[str]:
    """Distinctive words of a root-cause claim (len >= 4)."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4}


def _is_grounded(cause: str, evidence: list) -> bool:
    """True when a root cause is supported by at least one evidence item:
    it shares a distinctive token with an evidence summary, or an
    evidence item's id appears in it. The honest "unknown — insufficient
    evidence" fallback is not a claim, so it counts as grounded. A cause
    with no checkable tokens (e.g. a bare code) is not punished."""
    lowered = cause.strip().lower()
    if lowered.startswith("unknown"):
        return True
    tokens = _grounding_tokens(cause)
    for item in evidence:
        summary = str((item or {}).get("summary") or "").lower()
        if tokens and any(token in summary for token in tokens):
            return True
        item_id = str(((item or {}).get("data") or {}).get("id") or "")
        if item_id and item_id in cause:
            return True
    return not tokens


def _allowed_tool_names(mission: dict) -> set[str] | None:
    """Tool names this mission's capabilities allow (None = cannot tell,
    e.g. the tool modules failed to import — the trace check is then
    skipped instead of flagging everything)."""
    try:
        import runtime.tools  # noqa: F401 — registers built-in tools
        from runtime.tools.registry import registry
    except Exception:  # noqa: BLE001 — validation must never crash
        return None
    allowed = {
        e.name
        for e in registry.list_for_context(
            {"capabilities": mission.get("capabilities") or []}
        )
    }
    return allowed or None


def validate_results(state: MissionState) -> dict:
    """Check evidence, output schema and success criteria.

    Layer 5 (prompt-injection defense) adds three checks that FLAG but
    never fail a run — flags live in result["validation_flags"] and mark
    the execution for review:
    - grounding: each root_causes item must be supported by some
      collected evidence; ungrounded causes move to
      result["ungrounded_root_causes"] and never reach the official
      report.
    - tool trace: actions_taken must be a subset of the tools this
      mission's capabilities allow.
    - detection: evidence/root-cause text matching instruction-like
      patterns (Layer 6) is flagged with the matched signal.
    """
    errors = list(state.get("errors") or [])
    evidence = state.get("evidence") or []
    result = dict(state.get("result") or {})
    mission = state.get("mission") or {}
    required_keys = list((mission.get("output_schema") or {}).get("properties", {}).keys())

    if not evidence:
        errors.append("No evidence collected")
    if "evidence" not in result:
        result["evidence"] = evidence
    missing_keys = [k for k in required_keys if k not in result]
    if missing_keys:
        errors.append(f"Result missing output-schema keys: {missing_keys}")

    flags: list[dict] = list(result.get("validation_flags") or [])
    causes = result.get("root_causes")
    if isinstance(causes, list) and causes:
        grounded = [c for c in causes if _is_grounded(str(c), evidence)]
        ungrounded = [c for c in causes if not _is_grounded(str(c), evidence)]
        if ungrounded:
            result["root_causes"] = grounded
            result["ungrounded_root_causes"] = ungrounded
            for cause in ungrounded:
                flags.append(
                    {"type": "ungrounded_root_cause", "cause": str(cause)[:200]}
                )
    actions = result.get("actions_taken")
    if isinstance(actions, list) and actions:
        allowed = _allowed_tool_names(mission)
        if allowed is not None:
            for name in actions:
                if name not in allowed:
                    flags.append(
                        {"type": "tool_outside_allowed_list", "tool": str(name)}
                    )
    for item in evidence:
        text = str((item or {}).get("summary") or "") + "\n" + json.dumps(
            (item or {}).get("data") or {}, ensure_ascii=False, default=str
        )[:8000]
        for signal in detect_injection_signals(text):
            flag = {
                "type": "injection_signal",
                "signal": signal,
                "source": str((item or {}).get("source") or ""),
            }
            if flag not in flags:
                flags.append(flag)
    for cause in (result.get("root_causes") or []) + (
        result.get("ungrounded_root_causes") or []
    ):
        for signal in detect_injection_signals(str(cause)):
            flag = {
                "type": "injection_signal",
                "signal": signal,
                "source": "root_causes",
            }
            if flag not in flags:
                flags.append(flag)
    if flags:
        result["validation_flags"] = flags
    if errors:
        return {"result": result, "status": "FAILED", "errors": errors}
    return {"result": result, "status": "SUCCEEDED"}


# --- Routing ---
def route_after_understand(state: MissionState) -> str:
    mode = state.get("mode", "hybrid")
    if mode == "autonomous":
        return "deep_agent"
    return "capability"  # deterministic and hybrid both start with capability


def route_after_capability(state: MissionState) -> str:
    mode = state.get("mode", "hybrid")
    substantive = _substantive_evidence_count(state.get("evidence"))
    if mode == "hybrid" and substantive < MIN_EVIDENCE_FOR_HYBRID:
        return "deep_agent"
    return "validate"


def build_mission_graph(checkpointer=None):
    """Compile the Mission Graph. Pass a checkpointer for persistent state
    (see get_checkpointer); without one the graph is stateless per run."""
    graph = StateGraph(MissionState)
    graph.add_node("understand", understand_mission)
    graph.add_node("capability", capability_node)
    graph.add_node("deep_agent", deep_agent_node)
    graph.add_node("validate", validate_results)
    graph.add_edge(START, "understand")
    graph.add_conditional_edges(
        "understand",
        route_after_understand,
        {"capability": "capability", "deep_agent": "deep_agent"},
    )
    graph.add_conditional_edges(
        "capability", route_after_capability, {"deep_agent": "deep_agent", "validate": "validate"}
    )
    graph.add_edge("deep_agent", "validate")
    graph.add_edge("validate", END)
    return graph.compile(checkpointer=checkpointer)


# Module-level compiled graph — the entry point for `langgraph dev`
# (see langgraph.json) and the default graph the worker runs. Building it
# is pure graph construction: no LLM call, no DB access, no checkpoint.
mission_graph = build_mission_graph()


def _to_libpq_dsn(url: str) -> str:
    """Convert a SQLAlchemy postgres URL to a libpq DSN.

    ``PostgresSaver.from_conn_string`` expects a libpq connection
    string (``postgresql://...``), but ``settings.database_url`` is a
    SQLAlchemy URL (``postgresql+psycopg://...``) — passing it through
    verbatim fails at parse time with ``missing "=" after
    "postgresql+psycopg://..."`` and the graph silently runs
    stateless. Strip the driver (``postgresql+psycopg`` ->
    ``postgresql``) via ``sqlalchemy.make_url``; already-plain
    ``postgresql://`` URLs and non-postgres URLs pass through
    untouched. Best-effort: any parse/import problem returns the
    original URL and the caller's warning path reports the failure.
    """
    try:
        from sqlalchemy.engine import make_url

        parsed = make_url(url)
        if parsed.drivername.startswith("postgresql+"):
            return parsed.set(drivername="postgresql").render_as_string(
                hide_password=False
            )
    except Exception:  # noqa: BLE001 — fall back to the original URL
        logger.debug("Could not convert database URL to a libpq DSN: %r", url)
    return url


_CHECKPOINTER_SETUP_LOCK = threading.Lock()
_checkpointer_setup_done = False


@contextlib.contextmanager
def get_checkpointer():
    """Postgres checkpointer (optional, off by default), as a context manager.

    Needs CHECKPOINT_ENABLED=true and the `postgres-checkpoint` extra.
    Yields None when unavailable — the graph then runs without persistent
    checkpoints.

    In langgraph-checkpoint-postgres >= 3, `PostgresSaver.from_conn_string`
    is itself a context manager: the saver is only usable while that
    context is open, so callers must hold this context for the whole graph
    run (see run_mission). Entering it opens the connection and `setup()`
    creates the checkpoint tables.
    """
    global _checkpointer_setup_done
    if not settings.checkpoint_enabled:
        yield None
        return
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError:
        try:
            from langgraph_checkpoint_postgres import PostgresSaver
        except ImportError:
            logger.warning("Postgres checkpointer package not installed; running without it.")
            yield None
            return
    try:
        with PostgresSaver.from_conn_string(
            _to_libpq_dsn(settings.database_url)
        ) as checkpointer:
            # setup() runs CREATE INDEX CONCURRENTLY, which waits for
            # every older transaction and cannot run concurrently with
            # itself. Set the schema up once per process (under a lock)
            # instead of on every run: concurrent executions would
            # otherwise race and block each other on the index lock.
            if not _checkpointer_setup_done:
                try:
                    with _CHECKPOINTER_SETUP_LOCK:
                        if not _checkpointer_setup_done:
                            checkpointer.setup()
                            _checkpointer_setup_done = True
                except Exception as exc:  # noqa: BLE001 — checkpoint is best-effort
                    logger.warning("Could not set up Postgres checkpointer: %s", exc)
                    yield None
                    return
            yield checkpointer
    except Exception as exc:  # noqa: BLE001 — checkpoint is best-effort
        logger.warning("Could not set up Postgres checkpointer: %s", exc)
        yield None


def run_mission(
    mission: dict,
    input_data: dict,
    execution_id: str,
    session_history: dict | None = None,
    session_id: str | None = None,
) -> dict:
    """Run one execution through the Mission Graph and return final state.

    Reuses the module-level compiled `mission_graph` unless a Postgres
    checkpointer is enabled (then a graph is built with it).

    ``session_history`` is the Option B history dict from
    ``session_service.load_session_history`` ({"turns", "omitted",
    "total_prior"}) — None means no history, and the run behaves
    exactly as before (fresh session, or callers that pass nothing).
    """

    # Tracing for direct callers (worker, eval, langgraph dev): the API
    # lifespan already calls this for its own process. No-op without a key.
    from infra.langsmith_setup import setup_langsmith

    setup_langsmith()
    setup_mlflow_tracing()
    # Option C: resolve the active AgentVersion ONCE, up front, so
    # every execution records it — including hybrid runs that never
    # escalate to the Deep Agent (previously the version was resolved
    # only inside deep_agent_node, so capability-only runs kept
    # agent_version NULL). Best-effort: a DB failure leaves it NULL
    # and the run continues as before.
    agent_version, agent_version_drift = _resolve_agent_version(
        mission.get("agent_id") or ""
    )
    initial_state: MissionState = {
        "execution_id": execution_id,
        "mission_id": mission.get("mission_id", ""),
        "mission": mission,
        "input": input_data,
        "mode": (mission.get("execution") or {}).get("mode", "hybrid"),
        "plan": [],
        "evidence": [],
        "result": None,
        "status": "RUNNING",
        "errors": [],
        "notes": [],
        "hook_events": [],
        "agent_version": agent_version.version if agent_version is not None else None,
        "agent_version_drift": agent_version_drift,
        "session_id": session_id,
        "session_history": session_history,
    }
    # Tracing (docs/sessions.md): the execution/session identity rides
    # in the LangSmith config (metadata + tags, propagated to child
    # runs) and in one tagged MLflow run per execution (autolog traces
    # link to it via mlflow.sourceRun).
    from runtime.tracing import (
        build_run_metadata,
        build_run_tags,
        mlflow_execution_run,
    )

    agent_id = mission.get("agent_id") or ""
    trace_metadata = build_run_metadata(
        execution_id, mission.get("mission_id", ""), agent_id or None, session_id
    )
    trace_tags = build_run_tags(
        mission.get("mission_id", ""), agent_id or None, session_id
    )
    # The checkpointer context must stay open for the whole invoke: the
    # saver borrows the connection from PostgresSaver.from_conn_string.
    with mlflow_execution_run(
        execution_id, mission.get("mission_id", ""), agent_id or None, session_id
    ):
        with get_checkpointer() as checkpointer:
            graph = (
                build_mission_graph(checkpointer=checkpointer)
                if checkpointer
                else mission_graph
            )
            config: dict = {"metadata": trace_metadata, "tags": trace_tags}
            if checkpointer:
                config["configurable"] = {"thread_id": execution_id}
            final_state = graph.invoke(initial_state, config=config)
    return dict(final_state)
