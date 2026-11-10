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

# Below this much evidence after the Capability Graph, a hybrid mission
# escalates to the Deep Agent for further investigation.
MIN_EVIDENCE_FOR_HYBRID = 3

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
    capability_to_graph = {
        "fault_history": "fault_analysis",
        "maintenance_analysis": "fault_analysis",
        "knowledge_retrieval": "fault_analysis",
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
        from runtime.deep_agent import build_deep_agent, run_deep_agent
        from runtime.middleware import TokenAuditCallbackHandler
        from runtime.tools.mcp_adapter import load_mcp_tools
        from runtime.tools.registry import registry

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
            model=settings.llm_model,
        )
        # Skills: deterministic selection (explicit mission declaration
        # wins, else keyword matching over mission + request — see
        # runtime/skills/). Only the selected skill files are handed to
        # the agent via invoke(files=...); the native deepagents
        # SkillsMiddleware discloses them, and the default in-memory
        # backend stays in use. No matching skill means no skills
        # argument at all — the run behaves exactly as before.
        from runtime.skills import SKILLS_SOURCE, select_skills, skill_files_for_run

        selected_skills = select_skills(mission, input_data)
        skill_sources = [SKILLS_SOURCE] if selected_skills else None
        skill_files = skill_files_for_run(selected_skills) if selected_skills else None
        if (mission.get("agent_id") or "") == "admin_bot":
            from runtime.deep_agent import ADMIN_SYSTEM_PROMPT
            from runtime.prompts import get_system_prompt

            agent = build_deep_agent(
                tools,
                system_prompt=get_system_prompt(ADMIN_SYSTEM_PROMPT, prompt_name="admin_bot"),
                skills=skill_sources,
            )
        else:
            agent = build_deep_agent(tools, skills=skill_sources)
        outcome = run_deep_agent(agent, mission, input_data, callbacks=[handler], files=skill_files)
        notes = [f"Deep agent: {outcome.get('summary', '')}"] if outcome.get("summary") else []
        # Fill the mission's declared output-schema keys from the agent
        # outcome so an autonomous run can validate: without this, an
        # autonomous mission could never SUCCEED, because validate_results
        # requires every output key to be present and nothing ever set
        # them. Keys the outcome cannot fill (e.g. root_causes) stay
        # missing and validation still fails honestly.
        result: dict = {}
        required_keys = list((mission.get("output_schema") or {}).get("properties", {}).keys())
        summary = outcome.get("summary", "")
        if "answer" in required_keys:
            result["answer"] = summary
        if "summary" in required_keys:
            result["summary"] = summary
        if "actions_taken" in required_keys:
            result["actions_taken"] = outcome.get("actions_taken") or []
        if "evidence" in required_keys:
            result["evidence"] = outcome.get("evidence") or []
        return {
            "evidence": outcome.get("evidence") or [],
            "result": result,
            "notes": notes,
            "errors": outcome.get("errors") or [],
        }
    except Exception as exc:  # noqa: BLE001 — model/package problems are state
        logger.warning("Deep agent unavailable: %s", exc)
        return {"errors": [f"Deep agent unavailable: {exc}"]}


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
    if mode == "hybrid" and len(state.get("evidence") or []) < MIN_EVIDENCE_FOR_HYBRID:
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
        with PostgresSaver.from_conn_string(settings.database_url) as checkpointer:
            try:
                checkpointer.setup()
            except Exception as exc:  # noqa: BLE001 — checkpoint is best-effort
                logger.warning("Could not set up Postgres checkpointer: %s", exc)
                yield None
                return
            yield checkpointer
    except Exception as exc:  # noqa: BLE001 — checkpoint is best-effort
        logger.warning("Could not set up Postgres checkpointer: %s", exc)
        yield None


def run_mission(mission: dict, input_data: dict, execution_id: str) -> dict:
    """Run one execution through the Mission Graph and return final state.

    Reuses the module-level compiled `mission_graph` unless a Postgres
    checkpointer is enabled (then a graph is built with it).
    """
    # Tracing for direct callers (worker, eval, langgraph dev): the API
    # lifespan already calls this for its own process. No-op without a key.
    from infra.langsmith_setup import setup_langsmith

    setup_langsmith()
    setup_mlflow_tracing()
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
    }
    # The checkpointer context must stay open for the whole invoke: the
    # saver borrows the connection from PostgresSaver.from_conn_string.
    with get_checkpointer() as checkpointer:
        graph = (
            build_mission_graph(checkpointer=checkpointer)
            if checkpointer
            else mission_graph
        )
        config = (
            {"configurable": {"thread_id": execution_id}} if checkpointer else None
        )
        final_state = graph.invoke(initial_state, config=config)
    return dict(final_state)
