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

import logging

from langgraph.graph import END, START, StateGraph

from app.config import settings
from infra.mlflow_setup import setup_mlflow_tracing
from runtime.capabilities.fault_analysis import build_fault_analysis_graph
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
    context = {
        "mission_id": state.get("mission_id"),
        "execution_id": state.get("execution_id"),
        "tenant": "default",
        "capabilities": mission.get("capabilities") or [],
    }
    try:
        from runtime.deep_agent import build_deep_agent, run_deep_agent
        from runtime.tools.mcp_adapter import load_mcp_tools
        from runtime.tools.registry import registry

        tools = registry.get_langchain_tools(context)
        tools.extend(load_mcp_tools())
        agent = build_deep_agent(tools)
        outcome = run_deep_agent(agent, mission, input_data)
        notes = [f"Deep agent: {outcome.get('summary', '')}"] if outcome.get("summary") else []
        return {
            "evidence": outcome.get("evidence") or [],
            "notes": notes,
            "errors": outcome.get("errors") or [],
        }
    except Exception as exc:  # noqa: BLE001 — model/package problems are state
        logger.warning("Deep agent unavailable: %s", exc)
        return {"errors": [f"Deep agent unavailable: {exc}"]}


def validate_results(state: MissionState) -> dict:
    """Check evidence, output schema and success criteria."""
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


def get_checkpointer():
    """Postgres checkpointer (optional, off by default).

    Needs CHECKPOINT_ENABLED=true and the `postgres-checkpoint` extra.
    Returns None when unavailable — the graph then runs without persistent
    checkpoints.
    """
    if not settings.checkpoint_enabled:
        return None
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError:
        try:
            from langgraph_checkpoint_postgres import PostgresSaver
        except ImportError:
            logger.warning("Postgres checkpointer package not installed; running without it.")
            return None
    try:
        checkpointer = PostgresSaver.from_conn_string(settings.database_url)
        checkpointer.setup()
        return checkpointer
    except Exception as exc:  # noqa: BLE001 — checkpoint is best-effort
        logger.warning("Could not set up Postgres checkpointer: %s", exc)
        return None


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
    checkpointer = get_checkpointer()
    graph = build_mission_graph(checkpointer=checkpointer) if checkpointer else mission_graph
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
    config = {"configurable": {"thread_id": execution_id}} if checkpointer else None
    final_state = graph.invoke(initial_state, config=config)
    return dict(final_state)
