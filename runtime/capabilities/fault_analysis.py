"""Fault Analysis capability — LangGraph subgraph (Layer 5).

Pipeline: fault_history (SQL) → maintenance (SQL) → knowledge_retrieval
(RAG/Qdrant) → asset_relations (FalkorDB) → evidence_aggregation.

Each node appends evidence items; the aggregation node drafts the mission
result (root_causes / evidence / recommendations). Repository calls go
through infra/repositories, which query the real stores. When a store is
unreachable the node records an error evidence item instead of inventing
data — the run still completes and validation reports what is missing.

Evidence kinds: "fault" / "maintenance" / "document" / "relation" are
real findings. "error" is a store failure. "empty" is a store that was
consulted and had nothing in range — information, not a finding: empty
items never become root causes and never count toward the hybrid
escalation threshold (see runtime/mission_graph.py).
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from infra import repositories
from ontology.canonical.schema import DEFAULT_RELATION_TYPE
from runtime.retrieval_weights import DEFAULT_DOC_LIMIT
from runtime.state import MissionState

logger = logging.getLogger(__name__)

# Knowledge-node question template: the hybrid/vector search the
# knowledge_retrieval node runs for an asset. Value unchanged.
KNOWLEDGE_RETRIEVAL_QUERY_TEMPLATE = "faults maintenance {asset_id}"


def _evidence(source: str, kind: str, asset_id: str, summary: str, data=None, stub=False) -> dict:
    return {
        "source": source,
        "kind": kind,
        "asset_id": asset_id,
        "summary": summary,
        "data": data or {},
        "stub": stub,
    }


def fault_history_node(state: MissionState) -> dict:
    inp = state.get("input") or {}
    asset_id = inp.get("asset_id", "")
    try:
        rows, is_stub = repositories.search_fault_history(
            asset_id, inp.get("start_date", ""), inp.get("end_date", "")
        )
    except Exception as exc:  # noqa: BLE001 — any store failure is error evidence
        return {
            "evidence": [
                _evidence(
                    "fault_history", "error", asset_id, f"Fault history unavailable: {exc}"
                )
            ]
        }
    items = [
        _evidence(
            "fault_history",
            "fault",
            asset_id,
            f"{row.get('code') or 'missing code'}: {row.get('title') or 'missing title'}",
            row,
            is_stub,
        )
        for row in rows
    ]
    if not items:
        items = [
            _evidence(
                "fault_history", "empty", asset_id, "No faults recorded in range", {}, is_stub
            )
        ]
    return {"evidence": items}


def maintenance_node(state: MissionState) -> dict:
    inp = state.get("input") or {}
    asset_id = inp.get("asset_id", "")
    try:
        rows, is_stub = repositories.get_maintenance_records(asset_id)
    except Exception as exc:  # noqa: BLE001 — any store failure is error evidence
        return {
            "evidence": [
                _evidence(
                    "maintenance", "error", asset_id, f"Maintenance records unavailable: {exc}"
                )
            ]
        }
    items = [
        _evidence(
            "maintenance",
            "maintenance",
            asset_id,
            f"Maintenance {row.get('date', '')}: {row.get('action', '')}".strip(),
            row,
            is_stub,
        )
        for row in rows
    ]
    if not items:
        items = [
            _evidence(
                "maintenance", "empty", asset_id, "No maintenance records found", {}, is_stub
            )
        ]
    return {"evidence": items}


def _load_knowledge_documents(retrieval: dict, asset_id: str, execution_id: str = ""):
    """Document rows for the knowledge_retrieval node.

    Hybrid first when the mission opts in (Seam B, never a plan —
    the deterministic spine holds no LLM call). On any hybrid failure
    falls back to plain vector search, so the node keeps working
    without live stores and the ``repositories`` mock seam in the
    eval harness / manual scripts keeps working. Raises only when
    both arms fail.
    """
    question = KNOWLEDGE_RETRIEVAL_QUERY_TEMPLATE.format(asset_id=asset_id)
    if retrieval.get("hybrid"):
        from runtime.retrieval_live import hybrid_documents

        try:
            # B2: the execution's retriever — the SessionStart hook
            # (Seam A) reuses this same retriever when it runs.
            return hybrid_documents(
                question, execution_id=execution_id or None
            )["docs"], False
        except Exception as exc:  # noqa: BLE001 — fallback, never fatal
            logger.warning(
                "knowledge_retrieval: hybrid documents failed (%s); "
                "falling back to vector search",
                exc,
            )
    return repositories.search_documents(question, limit=DEFAULT_DOC_LIMIT)


def knowledge_retrieval_node(state: MissionState) -> dict:
    inp = state.get("input") or {}
    asset_id = inp.get("asset_id", "")
    retrieval = (
        (state.get("mission") or {}).get("execution") or {}
    ).get("retrieval") or {}
    try:
        docs, is_stub = _load_knowledge_documents(
            retrieval, asset_id, execution_id=state.get("execution_id") or ""
        )
    except Exception as exc:  # noqa: BLE001 — any store failure is error evidence
        # No invented documents: record the failure as evidence.
        return {
            "evidence": [
                _evidence(
                    "knowledge_retrieval",
                    "error",
                    asset_id,
                    f"Technical documents unavailable: {exc}",
                )
            ]
        }
    items = [
        _evidence(
            "knowledge_retrieval",
            "document",
            asset_id,
            f"Document: {doc.get('title', '')}",
            doc,
            is_stub,
        )
        for doc in docs
    ]
    if not items:
        items = [
            _evidence(
                "knowledge_retrieval",
                "empty",
                asset_id,
                "No technical documents found",
                {},
                is_stub,
            )
        ]
    return {"evidence": items}


def asset_relations_node(state: MissionState) -> dict:
    inp = state.get("input") or {}
    asset_id = inp.get("asset_id", "")
    try:
        relations, is_stub = repositories.query_asset_relationships(asset_id)
    except Exception as exc:  # noqa: BLE001 — any store failure is error evidence
        return {
            "evidence": [
                _evidence(
                    "asset_relations", "error", asset_id, f"Asset relations unavailable: {exc}"
                )
            ]
        }
    items = [
        _evidence(
            "asset_relations",
            "relation",
            asset_id,
            f"{rel.get('from', asset_id)} "
            f"-[{rel.get('type', DEFAULT_RELATION_TYPE)}]-> {rel.get('to', '')}",
            rel,
            is_stub,
        )
        for rel in relations
    ]
    if not items:
        items = [
            _evidence(
                "asset_relations", "empty", asset_id, "No asset relations found", {}, is_stub
            )
        ]
    return {"evidence": items}


def evidence_aggregation_node(state: MissionState) -> dict:
    evidence = state.get("evidence") or []
    # Only real findings become root causes. "error" items (a store was
    # down) and "empty" items (a store was consulted and had nothing)
    # are information about what is missing, not causes — previously the
    # "No faults recorded…" / "No maintenance records found" placeholders
    # carried kind fault/maintenance and leaked into root_causes.
    root_causes = [
        item["summary"] for item in evidence if item.get("kind") in ("fault", "maintenance")
    ][:5]
    if not root_causes:
        root_causes = ["unknown — insufficient evidence"]
    recommendations = [
        "Review the collected evidence and confirm the suspected root cause on site.",
        "Schedule preventive maintenance for the affected component.",
    ]
    result = {
        "root_causes": root_causes,
        "evidence": evidence,
        "recommendations": recommendations,
    }
    return {"result": result}


def build_fault_analysis_graph():
    """Compile the fault_analysis capability subgraph."""
    graph = StateGraph(MissionState)
    graph.add_node("fault_history", fault_history_node)
    graph.add_node("knowledge_retrieval", knowledge_retrieval_node)
    graph.add_node("asset_relations", asset_relations_node)
    graph.add_node("evidence_aggregation", evidence_aggregation_node)
    graph.add_edge(START, "fault_history")
    graph.add_edge("fault_history", "knowledge_retrieval")
    graph.add_edge("knowledge_retrieval", "asset_relations")
    graph.add_edge("asset_relations", "evidence_aggregation")
    graph.add_edge("evidence_aggregation", END)
    return graph.compile()
