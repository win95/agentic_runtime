"""Fault Analysis capability — LangGraph subgraph (Layer 5).

Pipeline: fault_history (SQL) → maintenance (SQL) → knowledge_retrieval
(RAG/Qdrant) → asset_relations (FalkorDB) → evidence_aggregation.

Each node appends evidence items; the aggregation node drafts the mission
result (root_causes / evidence / recommendations). Repository calls go
through infra/repositories, which return clearly-marked stub data when the
real store is not configured or has no rows.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from infra import repositories
from runtime.state import MissionState


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
    rows, is_stub = repositories.search_fault_history(
        asset_id, inp.get("start_date", ""), inp.get("end_date", "")
    )
    items = [
        _evidence(
            "fault_history",
            "fault",
            asset_id,
            f"{row.get('code', 'FAULT')}: {row.get('title', 'Untitled fault')}",
            row,
            is_stub,
        )
        for row in rows
    ]
    if not items:
        items = [
            _evidence(
                "fault_history", "fault", asset_id, "No faults recorded in range", {}, is_stub
            )
        ]
    return {"evidence": items}


def maintenance_node(state: MissionState) -> dict:
    inp = state.get("input") or {}
    asset_id = inp.get("asset_id", "")
    rows, is_stub = repositories.get_maintenance_records(asset_id)
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
                "maintenance", "maintenance", asset_id, "No maintenance records found", {}, is_stub
            )
        ]
    return {"evidence": items}


def knowledge_retrieval_node(state: MissionState) -> dict:
    inp = state.get("input") or {}
    asset_id = inp.get("asset_id", "")
    docs, is_stub = repositories.search_documents(
        f"faults maintenance {asset_id}", limit=5
    )
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
                "document",
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
    relations, is_stub = repositories.query_asset_relationships(asset_id)
    items = [
        _evidence(
            "asset_relations",
            "relation",
            asset_id,
            f"{rel.get('from', asset_id)} -[{rel.get('type', 'RELATED')}]-> {rel.get('to', '')}",
            rel,
            is_stub,
        )
        for rel in relations
    ]
    if not items:
        items = [
            _evidence(
                "asset_relations", "relation", asset_id, "No asset relations found", {}, is_stub
            )
        ]
    return {"evidence": items}


def evidence_aggregation_node(state: MissionState) -> dict:
    evidence = state.get("evidence") or []
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
    graph.add_node("maintenance", maintenance_node)
    graph.add_node("knowledge_retrieval", knowledge_retrieval_node)
    graph.add_node("asset_relations", asset_relations_node)
    graph.add_node("evidence_aggregation", evidence_aggregation_node)
    graph.add_edge(START, "fault_history")
    graph.add_edge("fault_history", "maintenance")
    graph.add_edge("maintenance", "knowledge_retrieval")
    graph.add_edge("knowledge_retrieval", "asset_relations")
    graph.add_edge("asset_relations", "evidence_aggregation")
    graph.add_edge("evidence_aggregation", END)
    return graph.compile()
