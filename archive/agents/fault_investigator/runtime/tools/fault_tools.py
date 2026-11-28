"""Fault-analysis tools (Layer 5) — small callable actions.

Each tool has a name, description, typed input schema (inferred by
langchain from the signature), permission and timeout/retry metadata in
the registry. Implementations delegate to `infra.repositories` (real
stores when configured, clearly-marked stub data otherwise).
"""

from __future__ import annotations

from langchain_core.tools import tool

from infra import repositories
from runtime.tools.registry import registry


@tool
def search_fault_history(asset_id: str, start_date: str, end_date: str) -> dict:
    """Search equipment fault history within a date range."""
    try:
        rows, is_stub = repositories.search_fault_history(asset_id, start_date, end_date)
    except Exception as exc:  # noqa: BLE001 — a store failure is data, not a crash
        return {"asset_id": asset_id, "faults": [], "error": str(exc), "stub": False,
                "provenance": "postgres:fault_history"}
    return {"asset_id": asset_id, "faults": rows, "stub": is_stub,
            "provenance": "postgres:fault_history"}


@tool
def get_maintenance_records(asset_id: str) -> dict:
    """Get maintenance records for an asset."""
    try:
        rows, is_stub = repositories.get_maintenance_records(asset_id)
    except Exception as exc:  # noqa: BLE001 — a store failure is data, not a crash
        return {"asset_id": asset_id, "records": [], "error": str(exc), "stub": False,
                "provenance": "postgres:maintenance_records"}
    return {"asset_id": asset_id, "records": rows, "stub": is_stub,
            "provenance": "postgres:maintenance_records"}


@tool
def search_technical_documents(query: str, limit: int = 5) -> dict:
    """Search technical documents (RAG / Qdrant).

    When Qdrant or the embedding model is unreachable, returns a clear
    error instead of invented documents.
    """
    try:
        docs, is_stub = repositories.search_documents(query, limit=limit)
    except Exception as exc:  # noqa: BLE001 — a store failure is data, not a crash
        return {"query": query, "documents": [], "error": str(exc), "stub": False,
                "provenance": "qdrant:technical_documents"}
    return {"query": query, "documents": docs, "stub": is_stub,
            "provenance": "qdrant:technical_documents"}


@tool
def query_asset_relationships(asset_id: str) -> dict:
    """Query asset relationships from the knowledge graph (Neo4j)."""
    try:
        relations, is_stub = repositories.query_asset_relationships(asset_id)
    except Exception as exc:  # noqa: BLE001 — a store failure is data, not a crash
        return {"asset_id": asset_id, "relations": [], "error": str(exc), "stub": False,
                "provenance": "neo4j:asset_relationships"}
    return {"asset_id": asset_id, "relations": relations, "stub": is_stub,
            "provenance": "neo4j:asset_relationships"}


@tool
def get_sensor_data(asset_id: str, start_date: str, end_date: str) -> dict:
    """Get sensor data for an asset within a date range.

    Reads real rows from the Postgres ``sensor_readings`` table; an asset
    with no readings in the range returns an empty list (never invented
    data).
    """
    try:
        data, is_stub = repositories.get_sensor_data(asset_id, start_date, end_date)
    except Exception as exc:  # noqa: BLE001 — a store failure is data, not a crash
        return {"asset_id": asset_id, "readings": [], "error": str(exc), "stub": False,
                "provenance": "postgres:sensor_readings"}
    return {"asset_id": asset_id, "readings": data, "stub": is_stub,
            "provenance": "postgres:sensor_readings"}


@tool
def calculate_failure_rate(asset_id: str, start_date: str, end_date: str) -> dict:
    """Calculate the failure rate (faults per day) for an asset in a range."""
    try:
        result, is_stub = repositories.calculate_failure_rate(asset_id, start_date, end_date)
    except Exception as exc:  # noqa: BLE001 — a store failure is data, not a crash
        return {"asset_id": asset_id, "error": str(exc), "stub": False,
                "provenance": "postgres:fault_history (computed)"}
    return {"asset_id": asset_id, **result, "stub": is_stub,
            "provenance": "postgres:fault_history (computed)"}


# Register all six tools with their business capability + permission.
# All six are read-only (search/get/query/calculate) — Layer 4.
registry.register(search_fault_history, capability="fault_history",
                  permission="fault:read", timeout_seconds=30, read_only=True)
registry.register(get_maintenance_records, capability="maintenance_analysis",
                  permission="maintenance:read", timeout_seconds=30, read_only=True)
registry.register(search_technical_documents, capability="knowledge_retrieval",
                  permission="docs:read", timeout_seconds=60, read_only=True)
registry.register(query_asset_relationships, capability="asset_relations",
                  permission="graph:read", timeout_seconds=30, read_only=True)
registry.register(get_sensor_data, capability="sensor_data",
                  permission="sensors:read", timeout_seconds=30, read_only=True)
registry.register(calculate_failure_rate, capability="fault_history",
                  permission="fault:read", timeout_seconds=30, read_only=True)
