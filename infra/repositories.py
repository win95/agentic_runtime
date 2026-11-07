"""Repository layer — data access for tools and capability graphs.

Each function tries the real store first and falls back to clearly-marked
stub data (`"stub": True` in the second return value) when the store is
not configured, the table/collection is missing, or the query returns
nothing usable. Stub data is demo data — replace it with real repositories
once the data schema exists. Never silently present stub data as real.

Return shape: (rows, is_stub).
"""

from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger(__name__)


def _days(start_date: str, end_date: str) -> int:
    try:
        days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days
        return max(days, 1)
    except (ValueError, TypeError):
        return 30


def search_fault_history(asset_id: str, start_date: str, end_date: str):
    """Fault history from PostgreSQL table `fault_history` (if it exists)."""
    try:
        from infra.postgres import run_query

        rows = run_query(
            "SELECT code, title, occurred_at, severity FROM fault_history "
            "WHERE asset_id = :asset_id AND occurred_at >= :start_date "
            "AND occurred_at <= :end_date ORDER BY occurred_at DESC LIMIT 50",
            {"asset_id": asset_id, "start_date": start_date, "end_date": end_date},
        )
        return rows, False
    except Exception as exc:  # noqa: BLE001 — stub fallback is the design
        logger.debug("fault_history fallback to stub: %s", exc)
        return [
            {
                "code": "F-101",
                "title": f"Door sensor intermittent fault on {asset_id}",
                "occurred_at": start_date or "2026-10-05",
                "severity": "medium",
            },
            {
                "code": "F-204",
                "title": f"Traction motor overheating on {asset_id}",
                "occurred_at": start_date or "2026-10-12",
                "severity": "high",
            },
        ], True


def get_maintenance_records(asset_id: str):
    """Maintenance records from PostgreSQL table `maintenance_records`."""
    try:
        from infra.postgres import run_query

        rows = run_query(
            "SELECT date, action, technician FROM maintenance_records "
            "WHERE asset_id = :asset_id ORDER BY date DESC LIMIT 20",
            {"asset_id": asset_id},
        )
        return rows, False
    except Exception as exc:  # noqa: BLE001 — stub fallback is the design
        logger.debug("maintenance_records fallback to stub: %s", exc)
        return [
            {
                "date": "2026-09-20",
                "action": f"Routine inspection of {asset_id}",
                "technician": "demo-team",
            },
            {
                "date": "2026-10-08",
                "action": f"Door sensor replaced on {asset_id}",
                "technician": "demo-team",
            },
        ], True


def search_documents(query: str, limit: int = 5):
    """Technical documents via Qdrant vector search."""
    try:
        from infra.qdrant_client import search_documents as qdrant_search

        docs = qdrant_search(query, limit=limit)
        return docs, False
    except Exception as exc:  # noqa: BLE001 — stub fallback is the design
        logger.debug("Qdrant search fallback to stub: %s", exc)
        return [
            {
                "title": "Elevator Maintenance Manual (demo)",
                "content": f"Demo document matching query: {query}",
                "score": 0.0,
            }
        ], True


def query_asset_relationships(asset_id: str):
    """Asset relationships from the FalkorDB knowledge graph."""
    try:
        from infra.falkor_client import query_asset_relationships as falkor_query

        relations = falkor_query(asset_id)
        return relations, False
    except Exception as exc:  # noqa: BLE001 — stub fallback is the design
        logger.debug("FalkorDB query fallback to stub: %s", exc)
        return [
            {"from": asset_id, "type": "PART_OF", "to": "building-B1"},
            {"from": asset_id, "type": "USES", "to": "motor-M2"},
        ], True


def get_sensor_data(asset_id: str, start_date: str, end_date: str):
    """Sensor readings.

    STUB ONLY: no sensor store exists in this scaffold. Wire a real time
    series store here (e.g. TimescaleDB/InfluxDB) before using in production.
    """
    return [
        {
            "metric": "motor_temperature_c",
            "value": 72.5,
            "at": start_date or "2026-10-01",
            "asset_id": asset_id,
        },
        {
            "metric": "vibration_mm_s",
            "value": 4.1,
            "at": start_date or "2026-10-01",
            "asset_id": asset_id,
        },
    ], True


def calculate_failure_rate(asset_id: str, start_date: str, end_date: str):
    """Failure rate = fault count / days in range (computed from fault history)."""
    rows, is_stub = search_fault_history(asset_id, start_date, end_date)
    days = _days(start_date, end_date)
    count = len(rows)
    return {
        "fault_count": count,
        "days": days,
        "failure_rate_per_day": round(count / days, 4),
    }, is_stub
