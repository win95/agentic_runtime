"""Repository layer — data access for tools and capability graphs.

MVP rule: NO invented data. Every function queries a real store and
returns what is actually there (possibly an empty list). When a store is
unreachable or misconfigured the function raises a clear RuntimeError;
callers (capability nodes, tools) surface that error in their result
instead of fabricating rows.

Return shape: (rows, is_stub) where is_stub is False for real data.
`get_sensor_data` is the single documented exception: no sensor store
exists yet, so it still returns clearly-marked stub data.

Date/datetime values are converted to ISO strings so results stay JSON
serializable (ExecutionResult stores evidence as JSON).
"""

from __future__ import annotations

import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)


def _days(start_date: str, end_date: str) -> int:
    try:
        days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days
        return max(days, 1)
    except (ValueError, TypeError):
        return 30


def _jsonable(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (datetime, date)):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


def search_fault_history(asset_id: str, start_date: str, end_date: str):
    """Fault history from PostgreSQL table `fault_history`.

    Raises RuntimeError when the DB/table is unreachable. Returns
    ([], False) when the asset simply has no faults in the range.
    """
    from infra.postgres import run_query

    # A bare date end ("2026-10-31") means the whole day, not midnight.
    end_inclusive = end_date
    if end_date and len(end_date) == 10:
        end_inclusive = f"{end_date}T23:59:59"
    try:
        rows = run_query(
            "SELECT id, asset_id, code, title, description, severity, "
            "occurred_at, resolved_at, technician, action FROM fault_history "
            "WHERE asset_id = :asset_id AND occurred_at >= :start_date "
            "AND occurred_at <= :end_date ORDER BY occurred_at DESC LIMIT 50",
            {"asset_id": asset_id, "start_date": start_date, "end_date": end_inclusive},
        )
    except Exception as exc:  # noqa: BLE001 — wrapped with a clear message
        raise RuntimeError(
            f"Could not read fault_history from PostgreSQL: {exc}. "
            "Is the database up and seeded? Run `make seed`."
        ) from exc
    return [_jsonable(r) for r in rows], False


def get_maintenance_records(asset_id: str):
    """Maintenance records from PostgreSQL table `maintenance_records`."""
    from infra.postgres import run_query

    try:
        rows = run_query(
            "SELECT id, asset_id, date, action, technician, notes "
            "FROM maintenance_records WHERE asset_id = :asset_id "
            "ORDER BY date DESC LIMIT 20",
            {"asset_id": asset_id},
        )
    except Exception as exc:  # noqa: BLE001 — wrapped with a clear message
        raise RuntimeError(
            f"Could not read maintenance_records from PostgreSQL: {exc}. "
            "Is the database up and seeded? Run `make seed`."
        ) from exc
    return [_jsonable(r) for r in rows], False


def search_documents(query: str, limit: int = 5):
    """Technical documents via real Qdrant vector search.

    Raises RuntimeError when Qdrant or the embedding model is unreachable,
    or when the collection has not been seeded yet. Never returns
    invented documents.
    """
    from infra.qdrant_client import search_documents as qdrant_search

    docs = qdrant_search(query, limit=limit)
    return docs, False


def query_asset_relationships(asset_id: str):
    """Asset relationships from the FalkorDB knowledge graph."""
    from infra.falkor_client import query_asset_relationships as falkor_query

    try:
        relations = falkor_query(asset_id)
    except Exception as exc:  # noqa: BLE001 — wrapped with a clear message
        raise RuntimeError(
            f"Could not query FalkorDB for asset {asset_id}: {exc}. "
            "Is FalkorDB up and seeded? Run `make seed`."
        ) from exc
    return relations, False


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
