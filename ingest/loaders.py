"""Loaders — validated rows into the four stores (Option B).

Every loader is idempotent: Postgres rows upsert by fixed ``id``
(``db.merge``), Qdrant points upsert by uuid5 of the document id
(existing points are skipped), FalkorDB relations MERGE. Re-running a
batch therefore changes nothing.

Return shape per loader: ``(ok, failures)`` where ``failures`` is a
list of ``{"row": <raw dict>, "error": <str>}``. Postgres loaders
isolate failures per row (one SAVEPOINT per row, one commit at the
end); the Qdrant/FalkorDB loaders work file-at-a-time, so a failure
there fails every row in the file with the same error.
"""

from __future__ import annotations

import logging

from app.models import FaultHistory, MaintenanceRecord, SensorReading
from ingest.schemas import (
    DocumentRow,
    FaultRow,
    MaintenanceRow,
    RelationRow,
    SensorRow,
)

logger = logging.getLogger(__name__)


def _merge_rows(db, model, rows) -> tuple[int, list[dict]]:
    """Merge rows one by one; a bad row is collected, not fatal.

    Each row merges inside its own SAVEPOINT (``db.begin_nested()``),
    so a failing row rolls back only itself — previously the
    ``db.rollback()`` here discarded every uncommitted row merged
    before it while ``ok`` had already counted them, so the report
    claimed rows that were never stored. One commit at the end still
    applies to the surviving rows.
    """
    ok = 0
    failures: list[dict] = []
    for row in rows:
        try:
            with db.begin_nested():
                db.merge(model(**row.model_dump()))
                db.flush()
            ok += 1
        except Exception as exc:  # noqa: BLE001 — per-row isolation
            failures.append({"row": row.model_dump(mode="json"), "error": str(exc)})
    if ok:
        db.commit()
    return ok, failures


def load_faults(db, rows: list[FaultRow]) -> tuple[int, list[dict]]:
    return _merge_rows(db, FaultHistory, rows)


def load_maintenance(db, rows: list[MaintenanceRow]) -> tuple[int, list[dict]]:
    return _merge_rows(db, MaintenanceRecord, rows)


def load_sensor_readings(db, rows: list[SensorRow]) -> tuple[int, list[dict]]:
    return _merge_rows(db, SensorReading, rows)


def load_documents(rows: list[DocumentRow], embeddings=None) -> tuple[int, list[dict]]:
    """Embed + upsert documents into Qdrant (file-level: all or nothing)."""
    from infra.qdrant_client import ensure_collection, probe_dimension, upsert_documents

    docs = [row.model_dump(mode="json") for row in rows]
    try:
        dim = probe_dimension(embeddings)
        ensure_collection(dim)
        result = upsert_documents(docs, embeddings)
    except Exception as exc:  # noqa: BLE001 — file-level failure
        return 0, [{"row": d, "error": str(exc)} for d in docs]
    ok = result.get("upserted", 0) + result.get("skipped", 0)
    return ok, []


def load_relations(rows: list[RelationRow]) -> tuple[int, list[dict]]:
    """Write relations into canonical ontology + projection drain."""
    from infra.falkor_client import merge_asset_relations

    tuples = [
        (r.from_id, r.from_label, r.relation, r.to_id, r.to_label) for r in rows
    ]
    try:
        merge_asset_relations(tuples)
    except Exception as exc:  # noqa: BLE001 — file-level failure
        return 0, [
            {"row": r.model_dump(mode="json"), "error": str(exc)} for r in rows
        ]
    return len(rows), []


LOADERS = {
    "faults": load_faults,
    "maintenance": load_maintenance,
    "sensor_readings": load_sensor_readings,
    "documents": load_documents,
    "relations": load_relations,
}
