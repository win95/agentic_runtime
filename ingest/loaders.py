"""Loaders — validated rows into the four stores (Option B).

Every loader is idempotent: Postgres rows upsert by fixed ``id``
(``db.merge``), Qdrant points upsert by uuid5 of the document id
(existing points are skipped), canonical relations MERGE. Re-running a
batch therefore changes nothing.

Return shape per loader: ``(ok, failures)`` where ``failures`` is a
list of ``{"row": <raw dict>, "error": <str>}``. Postgres loaders
isolate failures per row (one SAVEPOINT per row, one commit at the
end); the Qdrant loader works file-at-a-time, so a failure there fails
every row in the file with the same error. The relations loader
is per-row like the Postgres loaders (one SAVEPOINT per row) but
commits nothing itself — it shares the batch session and the caller
commits.
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
    """Embed + upsert documents into Qdrant (file-level: all or nothing).

    Phase 1: every doc carries payload node links. A row that omits
    them gets the grounded links from ``runtime.retrieval_linking``
    (demo docs); an explicit row value always wins.
    """
    from infra.qdrant_client import ensure_collection, probe_dimension, upsert_documents
    from runtime.retrieval_linking import with_document_links

    docs = [with_document_links(row.model_dump(mode="json")) for row in rows]
    try:
        dim = probe_dimension(embeddings)
        ensure_collection(dim)
        result = upsert_documents(docs, embeddings)
    except Exception as exc:  # noqa: BLE001 — file-level failure
        return 0, [{"row": d, "error": str(exc)} for d in docs]
    ok = result.get("upserted", 0) + result.get("skipped", 0)
    return ok, []


def load_relations(
    db, rows: list[RelationRow], *, source: str = "ingest"
) -> tuple[int, list[dict]]:
    """Write relations into the canonical ontology (same transaction as
    the batch — the caller commits). One SAVEPOINT per row, like
    ``_merge_rows``: a bad row is quarantined, not fatal to the file.

    Each row gets a stable per-relation ``source_event_id`` so a
    re-run replays instead of versioning again. Endpoints are created
    only when missing: a relation row carries no endpoint properties,
    so writing them would null out properties an earlier write stored.
    Projection is not drained here; beat/CLI own it.
    """
    from ontology.canonical.provenance import WriteProvenance
    from ontology.canonical.repository import CanonicalOntologyRepository
    from ontology.canonical.write_service import OntologyWriteService

    writer = OntologyWriteService(CanonicalOntologyRepository(db))
    trace_id = f"ingest:{source}"
    ok = 0
    failures: list[dict] = []
    for row in rows:
        event_id = (
            f"ingest:{source}:{row.from_id}:{row.relation}:{row.to_id}"
        )
        prov = WriteProvenance(trace_id=trace_id, source_event_id=event_id)
        try:
            with db.begin_nested():
                writer.ensure_entity(
                    entity_type=row.from_label,
                    entity_key=row.from_id,
                    properties=None,
                    provenance=prov,
                )
                writer.ensure_entity(
                    entity_type=row.to_label,
                    entity_key=row.to_id,
                    properties=None,
                    provenance=prov,
                )
                writer.write_relationship(
                    relationship_type=row.relation,
                    source_type=row.from_label,
                    source_key=row.from_id,
                    target_type=row.to_label,
                    target_key=row.to_id,
                    properties=None,
                    provenance=prov,
                )
                db.flush()
            ok += 1
        except Exception as exc:  # noqa: BLE001 — per-row isolation
            failures.append({"row": row.model_dump(mode="json"), "error": str(exc)})
    return ok, failures


LOADERS = {
    "faults": load_faults,
    "maintenance": load_maintenance,
    "sensor_readings": load_sensor_readings,
    "documents": load_documents,
    "relations": load_relations,
}
