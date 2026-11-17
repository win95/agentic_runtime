"""Manual test: batch ingest end-to-end across all row types (live stack).

``documents_ingest.py`` covers the Document Management API and one
documents batch (Qdrant + DOCUMENTED_BY). This script exercises the rest
of the production path in ``docs/ingest.md``:

- **dry-run** — all five row types validate and write NOTHING (no
  ``ingest_runs``, no ``ingest_quarantine``, no store writes), including
  a bad row, invalid JSON, a non-array file and an unknown file type;
- **real batch** — one inbox with faults, maintenance, sensor_readings,
  documents (one linked to ``O21``, one plain) and relations; assert
  every row lands in its store, DOCUMENTED_BY is derived, the projection
  queue drains, and Neo4j sees the new edges / documents;
- **watermark** — a re-run skips SUCCEEDED files without writing; a
  forced re-run replays every row without duplicating Postgres rows,
  Qdrant points, canonical edges or ontology versions;
- **PARTIAL + quarantine** — one valid + one invalid fault row produces
  PARTIAL, the bad row is quarantined, a PARTIAL file is re-processed
  (not watermark-skipped), the quarantine rows are fixed in place and
  ``retry_quarantine`` loads them;
- **file-level failures** — malformed JSON, a non-array document and an
  unknown file type record FAILED runs with errors and no rows;
- **MinIO inbox** — ``MinioInboxSource`` reads an object from the
  ``ingest-inbox`` bucket, leaves the object in place, and the watermark
  skips it on the next pass;
- **service wrapper** — ``app.services.ingest_service.run_ingest_batch``
  (what the Celery beat task calls) runs a batch from an explicit inbox;
- **worker task** — ``app.worker.tasks.run_ingest_batch`` executed inside
  the running ``worker`` container (SKIP when the container or docker
  CLI is unavailable) reads its configured inbox, loads the row, and
  watermark-skips it on the second call.

Everything this run writes is tagged with a per-run hex token, and the
default mode removes it afterwards (unless ``--keep``) then compares the
Postgres / Qdrant / Neo4j totals against a baseline captured after the
projection queue is settled — so a passing run also proves cleanup left
the seeded stack unchanged.

Requires the live stack: Postgres, Qdrant, Neo4j, MinIO and the
embedding endpoint (documents ingest embeds content). No API call is
needed. Run from the repo root with the project venv:

    .venv/bin/python scripts/manual/ingest_batch.py
    .venv/bin/python scripts/manual/ingest_batch.py --quiet
    .venv/bin/python scripts/manual/ingest_batch.py --keep   # leave rows

Exit code: 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import io
import json
import secrets
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _api import set_quiet  # noqa: E402
from agent_config_check import Report  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.db import SessionLocal  # noqa: E402
from app.models import (  # noqa: E402
    FaultHistory,
    IngestQuarantine,
    IngestRun,
    MaintenanceRecord,
    SensorReading,
)
from infra import minio_client, qdrant_client  # noqa: E402
from infra.qdrant_client import get_collection_dim  # noqa: E402
from ingest.run import retry_quarantine, run_batch  # noqa: E402
from ingest.sources import LocalDirSource, MinioInboxSource  # noqa: E402
from ontology.canonical.types import entity_ref, relationship_ref  # noqa: E402
from ontology.knowledge import graph_query  # noqa: E402
from ontology.models import (  # noqa: E402
    OntologyEntity,
    OntologyProjectionState,
    OntologyRelationship,
)
from ontology.projection.engine import drain_pending_projections  # noqa: E402
from ontology.projection.provider import get_graph_store  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "ingest" / "samples"
ASSET_ID = "O21"
OTHER_ASSET = "O22"
BUILDING = "building-B2"
ROW_TYPES = ("faults", "maintenance", "sensor_readings", "documents", "relations")


# --- helpers -----------------------------------------------------------------


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{doc_id}"))


def _write_json(directory: str | Path, name: str, rows) -> None:
    Path(directory, name).write_text(
        json.dumps(rows, default=str), encoding="utf-8"
    )


def _write_text(directory: str | Path, name: str, body: bytes) -> None:
    Path(directory, name).write_bytes(body)


def _load_sample(row_type: str) -> list[dict]:
    return json.loads((SAMPLES / f"{row_type}.json").read_text(encoding="utf-8"))


def _summary_line(summary: dict) -> str:
    return json.dumps(
        {
            "files": summary["files"],
            "processed": summary["processed"],
            "skipped": summary["skipped"],
            "failed": summary["failed"],
            "rows_ok": summary["rows_ok"],
            "rows_failed": summary["rows_failed"],
        },
        default=str,
    )


def _run_by_source(db, source: str) -> IngestRun | None:
    return (
        db.query(IngestRun)
        .filter_by(source=source)
        .order_by(IngestRun.started_at.desc())
        .first()
    )


def _run_statuses(summary: dict) -> dict[str, str]:
    return {r["source"]: r["status"] for r in summary["runs"]}


def _qdrant_count() -> int:
    client = qdrant_client.get_client()
    return client.count(collection_name=settings.qdrant_collection, exact=True).count


def _neo4j_counts(store) -> tuple[int, int]:
    nodes = store.run_query("MATCH (n) RETURN count(n) AS n")[0]["n"]
    rels = store.run_query("MATCH ()-[r]->() RETURN count(r) AS r")[0]["r"]
    return nodes, rels


def _snapshot(db, store) -> dict:
    nodes, rels = _neo4j_counts(store)
    return {
        "fault_history": db.query(FaultHistory).count(),
        "maintenance_records": db.query(MaintenanceRecord).count(),
        "sensor_readings": db.query(SensorReading).count(),
        "ingest_runs": db.query(IngestRun).count(),
        "ingest_quarantine": db.query(IngestQuarantine).count(),
        "ontology_entities": db.query(OntologyEntity).count(),
        "ontology_relationships": db.query(OntologyRelationship).count(),
        # No "ontology_versions" total: the versions table is gone
        # (L1 state-only, design 2026-11-17 §3.2) — version history is
        # the state rows' ``version`` column, checked per subject by
        # _state_version_sum below.
        "projection_state": db.query(OntologyProjectionState).count(),
        "qdrant_points": _qdrant_count(),
        "neo4j_nodes": nodes,
        "neo4j_relationships": rels,
    }


def _state_version_sum(db, refs: list[str]) -> int:
    """Sum of the L1 state-row ``version`` column over ``refs``.

    Replaces the old ``_version_count`` (which counted
    ``ontology_versions`` rows): with L1 state-only there is no
    version table, so replay-safety is checked on the state rows
    themselves — a replay must not raise the sum, and after cleanup
    (state rows deleted) the sum is 0. ``refs`` are ``entity:`` /
    ``rel:`` subject refs (see ``ontology.canonical.types``).
    """
    if not refs:
        return 0
    total = 0
    for ref in refs:
        parts = ref.split(":")
        if parts[0] == "entity" and len(parts) == 3:
            row = (
                db.query(OntologyEntity)
                .filter_by(entity_type=parts[1], entity_key=parts[2])
                .first()
            )
        elif parts[0] == "rel" and len(parts) == 6:
            row = (
                db.query(OntologyRelationship)
                .filter_by(
                    relationship_type=parts[1],
                    source_type=parts[2],
                    source_key=parts[3],
                    target_type=parts[4],
                    target_key=parts[5],
                )
                .first()
            )
        else:
            continue
        if row is not None:
            total += row.version
    return total


def _version_count(db, refs: list[str]) -> int:
    """Backward-compatible name for the state-version sum."""
    return _state_version_sum(db, refs)


def _refs_for(cleanup: dict) -> list[str]:
    refs = [
        relationship_ref(*spec) for spec in cleanup["relations"]
    ]
    refs += [entity_ref(label, key) for label, key in cleanup["entities"]]
    return refs


def _run(cmd: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# --- environment -------------------------------------------------------------


def check_environment(report: Report, db) -> tuple[bool, object]:
    print("\n=== environment ===")
    ok = True

    try:
        db.execute(text("SELECT 1"))
        report.check(True, "Postgres reachable")
    except Exception as exc:  # noqa: BLE001
        report.check(False, "Postgres reachable", str(exc))
        ok = False

    try:
        client = qdrant_client.get_client()
        if client is None:
            raise RuntimeError("QDRANT_URL is empty")
        dim = get_collection_dim(client)
        if dim is None:
            raise RuntimeError(f"collection {settings.qdrant_collection!r} missing")
        report.check(True, "Qdrant collection reachable",
                     f"{settings.qdrant_collection} dim={dim}")
    except Exception as exc:  # noqa: BLE001
        report.check(False, "Qdrant collection reachable", str(exc))
        ok = False

    try:
        probed = qdrant_client.probe_dimension()
        report.check(probed > 0, "embedding endpoint returns a vector",
                     f"dim={probed}")
    except Exception as exc:  # noqa: BLE001
        report.check(False, "embedding endpoint returns a vector", str(exc))
        ok = False

    store = None
    try:
        store = get_graph_store()
        report.check(store.is_healthy(), "Neo4j reachable")
    except Exception as exc:  # noqa: BLE001
        report.check(False, "Neo4j reachable", str(exc))
        ok = False

    try:
        minio_client.ensure_bucket(settings.ingest_minio_bucket)
        report.check(True, "MinIO ingest-inbox bucket reachable",
                     settings.ingest_minio_bucket)
    except Exception as exc:  # noqa: BLE001
        report.check(False, "MinIO ingest-inbox bucket reachable", str(exc))
        ok = False

    missing = [
        f"{row_type}.json" for row_type in ROW_TYPES
        if not (SAMPLES / f"{row_type}.json").is_file()
    ]
    report.check(not missing, "sample file for every row type", ",".join(missing))

    if not ok:
        return False, store

    # Settle any projection rows left by another process before we take
    # the baseline, so "totals unchanged" means *we* changed nothing.
    try:
        settled = drain_pending_projections(db, limit=500)
        report.check(
            settled.get("errors", 0) == 0,
            "projection queue settled before baseline",
            json.dumps(settled, default=str),
        )
        report.check(
            settled.get("pending", 0) == 0,
            "projection queue empty before baseline",
            f"pending={settled.get('pending')}",
        )
    except Exception as exc:  # noqa: BLE001
        report.check(False, "projection queue settled before baseline", str(exc))
        return False, store

    snapshot = _snapshot(db, store)
    report.check(
        snapshot["fault_history"] > 0
        and snapshot["maintenance_records"] > 0
        and snapshot["sensor_readings"] > 0,
        "seed rows present in Postgres",
        json.dumps({k: snapshot[k] for k in (
            "fault_history", "maintenance_records", "sensor_readings")}),
    )
    report.check(
        snapshot["ontology_relationships"] > 0
        and snapshot["qdrant_points"] > 0
        and snapshot["neo4j_nodes"] > 0,
        "seeded ontology + Qdrant + Neo4j present",
        json.dumps({k: snapshot[k] for k in (
            "ontology_relationships", "qdrant_points", "neo4j_nodes")}),
    )
    return True, store


# --- scenarios ---------------------------------------------------------------


def build_rows(tag: str) -> dict:
    """The five sample files rewritten with tagged ids / links."""
    faults = _load_sample("faults")
    for i, row in enumerate(faults, start=1):
        row["id"] = f"manual-fh-{tag}-{i}"
    maintenance = _load_sample("maintenance")
    for i, row in enumerate(maintenance, start=1):
        row["id"] = f"manual-mr-{tag}-{i}"
    sensors = _load_sample("sensor_readings")
    for i, row in enumerate(sensors, start=1):
        row["id"] = f"manual-sr-{tag}-{i}"

    docs = [
        {
            "id": f"manual-doc-{tag}-linked",
            "title": f"Manual ingest linked doc {tag}",
            "content": f"Batch ingest probe for {ASSET_ID}: {uuid.uuid4()}",
            "category": "manual",
            "asset_id": ASSET_ID,
        },
        {
            "id": f"manual-doc-{tag}-plain",
            "title": f"Manual ingest plain doc {tag}",
            "content": f"Batch ingest probe with no asset link: {uuid.uuid4()}",
            "category": "manual",
        },
    ]

    component = f"manual-comp-{tag}"
    relations = [
        {
            "from_id": ASSET_ID,
            "from_label": "Asset",
            "relation": "HAS_COMPONENT",
            "to_id": component,
            "to_label": "Component",
        },
        {
            "from_id": component,
            "from_label": "Component",
            "relation": "PART_OF",
            "to_id": BUILDING,
            "to_label": "Building",
        },
    ]
    return {
        f"faults-manual-{tag}.json": faults,
        f"maintenance-manual-{tag}.json": maintenance,
        f"sensor_readings-manual-{tag}.json": sensors,
        f"documents-manual-{tag}.json": docs,
        f"relations-manual-{tag}.json": relations,
    }


def track_cleanup(cleanup: dict, tag: str, files: dict) -> None:
    cleanup["fault_ids"].update(r["id"] for r in files[f"faults-manual-{tag}.json"])
    cleanup["maint_ids"].update(
        r["id"] for r in files[f"maintenance-manual-{tag}.json"])
    cleanup["sensor_ids"].update(
        r["id"] for r in files[f"sensor_readings-manual-{tag}.json"])
    cleanup["doc_ids"].update(
        r["id"] for r in files[f"documents-manual-{tag}.json"])
    component = next(
        r["to_id"] for r in files[f"relations-manual-{tag}.json"]
        if r["to_label"] == "Component"
    )
    cleanup["entities"].add(("Component", component))
    cleanup["entities"].add(("Document", f"manual-doc-{tag}-linked"))
    for row in files[f"relations-manual-{tag}.json"]:
        cleanup["relations"].append((
            row["relation"], row["from_label"], row["from_id"],
            row["to_label"], row["to_id"],
        ))
    cleanup["relations"].append((
        "DOCUMENTED_BY", "Asset", ASSET_ID, "Document",
        f"manual-doc-{tag}-linked",
    ))
    cleanup["sources"].update(files)
    cleanup["tmpdirs"].add(cleanup["main_inbox"])


def scenario_dry_run(report: Report, db, inbox: str, files: dict) -> None:
    print("\n=== dry-run (all five row types, writes nothing) ===")
    source_before = {
        name: db.query(IngestRun).filter_by(source=name).count() for name in files
    }
    expected_rows = sum(len(rows) for rows in files.values())

    summary = run_batch(source=LocalDirSource(inbox), db=db, dry_run=True)
    report.check(
        summary["files"] == 5 and summary["processed"] == 5
        and not summary["failed"],
        "dry-run processed all five files", _summary_line(summary),
    )
    report.check(
        summary["rows_ok"] == expected_rows and summary["rows_failed"] == 0,
        f"dry-run validated all {expected_rows} rows",
        f"rows_ok={summary['rows_ok']} rows_failed={summary['rows_failed']}",
    )
    statuses = _run_statuses(summary)
    report.check(
        set(statuses.values()) == {"DRY_RUN"},
        "every dry-run file reports DRY_RUN", json.dumps(statuses),
    )
    report.check(
        all(db.query(IngestRun).filter_by(source=name).count()
            == source_before[name] for name in files),
        "dry-run created no ingest_runs rows", "watermark untouched",
    )
    report.check(
        db.query(IngestQuarantine).filter(
            IngestQuarantine.source.in_(list(files))).count() == 0,
        "dry-run created no quarantine rows", "0",
    )

    # Invalid inputs in dry-run: bad row, malformed JSON, non-array,
    # unknown type. Nothing may be recorded for any of them.
    tag = secrets.token_hex(3)
    bad_dir = tempfile.mkdtemp(prefix="manual-ingest-drybad-")
    try:
        _write_json(bad_dir, f"faults-manual-drybad-{tag}.json", [
            {"id": f"manual-dry-{tag}", "asset_id": ASSET_ID, "code": "F-1",
             "title": "ok", "severity": "low",
             "occurred_at": "2026-11-01T00:00:00"},
            {"id": f"manual-dry-{tag}-bad", "asset_id": "not allowed!",
             "code": "F-2", "title": "bad", "severity": "critical",
             "occurred_at": "not-a-date"},
        ])
        _write_text(bad_dir, f"faults-broken-{tag}.json", b"{not json")
        _write_text(bad_dir, f"faults-object-{tag}.json", b'{"rows": []}')
        _write_text(bad_dir, f"widgets-{tag}.json", b"[]")

        dry = run_batch(source=LocalDirSource(bad_dir), db=db, dry_run=True)
        statuses = _run_statuses(dry)
        report.check(
            statuses.get(f"faults-manual-drybad-{tag}.json") == "DRY_RUN"
            and dry["rows_failed"] == 1,
            "dry-run reports the invalid row without writing",
            json.dumps(statuses),
        )
        report.check(
            statuses.get(f"faults-broken-{tag}.json") == "FAILED"
            and statuses.get(f"faults-object-{tag}.json") == "FAILED"
            and statuses.get(f"widgets-{tag}.json") == "FAILED",
            "dry-run flags malformed / non-array / unknown files",
            json.dumps(statuses),
        )
        report.check(
            db.query(IngestRun).filter(
                IngestRun.source.like(f"%{tag}%")).count() == 0,
            "dry-run invalid inputs created no ingest_runs", "0",
        )
        report.check(
            db.query(IngestQuarantine).filter(
                IngestQuarantine.source.like(f"%{tag}%")).count() == 0,
            "dry-run invalid inputs created no quarantine rows", "0",
        )
    finally:
        shutil.rmtree(bad_dir, ignore_errors=True)


def scenario_batch(report: Report, db, store, inbox: str, files: dict,
                   tag: str) -> None:
    print("\n=== real batch: all five row types ===")
    expected_rows = sum(len(rows) for rows in files.values())
    summary = run_batch(source=LocalDirSource(inbox), db=db)
    report.check(
        summary["files"] == 5 and summary["processed"] == 5
        and not summary["failed"],
        "batch processed all five files", _summary_line(summary),
    )
    report.check(
        summary["rows_ok"] == expected_rows and summary["rows_failed"] == 0,
        f"all {expected_rows} rows loaded",
        f"rows_ok={summary['rows_ok']} rows_failed={summary['rows_failed']}",
    )
    statuses = _run_statuses(summary)
    report.check(
        set(statuses.values()) == {"SUCCEEDED"},
        "every file landed SUCCEEDED", json.dumps(statuses),
    )
    runs = {r["source"]: r for r in summary["runs"]}
    report.check(
        all(_run_by_source(db, name) is not None
            and _run_by_source(db, name).status == "SUCCEEDED"
            for name in files),
        "ingest_runs has one SUCCEEDED row per file", "5 rows",
    )
    report.check(
        all(runs[name]["rows_ok"] == len(rows) for name, rows in files.items()),
        "per-file rows_ok matches the file",
        json.dumps({n: runs[n]["rows_ok"] for n in files}),
    )

    # -- Postgres loaders --------------------------------------------------
    for label, model, rows in (
        ("fault_history", FaultHistory, files[f"faults-manual-{tag}.json"]),
        ("maintenance_records", MaintenanceRecord,
         files[f"maintenance-manual-{tag}.json"]),
        ("sensor_readings", SensorReading,
         files[f"sensor_readings-manual-{tag}.json"]),
    ):
        loaded = [db.get(model, row["id"]) for row in rows]
        report.check(
            all(row is not None for row in loaded),
            f"{label}: every ingested row is present",
            f"{sum(r is not None for r in loaded)}/{len(rows)}",
        )
        report.check(
            all(row is not None and row.asset_id == src["asset_id"]
                for row, src in zip(loaded, rows)),
            f"{label}: asset_id preserved",
            ",".join(sorted({r.asset_id for r in loaded if r})),
        )

    # -- Qdrant + derived links -------------------------------------------
    docs = files[f"documents-manual-{tag}.json"]
    linked, plain = docs[0]["id"], docs[1]["id"]
    client = qdrant_client.get_client()
    points = client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[_point_id(linked), _point_id(plain)],
        with_payload=True,
    )
    by_id = {str(p.id): (p.payload or {}) for p in points}
    report.check(
        len(by_id) == 2,
        "Qdrant holds both ingested document points", f"points={len(by_id)}",
    )
    report.check(
        by_id.get(_point_id(linked), {}).get("asset_id") == ASSET_ID,
        "linked doc payload carries asset_id", ASSET_ID,
    )
    report.check(
        not by_id.get(_point_id(plain), {}).get("asset_id"),
        "plain doc payload stays unlinked", str(
            by_id.get(_point_id(plain), {}).get("asset_id")),
    )

    # -- canonical relationships (explicit + derived) ---------------------
    for spec in [tuple(s) for s in cleanup_specs(files, tag)]:
        edge = db.query(OntologyRelationship).filter_by(
            relationship_type=spec[0], source_type=spec[1],
            source_key=spec[2], target_type=spec[3], target_key=spec[4],
        ).first()
        report.check(
            edge is not None,
            f"canonical edge {spec[0]} {spec[2]} → {spec[4]}", "present",
        )
    report.check(
        db.query(OntologyRelationship).filter_by(
            relationship_type="DOCUMENTED_BY", target_key=plain).count() == 0,
        "no invented DOCUMENTED_BY edge for the plain doc", "0",
    )
    report.check(
        db.query(OntologyEntity).filter_by(
            entity_type="Document", entity_key=linked).count() == 1,
        "canonical Document endpoint for the linked doc", linked,
    )

    # -- projection drain → Neo4j -----------------------------------------
    try:
        drained = drain_pending_projections(db, limit=500)
        drain_ok = drained.get("errors", 0) == 0 and drained.get("pending", 0) == 0
        drain_detail = json.dumps(drained, default=str)
    except Exception as exc:  # noqa: BLE001
        drain_ok, drain_detail = False, str(exc)
    report.check(drain_ok, "projection drain completes with an empty queue",
                 drain_detail)

    refs = _refs_for({
        "relations": [tuple(s) for s in cleanup_specs(files, tag)],
        "entities": [
            ("Component", f"manual-comp-{tag}"),
            ("Document", linked),
        ],
    })
    states = db.query(OntologyProjectionState).filter(
        OntologyProjectionState.subject_ref.in_(refs)).all()
    report.check(
        len(states) == len(refs)
        and all(s.status == "synced"
                and s.projected_version == s.canonical_version for s in states),
        "every new canonical subject is projected",
        f"{len(states)}/{len(refs)} synced",
    )

    try:
        component = f"manual-comp-{tag}"
        node = store.run_query(
            "MATCH (n:Component {id: $id}) RETURN n.id AS id", {"id": component})
        edges = store.run_query(
            "MATCH (a:Asset {id: $a})-[r:HAS_COMPONENT]->(c:Component {id: $c}) "
            "RETURN type(r) AS t",
            {"a": ASSET_ID, "c": component},
        )
        part_of = store.run_query(
            "MATCH (c:Component {id: $c})-[r:PART_OF]->(b:Building {id: $b}) "
            "RETURN type(r) AS t",
            {"c": component, "b": BUILDING},
        )
    except Exception as exc:  # noqa: BLE001
        node, edges, part_of = [], [], []
        graph_detail = str(exc)
    else:
        graph_detail = "present"
    report.check(bool(node), "Neo4j has the relation endpoint component",
                 graph_detail)
    report.check(bool(edges), "Neo4j has Asset —HAS_COMPONENT→ Component",
                 f"{len(edges)} edge(s)")
    report.check(bool(part_of), "Neo4j has Component —PART_OF→ Building",
                 f"{len(part_of)} edge(s)")

    try:
        neighbours = graph_query.neighbors("Asset", ASSET_ID, depth=1, limit=300)
        linked_docs = graph_query.linked_documents("Asset", ASSET_ID, limit=300)
    except Exception as exc:  # noqa: BLE001
        neighbours, linked_docs = [], []
        neigh_detail = str(exc)
    else:
        neigh_detail = f"{len(neighbours)} neighbours"
    report.check(
        any(n.get("type") == "HAS_COMPONENT"
            and n.get("to") == f"manual-comp-{tag}" for n in neighbours),
        "neighbors(O21) sees the ingested component", neigh_detail,
    )
    report.check(
        any(n.get("type") == "DOCUMENTED_BY" and n.get("to") == linked
            for n in neighbours),
        "neighbors(O21) sees the ingested linked doc", neigh_detail,
    )
    report.check(
        any(d.get("document_id") == linked for d in linked_docs),
        "linked_documents(O21) sees the ingested doc", f"{len(linked_docs)} docs",
    )
    report.check(
        not any(d.get("document_id") == plain for d in linked_docs),
        "linked_documents(O21) excludes the plain doc", "absent",
    )


def cleanup_specs(files: dict, tag: str) -> list[tuple]:
    specs = [
        (
            row["relation"], row["from_label"], row["from_id"],
            row["to_label"], row["to_id"],
        )
        for row in files[f"relations-manual-{tag}.json"]
    ]
    specs.append((
        "DOCUMENTED_BY", "Asset", ASSET_ID, "Document",
        f"manual-doc-{tag}-linked",
    ))
    return specs


def scenario_watermark_force(report: Report, db, inbox: str, files: dict,
                             tag: str) -> None:
    print("\n=== watermark + forced replay ===")
    fault_ids = [r["id"] for r in files[f"faults-manual-{tag}.json"]]
    sensor_ids = [r["id"] for r in files[f"sensor_readings-manual-{tag}.json"]]
    maint_ids = [r["id"] for r in files[f"maintenance-manual-{tag}.json"]]
    docs = [r["id"] for r in files[f"documents-manual-{tag}.json"]]
    specs = cleanup_specs(files, tag)
    refs = [
        relationship_ref(*spec) for spec in specs
    ] + [
        entity_ref("Component", f"manual-comp-{tag}"),
        entity_ref("Document", docs[0]),
    ]
    before = {
        "faults": db.query(FaultHistory).filter(
            FaultHistory.id.in_(fault_ids)).count(),
        "maintenance": db.query(MaintenanceRecord).filter(
            MaintenanceRecord.id.in_(maint_ids)).count(),
        "sensors": db.query(SensorReading).filter(
            SensorReading.id.in_(sensor_ids)).count(),
        "edges": db.query(OntologyRelationship).filter(
            OntologyRelationship.source_key.in_(
                [s[2] for s in specs])).count(),
        "versions": _version_count(db, refs),
        "qdrant": _qdrant_count(),
    }

    skipped = run_batch(source=LocalDirSource(inbox), db=db)
    report.check(
        skipped["processed"] == 0
        and set(skipped["skipped"]) == set(files)
        and skipped["rows_ok"] == 0,
        "re-run without force skips every SUCCEEDED file",
        _summary_line(skipped),
    )
    after_skip = {
        "faults": db.query(FaultHistory).filter(
            FaultHistory.id.in_(fault_ids)).count(),
        "maintenance": db.query(MaintenanceRecord).filter(
            MaintenanceRecord.id.in_(maint_ids)).count(),
        "sensors": db.query(SensorReading).filter(
            SensorReading.id.in_(sensor_ids)).count(),
        "edges": db.query(OntologyRelationship).filter(
            OntologyRelationship.source_key.in_(
                [s[2] for s in specs])).count(),
        "versions": _version_count(db, refs),
        "qdrant": _qdrant_count(),
    }
    report.check(
        after_skip == before,
        "watermark skip wrote nothing and changed no total",
        f"{before} == {after_skip}",
    )

    forced = run_batch(source=LocalDirSource(inbox), db=db, force=True)
    expected_rows = sum(len(rows) for rows in files.values())
    report.check(
        forced["processed"] == 5 and forced["rows_ok"] == expected_rows
        and forced["rows_failed"] == 0 and not forced["failed"],
        "forced re-run replays every row", _summary_line(forced),
    )
    after_force = {
        "faults": db.query(FaultHistory).filter(
            FaultHistory.id.in_(fault_ids)).count(),
        "maintenance": db.query(MaintenanceRecord).filter(
            MaintenanceRecord.id.in_(maint_ids)).count(),
        "sensors": db.query(SensorReading).filter(
            SensorReading.id.in_(sensor_ids)).count(),
        "edges": db.query(OntologyRelationship).filter(
            OntologyRelationship.source_key.in_(
                [s[2] for s in specs])).count(),
        "versions": _version_count(db, refs),
        "qdrant": _qdrant_count(),
    }
    report.check(
        after_force == before,
        "forced replay does not duplicate rows, edges, versions or points",
        f"before={before} after={after_force}",
    )

    try:
        drained = drain_pending_projections(db, limit=500)
        replay_ok = drained.get("errors", 0) == 0
        replay_detail = json.dumps(drained, default=str)
    except Exception as exc:  # noqa: BLE001
        replay_ok, replay_detail = False, str(exc)
    report.check(replay_ok, "drain after forced replay stays error-free",
                 replay_detail)


def scenario_partial_retry(report: Report, db, cleanup: dict, tag: str) -> None:
    print("\n=== PARTIAL + quarantine fix + retry ===")
    source = f"faults-manual-partial-{tag}.json"
    ok_id = f"manual-fh-{tag}-partial-ok"
    bad_id = f"manual-fh-{tag}-partial-bad"
    cleanup["sources"].add(source)
    cleanup["fault_ids"].update({ok_id, bad_id})
    inbox = tempfile.mkdtemp(prefix="manual-ingest-partial-")
    cleanup["tmpdirs"].add(inbox)
    _write_json(inbox, source, [
        {"id": ok_id, "asset_id": ASSET_ID, "code": "F-PART",
         "title": "Valid partial row", "severity": "low",
         "occurred_at": "2026-11-05T08:00:00"},
        {"id": bad_id, "asset_id": ASSET_ID, "code": "F-BAD",
         "title": "Invalid partial row", "severity": "critical",
         "occurred_at": "2026-11-05T09:00:00"},
    ])

    summary = run_batch(source=LocalDirSource(inbox), db=db)
    run = _run_by_source(db, source)
    report.check(
        summary["processed"] == 1 and summary["rows_ok"] == 1
        and summary["rows_failed"] == 1,
        "one good row loads, one bad row quarantines",
        _summary_line(summary),
    )
    report.check(
        run is not None and run.status == "PARTIAL"
        and run.rows_ok == 1 and run.rows_failed == 1,
        "ingest_runs records PARTIAL 1/1",
        f"status={getattr(run, 'status', None)} "
        f"ok={getattr(run, 'rows_ok', None)} "
        f"failed={getattr(run, 'rows_failed', None)}",
    )
    report.check(
        db.get(FaultHistory, ok_id) is not None,
        "the valid row is in Postgres", ok_id,
    )
    pending = db.query(IngestQuarantine).filter_by(
        source=source, status="pending").all()
    report.check(
        len(pending) == 1 and pending[0].row.get("id") == bad_id,
        "the invalid row is quarantined once",
        f"pending={len(pending)}",
    )
    report.check(
        pending and "severity" in (pending[0].error or ""),
        "quarantine error names the bad field",
        (pending[0].error or "")[:120] if pending else "",
    )

    again = run_batch(source=LocalDirSource(inbox), db=db)
    report.check(
        again["processed"] == 1 and again["rows_ok"] == 1
        and again["rows_failed"] == 1,
        "PARTIAL files are re-processed (not watermark-skipped)",
        _summary_line(again),
    )
    pending = db.query(IngestQuarantine).filter_by(
        source=source, status="pending").all()
    report.check(
        len(pending) == 2,
        "the re-run quarantines the bad row again",
        f"pending={len(pending)}",
    )

    for row in pending:
        fixed = dict(row.row)
        fixed["severity"] = "medium"
        row.row = fixed
    db.commit()
    report.check(True, "fixed the quarantined rows in place",
                 f"{len(pending)} row(s) severity=medium")

    retried = retry_quarantine(db, limit=50)
    remaining = db.query(IngestQuarantine).filter_by(
        source=source, status="pending").count()
    report.check(
        retried["retried"] >= 2,
        "retry_quarantine replays the fixed rows",
        json.dumps(retried, default=str),
    )
    report.check(
        remaining == 0,
        "no pending quarantine row left for this file", f"pending={remaining}",
    )
    report.check(
        db.get(FaultHistory, bad_id) is not None,
        "the fixed row is now in Postgres", bad_id,
    )
    statuses = {
        q.status for q in db.query(IngestQuarantine).filter_by(source=source)
    }
    report.check(
        statuses == {"retried"},
        "quarantine rows are marked retried", json.dumps(sorted(statuses)),
    )


def scenario_bad_files(report: Report, db, cleanup: dict, tag: str) -> None:
    print("\n=== file-level failures ===")
    broken = f"faults-broken-{tag}.json"
    nonarray = f"faults-object-{tag}.json"
    unknown = f"widgets-{tag}.json"
    cleanup["sources"].update({broken, nonarray, unknown})
    inbox = tempfile.mkdtemp(prefix="manual-ingest-bad-")
    cleanup["tmpdirs"].add(inbox)
    _write_text(inbox, broken, b"{not json")
    _write_text(inbox, nonarray, b'{"rows": []}')
    _write_text(inbox, unknown, b"[]")

    summary = run_batch(source=LocalDirSource(inbox), db=db)
    report.check(
        set(summary["failed"]) == {broken, nonarray, unknown}
        and set(_run_statuses(summary).values()) == {"FAILED"},
        "all three bad files report FAILED", _summary_line(summary),
    )
    runs = {r.source: r for r in db.query(IngestRun).filter(
        IngestRun.source.in_([broken, nonarray, unknown])).all()}
    report.check(
        len(runs) == 3 and all(r.status == "FAILED" for r in runs.values()),
        "ingest_runs records FAILED for every bad file",
        json.dumps({s: r.status for s, r in runs.items()}),
    )
    report.check(
        all(r.errors for r in runs.values()),
        "each FAILED run stores a file-level error",
        json.dumps({s: (r.errors or [""])[0][:80] for s, r in runs.items()}),
    )
    report.check(
        db.query(IngestQuarantine).filter(
            IngestQuarantine.source.in_([broken, nonarray, unknown])
        ).count() == 0,
        "file-level failures create no quarantine rows", "0",
    )


def scenario_minio(report: Report, db, cleanup: dict, tag: str) -> None:
    print("\n=== MinIO inbox source ===")
    bucket = settings.ingest_minio_bucket
    name = f"sensor_readings-manual-minio-{tag}.json"
    row_id = f"manual-sr-{tag}-minio"
    cleanup["sources"].add(name)
    cleanup["objects"].add((bucket, name))
    cleanup["sensor_ids"].add(row_id)
    body = json.dumps([{
        "id": row_id, "asset_id": ASSET_ID, "metric": "vibration_mm_s",
        "unit": "mm/s", "value": 2.4, "at": "2026-11-06T10:00:00",
    }]).encode()

    client = minio_client.get_client()
    client.put_object(bucket, name, io.BytesIO(body), len(body),
                      content_type="application/json")

    summary = run_batch(source=MinioInboxSource(bucket), db=db)
    statuses = _run_statuses(summary)
    report.check(
        statuses.get(name) == "SUCCEEDED",
        "MinIO inbox object is ingested",
        json.dumps({k: v for k, v in statuses.items() if k == name}),
    )
    report.check(
        db.get(SensorReading, row_id) is not None,
        "the MinIO row is in Postgres", row_id,
    )
    objects = {o.object_name for o in client.list_objects(bucket, recursive=True)}
    report.check(
        name in objects,
        "the object stays in the bucket after ingest (watermark owns re-runs)",
        name,
    )

    again = run_batch(source=MinioInboxSource(bucket), db=db)
    report.check(
        name in again["skipped"] and again["processed"] == 0,
        "watermark skips the MinIO object on the next pass",
        _summary_line(again),
    )


def scenario_service(report: Report, db, cleanup: dict, tag: str) -> None:
    print("\n=== ingest_service wrapper (beat task path) ===")
    name = f"sensor_readings-service-{tag}.json"
    row_id = f"manual-sr-{tag}-service"
    cleanup["sources"].add(name)
    cleanup["sensor_ids"].add(row_id)
    inbox = tempfile.mkdtemp(prefix="manual-ingest-service-")
    cleanup["tmpdirs"].add(inbox)
    _write_json(inbox, name, [{
        "id": row_id, "asset_id": ASSET_ID, "metric": "motor_temperature_c",
        "unit": "C", "value": 68.5, "at": "2026-11-06T11:00:00",
    }])

    from app.services import ingest_service

    summary = ingest_service.run_ingest_batch(db, inbox=inbox)
    report.check(
        summary["processed"] == 1 and summary["rows_ok"] == 1
        and summary["rows_failed"] == 0,
        "run_ingest_batch(inbox=...) loads the file", _summary_line(summary),
    )
    report.check(
        db.get(SensorReading, row_id) is not None,
        "the service-loaded row is in Postgres", row_id,
    )
    again = ingest_service.run_ingest_batch(db, inbox=inbox)
    report.check(
        name in again["skipped"] and again["processed"] == 0,
        "a second service pass is watermark-skipped", _summary_line(again),
    )


def scenario_worker(report: Report, db, cleanup: dict, tag: str) -> None:
    print("\n=== worker container: app.worker.tasks.run_ingest_batch ===")
    name = f"sensor_readings-worker-{tag}.json"
    row_id = f"manual-sr-{tag}-worker"
    cleanup["sources"].add(name)
    cleanup["sensor_ids"].add(row_id)

    try:
        ps = _run(["docker", "compose", "ps", "-q", "worker"], timeout=30)
    except Exception as exc:  # noqa: BLE001
        report.skip("worker container task", f"docker unavailable: {exc}")
        return
    if ps.returncode != 0 or not ps.stdout.strip():
        report.skip("worker container task", "worker container is not running")
        return

    env = _run([
        "docker", "compose", "exec", "-T", "worker", "python", "-c",
        "import json,os\n"
        "from app.config import settings\n"
        "print('ENV::'+json.dumps({'inbox': os.path.abspath("
        "settings.ingest_inbox_dir), 'minio': settings.ingest_use_minio}))",
    ])
    env_line = next(
        (ln for ln in env.stdout.splitlines() if ln.startswith("ENV::")), "")
    if env.returncode != 0 or not env_line:
        report.check(False, "worker exposes its inbox config",
                     (env.stderr or env.stdout)[-300:])
        return
    worker_env = json.loads(env_line[len("ENV::"):])
    if worker_env.get("minio"):
        report.skip("worker container task",
                    "worker reads the MinIO inbox (INGEST_USE_MINIO=true)")
        return
    remote_dir = worker_env["inbox"]
    remote = f"{remote_dir}/{name}"

    local = Path(tempfile.mkdtemp(prefix="manual-ingest-worker-")) / name
    cleanup["tmpdirs"].add(str(local.parent))
    _write_json(local.parent, name, [{
        "id": row_id, "asset_id": ASSET_ID, "metric": "door_cycles",
        "unit": "count", "value": 12, "at": "2026-11-06T12:00:00",
    }])

    def invoke() -> dict | None:
        proc = _run([
            "docker", "compose", "exec", "-T", "worker", "python", "-c",
            "import json\n"
            "from app.worker.tasks import run_ingest_batch\n"
            "print('SUMMARY::'+json.dumps(run_ingest_batch.apply().get()))",
        ])
        line = next(
            (ln for ln in proc.stdout.splitlines()
             if ln.startswith("SUMMARY::")), "")
        if proc.returncode != 0 or not line:
            report.check(False, "worker task returns a summary",
                         (proc.stderr or proc.stdout)[-400:])
            return None
        return json.loads(line[len("SUMMARY::"):])

    try:
        mk = _run(["docker", "compose", "exec", "-T", "worker",
                   "mkdir", "-p", remote_dir])
        if mk.returncode != 0:
            report.check(False, "worker inbox directory exists",
                         mk.stderr[-300:])
            return
        cp = _run(["docker", "compose", "cp", str(local),
                   f"worker:{remote}"])
        report.check(cp.returncode == 0, "copied the file into the worker inbox",
                     (cp.stderr or remote)[-200:])
        if cp.returncode != 0:
            return

        summary = invoke()
        if summary is None:
            return
        statuses = _run_statuses(summary)
        report.check(
            statuses.get(name) == "SUCCEEDED" and summary["rows_ok"] == 1,
            "worker task ingests the inbox file", _summary_line(summary),
        )
        db.expire_all()
        report.check(
            db.get(SensorReading, row_id) is not None,
            "the worker-loaded row is in Postgres", row_id,
        )

        again = invoke()
        if again is None:
            return
        report.check(
            name in again["skipped"] and again["processed"] == 0,
            "a second worker pass is watermark-skipped", _summary_line(again),
        )
    finally:
        cleanup["worker_files"].add(remote)


# --- cleanup -----------------------------------------------------------------


def cleanup_all(report: Report, db, store, cleanup: dict) -> None:
    print("\n=== cleanup ===")
    errors: list[str] = []

    for remote in cleanup["worker_files"]:
        try:
            _run(["docker", "compose", "exec", "-T", "worker",
                  "rm", "-f", remote], timeout=30)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"worker file {remote}: {exc}")

    for bucket, name in cleanup["objects"]:
        try:
            minio_client.get_client().remove_object(bucket, name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"minio {bucket}/{name}: {exc}")

    doc_ids = sorted(cleanup["doc_ids"])
    if doc_ids:
        try:
            from qdrant_client.http import models as qm

            qdrant_client.get_client().delete(
                collection_name=settings.qdrant_collection,
                points_selector=qm.PointIdsList(
                    points=[_point_id(d) for d in doc_ids]),
                wait=True,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"qdrant points: {exc}")

    rels = list(cleanup["relations"])
    if rels:
        try:
            for spec in rels:
                db.query(OntologyRelationship).filter_by(
                    relationship_type=spec[0], source_type=spec[1],
                    source_key=spec[2], target_type=spec[3],
                    target_key=spec[4],
                ).delete(synchronize_session=False)
            refs = [
                relationship_ref(*spec) for spec in rels
            ] + [entity_ref(*e) for e in cleanup["entities"]]
            db.query(OntologyProjectionState).filter(
                OntologyProjectionState.subject_ref.in_(refs)
            ).delete(synchronize_session=False)
            # No OntologyVersion delete: the versions table is gone
            # (L1 state-only, design 2026-11-17 §3.2).
            for entity_type, entity_key in cleanup["entities"]:
                db.query(OntologyEntity).filter_by(
                    entity_type=entity_type, entity_key=entity_key,
                ).delete(synchronize_session=False)
            db.commit()
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            errors.append(f"canonical rows: {exc}")

        try:
            for spec in rels:
                store.delete_relationship(
                    spec[1], spec[2], spec[0], spec[3], spec[4])
            for entity_type, entity_key in cleanup["entities"]:
                store.delete_node(entity_type, entity_key)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"neo4j: {exc}")

    try:
        for model, ids in (
            (FaultHistory, cleanup["fault_ids"]),
            (MaintenanceRecord, cleanup["maint_ids"]),
            (SensorReading, cleanup["sensor_ids"]),
        ):
            if ids:
                db.query(model).filter(model.id.in_(list(ids))).delete(
                    synchronize_session=False)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        errors.append(f"postgres rows: {exc}")

    sources = list(cleanup["sources"])
    if sources:
        try:
            db.query(IngestQuarantine).filter(
                IngestQuarantine.source.in_(sources)
            ).delete(synchronize_session=False)
            db.query(IngestRun).filter(
                IngestRun.source.in_(sources)
            ).delete(synchronize_session=False)
            db.commit()
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            errors.append(f"ingest rows: {exc}")

    for tmpdir in cleanup["tmpdirs"]:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if errors:
        report.warn("cleanup left residue", "; ".join(errors)[:900])
    else:
        print("  cleanup: removed rows, objects, points, canonical + Neo4j nodes, "
              "ingest runs")


def verify_clean(report: Report, db, store, cleanup: dict,
                 baseline: dict) -> None:
    print("\n=== cleanup verification ===")
    fault_ids = list(cleanup["fault_ids"])
    maint_ids = list(cleanup["maint_ids"])
    sensor_ids = list(cleanup["sensor_ids"])
    doc_ids = list(cleanup["doc_ids"])
    sources = list(cleanup["sources"])
    rels = list(cleanup["relations"])
    refs = _refs_for(cleanup)

    report.check(
        db.query(FaultHistory).filter(FaultHistory.id.in_(fault_ids)).count() == 0
        if fault_ids else True,
        "no ingested fault rows left", f"{len(fault_ids)} checked",
    )
    report.check(
        db.query(MaintenanceRecord).filter(
            MaintenanceRecord.id.in_(maint_ids)).count() == 0
        if maint_ids else True,
        "no ingested maintenance rows left", f"{len(maint_ids)} checked",
    )
    report.check(
        db.query(SensorReading).filter(
            SensorReading.id.in_(sensor_ids)).count() == 0
        if sensor_ids else True,
        "no ingested sensor rows left", f"{len(sensor_ids)} checked",
    )
    report.check(
        db.query(IngestRun).filter(IngestRun.source.in_(sources)).count() == 0
        if sources else True,
        "no ingest_runs rows left", f"{len(sources)} files checked",
    )
    report.check(
        db.query(IngestQuarantine).filter(
            IngestQuarantine.source.in_(sources)).count() == 0
        if sources else True,
        "no quarantine rows left", "0",
    )
    edge_left = sum(
        db.query(OntologyRelationship).filter_by(
            relationship_type=spec[0], source_type=spec[1],
            source_key=spec[2], target_type=spec[3], target_key=spec[4],
        ).count()
        for spec in rels
    )
    report.check(
        edge_left == 0,
        "no ingested canonical edges left", f"{edge_left} of {len(rels)} found",
    )
    report.check(
        db.query(OntologyEntity).filter(
            OntologyEntity.entity_key.in_([e[1] for e in cleanup["entities"]])
        ).count() == 0 if cleanup["entities"] else True,
        "no ingested canonical endpoints left",
        f"{len(cleanup['entities'])} checked",
    )
    report.check(
        _version_count(db, refs) == 0,
        "no ontology versions left for ingested subjects", f"{len(refs)} refs",
    )
    report.check(
        db.query(OntologyProjectionState).filter(
            OntologyProjectionState.subject_ref.in_(refs)).count() == 0,
        "no projection-state rows left for ingested subjects", f"{len(refs)} refs",
    )
    if doc_ids:
        points = qdrant_client.get_client().retrieve(
            collection_name=settings.qdrant_collection,
            ids=[_point_id(d) for d in doc_ids], with_payload=False,
        )
        report.check(not points, "no ingested Qdrant points left",
                     f"{len(points)} found")
    try:
        component = next(
            (key for label, key in cleanup["entities"] if label == "Component"),
            None)
        leftover = []
        if component:
            leftover = store.run_query(
                "MATCH (n:Component {id: $id}) RETURN n.id AS id",
                {"id": component},
            )
        doc_left = store.run_query(
            "MATCH (n:Document) WHERE n.id IN $ids RETURN n.id AS id",
            {"ids": doc_ids},
        ) if doc_ids else []
        report.check(
            not leftover and not doc_left,
            "no ingested Neo4j nodes left",
            json.dumps([r["id"] for r in leftover + doc_left]),
        )
        edge_left = 0
        for spec in rels:
            rows = store.run_query(
                "MATCH (a {id: $a})-[r]->(b {id: $b}) "
                "WHERE type(r) = $t RETURN count(r) AS n",
                {"a": spec[2], "b": spec[4], "t": spec[0]},
            )
            edge_left += rows[0]["n"]
        report.check(
            edge_left == 0,
            "no ingested Neo4j edges left", f"{edge_left} found",
        )
    except Exception as exc:  # noqa: BLE001
        report.check(False, "no ingested Neo4j nodes left", str(exc))

    try:
        current = _snapshot(db, store)
        report.check(
            current == baseline,
            "stack totals match the pre-run baseline",
            f"baseline={baseline} current={current}",
        )
    except Exception as exc:  # noqa: BLE001
        report.check(False, "stack totals match the pre-run baseline", str(exc))


# --- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true",
                        help="keep the rows/objects/points this run creates")
    parser.add_argument("--quiet", action="store_true",
                        help="disable HTTP request/response logging")
    args = parser.parse_args(argv)
    if args.quiet:
        set_quiet(True)

    report = Report()
    tag = secrets.token_hex(4)
    cleanup: dict = {
        "tag": tag,
        "fault_ids": set(),
        "maint_ids": set(),
        "sensor_ids": set(),
        "doc_ids": set(),
        "sources": set(),
        "objects": set(),
        "relations": [],
        "entities": set(),
        "tmpdirs": set(),
        "worker_files": set(),
    }
    db = SessionLocal()
    store = None
    baseline: dict | None = None
    try:
        ready, store = check_environment(report, db)
        if not ready:
            return 1
        baseline = _snapshot(db, store)

        files = build_rows(tag)
        inbox = tempfile.mkdtemp(prefix=f"manual-ingest-{tag}-")
        cleanup["main_inbox"] = inbox
        for name, rows in files.items():
            _write_json(inbox, name, rows)
        track_cleanup(cleanup, tag, files)

        scenario_dry_run(report, db, inbox, files)
        scenario_batch(report, db, store, inbox, files, tag)
        scenario_watermark_force(report, db, inbox, files, tag)
        scenario_partial_retry(report, db, cleanup, tag)
        scenario_bad_files(report, db, cleanup, tag)
        scenario_minio(report, db, cleanup, tag)
        scenario_service(report, db, cleanup, tag)
        scenario_worker(report, db, cleanup, tag)
    except Exception as exc:  # noqa: BLE001
        report.check(False, "scenario execution completed", f"{type(exc).__name__}: {exc}")
    finally:
        if args.keep:
            print("\n=== cleanup skipped (--keep) ===")
        else:
            if store is not None and baseline is not None:
                cleanup_all(report, db, store, cleanup)
                verify_clean(report, db, store, cleanup, baseline)
            else:
                cleanup_all(report, db, store, cleanup)
        db.close()

    print("\nchecks: " + (
        f"PASS — {report.passed} passed, {report.skipped} skipped, "
        f"{len(report.warnings)} warning(s), {len(report.failures)} failure(s)"
        if not report.failures else
        f"FAIL — {report.passed} passed, {report.skipped} skipped, "
        f"{len(report.warnings)} warning(s), {len(report.failures)} failure(s)"
    ))
    if report.failures:
        for failure in report.failures:
            print(f"  FAILED: {failure}")
        print("FAIL: ingest batch")
        return 1
    print("PASS: ingest batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
