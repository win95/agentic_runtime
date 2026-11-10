"""Batch ingest (Option B) tests — no live stores needed.

Postgres loaders run against the in-memory SQLite DB from conftest;
Qdrant/FalkorDB loaders are monkeypatched. The beat schedule is
asserted the same way test_schedules.py asserts the scheduler entry.
"""

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.db import SessionLocal
from app.models import (
    FaultHistory,
    IngestQuarantine,
    IngestRun,
    SensorReading,
)
from app.services import ingest_service
from app.worker import tasks  # noqa: F401 — importing registers the Celery tasks
from app.worker.celery_app import celery_app
from ingest import run as ingest_run
from ingest.loaders import load_faults, load_sensor_readings
from ingest.schemas import FaultRow, RelationRow, SensorRow
from ingest.sources import LocalDirSource

SAMPLES = Path(__file__).resolve().parent.parent / "ingest" / "samples"


def _copy_samples(tmp_path) -> Path:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for sample in SAMPLES.glob("*.json"):
        shutil.copy(sample, inbox / sample.name)
    return inbox


def _mock_vector_and_graph(monkeypatch):
    monkeypatch.setattr(
        "infra.qdrant_client.probe_dimension", lambda embeddings=None: 4
    )
    monkeypatch.setattr(
        "infra.qdrant_client.ensure_collection", lambda dim, **kw: "technical_documents"
    )
    monkeypatch.setattr(
        "infra.qdrant_client.upsert_documents",
        lambda docs, embeddings=None, **kw: {"upserted": len(docs), "skipped": 0},
    )
    monkeypatch.setattr(
        "infra.falkor_client.merge_asset_relations", lambda relations: None
    )


def test_schema_rejects_bad_asset_id():
    with pytest.raises(ValidationError):
        FaultRow.model_validate({
            "id": "x1", "asset_id": "O21; DROP TABLE", "code": "F-101",
            "title": "t", "occurred_at": "2026-11-01T00:00:00",
        })


def test_schema_rejects_oversized_title_and_bad_dates():
    with pytest.raises(ValidationError):
        FaultRow.model_validate({
            "id": "x1", "asset_id": "O21", "code": "F-101",
            "title": "x" * 501, "occurred_at": "2026-11-01T00:00:00",
        })
    with pytest.raises(ValidationError):
        FaultRow.model_validate({
            "id": "x1", "asset_id": "O21", "code": "F-101", "title": "t",
            "occurred_at": "2026-11-02T00:00:00",
            "resolved_at": "2026-11-01T00:00:00",
        })
    with pytest.raises(ValidationError):
        RelationRow.model_validate({
            "from_id": "O21", "from_label": "Asset", "relation": "has part",
            "to_id": "m1", "to_label": "Component",
        })
    with pytest.raises(ValidationError):
        SensorRow.model_validate({
            "id": "s1", "asset_id": "O21", "metric": "MotorTemp",
            "value": 70.0, "at": "2026-11-01T12:00:00",
        })


def test_file_type_from_name():
    assert ingest_run.file_type("faults.json") == "faults"
    assert ingest_run.file_type("faults-2026-11.json") == "faults"
    assert ingest_run.file_type("sensor_readings.json") == "sensor_readings"
    assert ingest_run.file_type("weird.json") is None


def test_postgres_loaders_idempotent(db_session):
    db = SessionLocal()
    try:
        rows = [FaultRow.model_validate({
            "id": "fh-t1", "asset_id": "O21", "code": "F-101",
            "title": "t", "occurred_at": "2026-11-01T00:00:00",
        })]
        assert load_faults(db, rows) == (1, [])
        assert load_faults(db, rows) == (1, [])
        assert db.query(FaultHistory).count() == 1
        sensors = [SensorRow.model_validate({
            "id": "sr-t1", "asset_id": "O21", "metric": "motor_temperature_c",
            "value": 71.0, "unit": "C", "at": "2026-11-01T12:00:00",
        })]
        assert load_sensor_readings(db, sensors) == (1, [])
        assert load_sensor_readings(db, sensors) == (1, [])
        assert db.query(SensorReading).count() == 1
    finally:
        db.close()


def test_run_batch_samples_then_skip_then_force(db_session, tmp_path, monkeypatch):
    _mock_vector_and_graph(monkeypatch)
    inbox = _copy_samples(tmp_path)
    db = SessionLocal()
    try:
        first = ingest_run.run_batch(source=LocalDirSource(inbox), db=db)
        assert first["files"] == 5
        assert first["rows_ok"] == 7  # 2 faults + 1 maint + 2 sensors + 1 doc + 1 rel
        assert first["rows_failed"] == 0
        assert db.query(FaultHistory).count() == 2
        assert db.query(SensorReading).count() == 2
        assert db.query(IngestRun).count() == 5

        second = ingest_run.run_batch(source=LocalDirSource(inbox), db=db)
        assert second["rows_ok"] == 0
        assert len(second["skipped"]) == 5

        forced = ingest_run.run_batch(source=LocalDirSource(inbox), db=db, force=True)
        assert forced["rows_ok"] == 7
        assert db.query(FaultHistory).count() == 2  # upsert, no duplicates
    finally:
        db.close()


def test_run_batch_quarantines_bad_rows_and_bad_files(db_session, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "faults.json").write_text(json.dumps([
        {"id": "fh-ok", "asset_id": "O21", "code": "F-101", "title": "ok",
         "occurred_at": "2026-11-01T00:00:00"},
        {"id": "fh-bad", "asset_id": "O21; DROP", "code": "F-101", "title": "bad",
         "occurred_at": "2026-11-01T00:00:00"},
        "not-a-dict",
    ]))
    (inbox / "broken.json").write_text("{not json")
    (inbox / "weird.json").write_text("[]")
    db = SessionLocal()
    try:
        summary = ingest_run.run_batch(source=LocalDirSource(inbox), db=db)
        assert summary["rows_ok"] == 1
        assert summary["rows_failed"] == 2
        assert summary["failed"] == ["broken.json", "weird.json"]
        assert db.query(IngestQuarantine).count() == 2
        run = db.query(IngestRun).filter_by(source="faults.json").one()
        assert run.status == "PARTIAL"
        assert run.rows_ok == 1 and run.rows_failed == 2
    finally:
        db.close()


def test_run_batch_dry_run_writes_nothing(db_session, tmp_path, monkeypatch):
    _mock_vector_and_graph(monkeypatch)
    inbox = _copy_samples(tmp_path)
    db = SessionLocal()
    try:
        summary = ingest_run.run_batch(source=LocalDirSource(inbox), db=db,
                                       dry_run=True)
        assert summary["rows_ok"] == 7
        assert db.query(IngestRun).count() == 0
        assert db.query(FaultHistory).count() == 0
    finally:
        db.close()


def test_retry_quarantine(db_session):
    db = SessionLocal()
    try:
        run = IngestRun(id="ing_retrytest", source="sensor_readings.json",
                        status="PARTIAL", rows_ok=0, rows_failed=1, errors=[],
                        started_at=ingest_run._utcnow())
        db.add(run)
        db.add(IngestQuarantine(
            id="inq_retrytest", run_id=run.id, source="sensor_readings.json",
            row={"id": "sr-r1", "asset_id": "O21", "metric": "motor_temperature_c",
                 "value": 70.0, "unit": "C", "at": "2026-11-02T12:00:00"},
            error="simulated load failure", status="pending"))
        db.add(IngestQuarantine(
            id="inq_badtest", run_id=run.id, source="sensor_readings.json",
            row={"id": "sr-r2", "asset_id": "BAD ID!", "metric": "x",
                 "value": 1.0, "at": "2026-11-02T12:00:00"},
            error="bad asset", status="pending"))
        db.commit()
        result = ingest_run.retry_quarantine(db)
        assert result == {"retried": 1, "still_pending": 1}
        assert db.get(IngestQuarantine, "inq_retrytest").status == "retried"
        assert db.get(IngestQuarantine, "inq_badtest").status == "pending"
        assert db.query(SensorReading).count() == 1
    finally:
        db.close()


def test_ingest_service_uses_inbox(db_session, tmp_path, monkeypatch):
    _mock_vector_and_graph(monkeypatch)
    inbox = _copy_samples(tmp_path)
    db = SessionLocal()
    try:
        summary = ingest_service.run_ingest_batch(db, inbox=str(inbox))
        assert summary["rows_ok"] == 7
        assert summary["failed"] == []
    finally:
        db.close()


def test_sensor_repository_reads_real_rows(db_session):
    from infra import repositories
    from seeds import demo_data

    db = SessionLocal()
    try:
        demo_data.seed_postgres_demo(db)
        rows, is_stub = repositories.get_sensor_data("O21", "2026-10-07", "2026-10-07")
        assert is_stub is False
        assert len(rows) == 2
        temps = [r for r in rows if r["metric"] == "motor_temperature_c"]
        assert temps[0]["value"] == 92.0
        empty, _ = repositories.get_sensor_data("O99", "2026-10-01", "2026-10-31")
        assert empty == []
    finally:
        db.close()


def test_beat_schedule_has_ingest_entry():
    entry = celery_app.conf.beat_schedule.get("run-ingest-batch")
    assert entry is not None
    assert entry["task"] == "app.worker.tasks.run_ingest_batch"
    assert entry["schedule"] == 1800.0
    assert "app.worker.tasks.run_ingest_batch" in celery_app.tasks
