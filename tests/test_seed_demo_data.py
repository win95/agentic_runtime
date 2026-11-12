"""Seed demo-data tests — no live stores needed.

The Postgres part runs against the in-memory SQLite DB from conftest
(`db.merge` is store-agnostic); Qdrant/FalkorDB seeding needs live
services and is covered by `make seed-dry-run` + the data validation
below, not by these tests.
"""

from app.core.db import SessionLocal
from app.models import FaultHistory, MaintenanceRecord, Policy, SensorReading, Tool
from seeds import demo_data
from seeds.seed_demo_data import main as seed_main


def test_seed_data_counts():
    counts = demo_data.validate()
    assert counts == {
        "faults": 10,
        "maintenance": 9,
        "documents": 7,
        "relations": 15,
        "sensor_readings": 76,
    }


def test_seed_data_per_asset_counts():
    """Per-asset counts — the O21 October 2026 dataset is fixed, and the
    edge-case assets' absences are load-bearing (see validate())."""
    assert demo_data.ASSET_ID == "O21"
    per_asset = {
        asset: (
            sum(1 for r in demo_data.FAULTS if r["asset_id"] == asset),
            sum(1 for r in demo_data.MAINTENANCE if r["asset_id"] == asset),
            sum(1 for r in demo_data.SENSOR_READINGS if r["asset_id"] == asset),
        )
        for asset in ("O21", "O22", "O23", "O24")
    }
    assert per_asset == {
        "O21": (8, 6, 62),
        "O22": (2, 1, 6),
        "O23": (0, 0, 4),
        "O24": (0, 2, 4),
    }
    # O20 (archived) has no Postgres rows at all.
    assert not any(r["asset_id"] == "O20" for r in demo_data.FAULTS)
    assert not any(r["asset_id"] == "O20" for r in demo_data.MAINTENANCE)
    assert not any(r["asset_id"] == "O20" for r in demo_data.SENSOR_READINGS)
    assert all(f["occurred_at"].year == 2026 and f["occurred_at"].month == 10
               for f in demo_data.FAULTS)


def test_seed_data_fault_codes_match_eval():
    # The eval's mock summary names F-101 and F-204 as the dominant
    # causes; the seed data must contain both.
    codes = {f["code"] for f in demo_data.FAULTS}
    assert {"F-101", "F-204"} <= codes


def test_seed_postgres_demo_idempotent(db_session):
    db = SessionLocal()
    try:
        first = demo_data.seed_postgres_demo(db)
        second = demo_data.seed_postgres_demo(db)
        assert first == second == {
            "faults": 10,
            "maintenance": 9,
            "sensor_readings": 76,
        }
        assert db.query(FaultHistory).count() == 10
        assert db.query(MaintenanceRecord).count() == 9
        assert db.query(SensorReading).count() == 76
        assert db.query(FaultHistory).filter_by(asset_id="O21").count() == 8
        assert db.query(FaultHistory).filter_by(asset_id="O22").count() == 2
        assert db.query(FaultHistory).filter_by(asset_id="O23").count() == 0
        assert db.query(FaultHistory).filter_by(asset_id="O24").count() == 0
        assert db.query(MaintenanceRecord).filter_by(asset_id="O23").count() == 0
        assert db.query(MaintenanceRecord).filter_by(asset_id="O24").count() == 2
        fault = db.get(FaultHistory, "fh-o21-002")
        assert fault is not None
        assert fault.code == "F-204"
        assert fault.title == "Traction motor overheating"
        reading = db.get(SensorReading, "sr-o21-temp-07")
        assert reading is not None
        assert reading.value == 92.0
        assert reading.metric == "motor_temperature_c"
        # O22's spike is coherent with its F-206 fault.
        spike = db.get(SensorReading, "sr-o22-temp-11")
        assert spike is not None
        assert spike.value == 88.5
    finally:
        db.close()


def test_seed_executions_demo_idempotent(db_session):
    from app.models import Execution, ExecutionResult
    from seeds.load_missions import main as seed_missions

    seed_missions()  # missions must exist for the FK
    db = SessionLocal()
    try:
        first = demo_data.seed_executions_demo(db)
        second = demo_data.seed_executions_demo(db)
        assert first == second == {"executions": 3}
        assert db.query(Execution).count() == 3
        assert db.query(ExecutionResult).count() == 3
        succeeded = db.get(Execution, "exec-seed-fa-001")
        assert succeeded.status == "SUCCEEDED"
        assert succeeded.finished_at is not None
        assert succeeded.mission_id == "fault_analysis"
        failed = db.get(Execution, "exec-seed-fa-002")
        assert failed.status == "FAILED"
        assert failed.error is not None
        assert failed.finished_at is not None
        admin = db.get(Execution, "exec-seed-sa-001")
        assert admin.status == "SUCCEEDED"
        assert admin.mission_id == "system_admin"
        result = db.get(ExecutionResult, "res-seed-fa-001")
        assert result is not None
        assert result.execution_id == "exec-seed-fa-001"
        assert result.output["root_causes"]
        assert result.evidence
    finally:
        db.close()


def test_seed_tools_and_policies_demo(db_session):
    db = SessionLocal()
    try:
        first = demo_data.seed_tools_demo(db)
        second = demo_data.seed_tools_demo(db)
        assert first == second == {"tools": 14}
        assert db.query(Tool).count() == 14
        assert db.get(Tool, "tool_get_sensor_data") is not None
        assert db.get(Tool, "tool_dispatch_execution") is not None
        policies = demo_data.seed_policies_demo(db)
        assert policies == {"policies": 1}
        policy = db.get(Policy, "policy_default")
        assert policy is not None
        assert policy.tenant == "default"
        assert "system_admin" in policy.allowed_capabilities
        assert "get_sensor_data" in policy.allowed_tools
    finally:
        db.close()


def test_seed_dry_run(capsys):
    assert seed_main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Dry run OK" in out
    assert "10 faults" in out and "9 maintenance" in out
    assert "76 sensor readings" in out
    assert "7 documents" in out and "15 relations" in out
    assert "O21/O22/O23/O24" in out
    assert "No store was touched" in out
