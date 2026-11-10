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
        "faults": 8,
        "maintenance": 6,
        "documents": 6,
        "relations": 5,
        "sensor_readings": 62,
    }


def test_seed_data_all_o21_october_2026():
    assert demo_data.ASSET_ID == "O21"
    assert all(f["asset_id"] == "O21" for f in demo_data.FAULTS)
    assert all(m["asset_id"] == "O21" for m in demo_data.MAINTENANCE)
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
            "faults": 8,
            "maintenance": 6,
            "sensor_readings": 62,
        }
        assert db.query(FaultHistory).count() == 8
        assert db.query(MaintenanceRecord).count() == 6
        assert db.query(SensorReading).count() == 62
        fault = db.get(FaultHistory, "fh-o21-002")
        assert fault is not None
        assert fault.code == "F-204"
        assert fault.title == "Traction motor overheating"
        reading = db.get(SensorReading, "sr-o21-temp-07")
        assert reading is not None
        assert reading.value == 92.0
        assert reading.metric == "motor_temperature_c"
    finally:
        db.close()


def test_seed_tools_and_policies_demo(db_session):
    db = SessionLocal()
    try:
        first = demo_data.seed_tools_demo(db)
        second = demo_data.seed_tools_demo(db)
        assert first == second == {"tools": 17}
        assert db.query(Tool).count() == 17
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
    assert "8 faults" in out and "6 maintenance" in out
    assert "62 sensor readings" in out
    assert "No store was touched" in out
