"""Harness index DB mode (sqlite db_session, HARNESS_SOURCE=db)."""

from __future__ import annotations

import pytest

from app.services.mission_service import upsert_mission
from runtime.harness.registry import (
    reset_harness_cache_for_tests,
    user_runnable_mission_ids,
    _get_index,
    _refresh_index,
)
from seeds import load_missions


@pytest.fixture(autouse=True)
def _harness_db_mode(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "harness_source", "db")
    reset_harness_cache_for_tests()
    yield
    reset_harness_cache_for_tests()


def test_db_harness_overrides_seed_rbac(db_session, monkeypatch):
    row = load_missions.load_mission_file(
        load_missions.BASE / "missions" / "hotel_reception.yaml"
    )
    harness = dict(row["execution_config"]["harness"])
    harness["security"] = dict(harness["security"])
    harness["security"]["user_runnable"] = False
    row["execution_config"] = {**row["execution_config"], "harness": harness}
    upsert_mission(db_session, row)
    reset_harness_cache_for_tests()
    assert "hotel_reception" not in user_runnable_mission_ids()


def test_db_row_without_harness_key_falls_back_to_seed(db_session):
    row = load_missions.load_mission_file(
        load_missions.BASE / "missions" / "hotel_reception.yaml"
    )
    ec = dict(row["execution_config"])
    del ec["harness"]
    row["execution_config"] = ec
    upsert_mission(db_session, row)
    reset_harness_cache_for_tests()
    assert user_runnable_mission_ids() == frozenset({"hotel_reception"})


def test_db_invalid_harness_uses_closed_default(db_session):
    row = load_missions.load_mission_file(
        load_missions.BASE / "missions" / "hotel_reception.yaml"
    )
    row["execution_config"]["harness"] = {"builtin_tools": {"deni": True}}
    upsert_mission(db_session, row)
    reset_harness_cache_for_tests()
    assert "hotel_reception" not in user_runnable_mission_ids()
    h = _get_index(force=True).by_mission["hotel_reception"]
    assert h.builtin_tools.deny is False


def test_db_empty_table_no_rbac_missions(db_session):
    reset_harness_cache_for_tests()
    assert user_runnable_mission_ids() == frozenset()


def test_stale_while_error_keeps_last_good(db_session, monkeypatch):
    row = load_missions.load_mission_file(
        load_missions.BASE / "missions" / "hotel_reception.yaml"
    )
    upsert_mission(db_session, row)
    reset_harness_cache_for_tests()
    _refresh_index()
    assert "hotel_reception" in user_runnable_mission_ids()

    def _fail():
        raise RuntimeError("simulated db outage")

    monkeypatch.setattr("runtime.harness.registry._load_index_from_db", _fail)
    _get_index(force=True)
    assert "hotel_reception" in user_runnable_mission_ids()


def test_db_refresh_picks_up_after_recover(db_session, monkeypatch):
    row = load_missions.load_mission_file(
        load_missions.BASE / "missions" / "hotel_reception.yaml"
    )
    upsert_mission(db_session, row)
    reset_harness_cache_for_tests()
    _refresh_index()
    harness = dict(row["execution_config"]["harness"])
    harness["security"] = dict(harness["security"])
    harness["security"]["user_runnable"] = False
    row["execution_config"] = {**row["execution_config"], "harness": harness}
    upsert_mission(db_session, row)
    reset_harness_cache_for_tests()
    assert "hotel_reception" not in user_runnable_mission_ids()