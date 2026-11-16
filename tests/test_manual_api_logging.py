"""Manual script HTTP log helpers — redaction and truncation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_MANUAL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "manual"
if str(_MANUAL_DIR) not in sys.path:
    sys.path.insert(0, str(_MANUAL_DIR))

import _api  # noqa: E402

from _api import (  # noqa: E402
    execution_id_from_path,
    extract_answer_from_result,
    extract_question_text,
    redact_for_log,
    reset_dotenv_state_for_tests,
    reset_turn_log_state_for_tests,
    truncate_log_text,
)


@pytest.fixture(autouse=True)
def _no_dotenv_during_pure_helper_tests(monkeypatch):
    """Pure helpers must not read repo ``.env`` (pytest import order regression)."""
    monkeypatch.setattr(_api, "load_dotenv", lambda path=None: None)
    reset_dotenv_state_for_tests()
    yield
    reset_dotenv_state_for_tests()


def test_redact_for_log_masks_secrets():
    raw = {
        "Authorization": "Bearer abc",
        "X-API-Key": "k",
        "password": "pw",
        "token": "t",
        "access_token": "at",
        "api_key": "ak",
        "secret": "s",
        "message": "hello",
        "nested": {"guest_id": "g1", "api_key": "inner"},
    }
    out = redact_for_log(raw)
    assert out["message"] == "hello"
    assert out["Authorization"] == "***"
    assert out["X-API-Key"] == "***"
    assert out["password"] == "***"
    assert out["nested"]["api_key"] == "***"
    assert out["nested"]["guest_id"] == "g1"


def test_truncate_log_text_appends_marker():
    text = "x" * 100
    out = truncate_log_text(text, limit=40)
    assert out.startswith("x" * 40)
    assert "60 more chars" in out


def test_extract_question_text_nested_and_top_level():
    assert extract_question_text({"message": "hello guest"}) == "hello guest"
    body = {
        "mission_id": "hotel_reception",
        "input": {"message": "giá phòng", "guest_id": "g1"},
    }
    assert extract_question_text(body) == "giá phòng"
    assert extract_question_text({"input": {"request": "health check"}}) == "health check"
    assert extract_question_text({"input": {"asset_id": "O21"}}) is None


def test_extract_answer_from_result():
    payload = {
        "output": {
            "answer": "Deluxe còn trống.",
            "actions_taken": ["check_room_availability", "reply_to_guest"],
        }
    }
    answer, actions = extract_answer_from_result(payload)
    assert answer == "Deluxe còn trống."
    assert actions == ["check_room_availability", "reply_to_guest"]
    payload["output"]["summary"] = "fallback"
    del payload["output"]["answer"]
    answer2, _ = extract_answer_from_result(payload)
    assert answer2 == "fallback"


def test_execution_id_from_path():
    assert execution_id_from_path("/executions/exec_abc/result") == "exec_abc"
    assert execution_id_from_path("http://localhost:8000/executions/exec_x") == "exec_x"


def test_redacted_json_roundtrip_is_valid_json():
    payload = {"token": "raw", "ok": [1, 2]}
    cleaned = redact_for_log(payload)
    dumped = json.dumps(cleaned)
    assert json.loads(dumped)["token"] == "***"


def test_pure_helpers_do_not_trigger_dotenv_load(monkeypatch):
    """Regression: log helpers must not load repo .env (test_config import order)."""
    load_calls: list[object] = []
    monkeypatch.setattr(_api, "load_dotenv", lambda path=None: load_calls.append(1))
    reset_dotenv_state_for_tests()
    assert extract_question_text({"message": "probe"}) == "probe"
    assert load_calls == []
    assert _api._DOTENV_ATTEMPTED is False


def test_api_base_loads_dotenv_once(monkeypatch):
    load_calls: list[object] = []

    def track_load(path=None):
        load_calls.append(path)
        return None

    monkeypatch.setattr(_api, "load_dotenv", track_load)
    reset_dotenv_state_for_tests()
    _api.api_base()
    _api.api_base()
    assert len(load_calls) == 1