"""Manual script HTTP log helpers — redaction and truncation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "manual"))

from _api import (  # noqa: E402
    redact_for_log,
    truncate_log_text,
)


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


def test_redacted_json_roundtrip_is_valid_json():
    payload = {"token": "raw", "ok": [1, 2]}
    cleaned = redact_for_log(payload)
    dumped = json.dumps(cleaned)
    assert json.loads(dumped)["token"] == "***"