"""Row schemas for batch-ingest files (Option B).

One JSON file holds one type of row; the file name decides the type:
``faults.json``, ``faults-2026-11.json``, ``sensor_readings.json``, ...
(the part before the first ``-`` in the stem). Every row is validated
here before any loader touches a store: ids, the ``asset_id`` allowlist,
string length caps and control-character stripping all reuse the
Layer 1 rules in ``runtime/security.py``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator, model_validator

from runtime.security import (
    ASSET_ID_RE,
    MAX_STRING_FIELD_CHARS,
    MAX_TEXT_FIELD_CHARS,
    strip_control_chars,
)


def _clean_str(value: object) -> object:
    if isinstance(value, str):
        return strip_control_chars(value)
    return value


class IngestRow(BaseModel):
    """Shared behaviour: strip control chars from every string field."""

    model_config = {"extra": "ignore"}

    @model_validator(mode="before")
    @classmethod
    def _strip_control_chars(cls, data: object) -> object:
        if isinstance(data, dict):
            return {k: _clean_str(v) for k, v in data.items()}
        return data


class FaultRow(IngestRow):
    id: str
    asset_id: str
    code: str
    title: str
    description: str | None = None
    severity: str = "medium"
    occurred_at: datetime
    resolved_at: datetime | None = None
    technician: str | None = None
    action: str | None = None

    @field_validator("id")
    @classmethod
    def _id_len(cls, v: str) -> str:
        assert 1 <= len(v) <= 128, "id must be 1..128 chars"
        return v

    @field_validator("asset_id")
    @classmethod
    def _asset_id(cls, v: str) -> str:
        assert ASSET_ID_RE.match(v), f"asset_id {v!r} is not allowed"
        return v

    @field_validator("code")
    @classmethod
    def _code_len(cls, v: str) -> str:
        assert 1 <= len(v) <= 64, "code must be 1..64 chars"
        return v

    @field_validator("title")
    @classmethod
    def _title_len(cls, v: str) -> str:
        assert 1 <= len(v) <= MAX_STRING_FIELD_CHARS, (
            f"title must be 1..{MAX_STRING_FIELD_CHARS} chars"
        )
        return v

    @field_validator("description", "action")
    @classmethod
    def _text_len(cls, v: str | None) -> str | None:
        if v is not None:
            assert len(v) <= MAX_TEXT_FIELD_CHARS, (
                f"text field must be at most {MAX_TEXT_FIELD_CHARS} chars"
            )
        return v

    @field_validator("severity")
    @classmethod
    def _severity(cls, v: str) -> str:
        assert v in ("low", "medium", "high"), f"unknown severity {v!r}"
        return v

    @model_validator(mode="after")
    def _resolved_after_occurred(self) -> FaultRow:
        if self.resolved_at is not None and self.resolved_at < self.occurred_at:
            raise ValueError("resolved_at is before occurred_at")
        return self


class MaintenanceRow(IngestRow):
    id: str
    asset_id: str
    date: datetime
    action: str
    technician: str | None = None
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def _id_len(cls, v: str) -> str:
        assert 1 <= len(v) <= 128, "id must be 1..128 chars"
        return v

    @field_validator("asset_id")
    @classmethod
    def _asset_id(cls, v: str) -> str:
        assert ASSET_ID_RE.match(v), f"asset_id {v!r} is not allowed"
        return v

    @field_validator("action")
    @classmethod
    def _action_len(cls, v: str) -> str:
        assert 1 <= len(v) <= MAX_TEXT_FIELD_CHARS, (
            f"action must be 1..{MAX_TEXT_FIELD_CHARS} chars"
        )
        return v

    @field_validator("notes")
    @classmethod
    def _notes_len(cls, v: str | None) -> str | None:
        if v is not None:
            assert len(v) <= MAX_TEXT_FIELD_CHARS, (
                f"notes must be at most {MAX_TEXT_FIELD_CHARS} chars"
            )
        return v


class DocumentRow(IngestRow):
    id: str
    title: str
    content: str
    category: str | None = None

    @field_validator("id")
    @classmethod
    def _id_len(cls, v: str) -> str:
        assert 1 <= len(v) <= 128, "id must be 1..128 chars"
        return v

    @field_validator("title")
    @classmethod
    def _title_len(cls, v: str) -> str:
        assert 1 <= len(v) <= MAX_STRING_FIELD_CHARS, (
            f"title must be 1..{MAX_STRING_FIELD_CHARS} chars"
        )
        return v

    @field_validator("content")
    @classmethod
    def _content_len(cls, v: str) -> str:
        assert 1 <= len(v) <= 10000, "content must be 1..10000 chars"
        return v


class RelationRow(IngestRow):
    from_id: str
    from_label: str
    relation: str
    to_id: str
    to_label: str

    @field_validator("from_id", "from_label", "relation", "to_id", "to_label")
    @classmethod
    def _field_len(cls, v: str) -> str:
        assert 1 <= len(v) <= 128, "relation field must be 1..128 chars"
        return v

    @field_validator("relation")
    @classmethod
    def _relation_shape(cls, v: str) -> str:
        assert v.isupper() and v.replace("_", "").isalpha(), (
            f"relation {v!r} must be UPPER_SNAKE_CASE"
        )
        return v


class SensorRow(IngestRow):
    id: str
    asset_id: str
    metric: str
    value: float
    unit: str = ""
    at: datetime

    @field_validator("id")
    @classmethod
    def _id_len(cls, v: str) -> str:
        assert 1 <= len(v) <= 128, "id must be 1..128 chars"
        return v

    @field_validator("asset_id")
    @classmethod
    def _asset_id(cls, v: str) -> str:
        assert ASSET_ID_RE.match(v), f"asset_id {v!r} is not allowed"
        return v

    @field_validator("metric")
    @classmethod
    def _metric_shape(cls, v: str) -> str:
        assert 1 <= len(v) <= 64, "metric must be 1..64 chars"
        assert v.replace("_", "").isalnum() and v.islower(), (
            f"metric {v!r} must be lower_snake_case"
        )
        return v

    @field_validator("unit")
    @classmethod
    def _unit_len(cls, v: str) -> str:
        assert len(v) <= 32, "unit must be at most 32 chars"
        return v


ROW_MODELS = {
    "faults": FaultRow,
    "maintenance": MaintenanceRow,
    "documents": DocumentRow,
    "relations": RelationRow,
    "sensor_readings": SensorRow,
}
