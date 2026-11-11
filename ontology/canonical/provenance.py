"""Minimal write provenance for canonical ontology."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WriteProvenance:
    trace_id: str
    source_event_id: str | None = None

    def validate(self) -> None:
        if not (self.trace_id or "").strip():
            raise ValueError("trace_id is required for ontology writes")
