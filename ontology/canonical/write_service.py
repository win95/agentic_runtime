"""OntologyWriteService — sync canonical writes + projection enqueue."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.schema import (
    assert_relationship_type,
    normalize_entity_label,
)
from ontology.canonical.types import (
    CanonicalWriteResult,
    entity_ref,
    relationship_ref,
)
from ontology.models import OntologyEntity, OntologyRelationship

_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_MAX_KEY_LEN = 200


def compute_state_hash(properties: dict[str, Any] | None, status: str) -> str:
    canon = json.dumps({"p": properties or {}, "s": status}, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _validate_key(value: str, field: str) -> None:
    if not value or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty string")
    if len(value) > _MAX_KEY_LEN:
        raise ValueError(f"{field} exceeds {_MAX_KEY_LEN} chars")


def _change_type(exists: bool, status: str) -> str:
    if not exists:
        return "create"
    if status in ("archived", "deleted"):
        return status
    return "update"


class OntologyWriteService:
    def __init__(self, repository: CanonicalOntologyRepository) -> None:
        self._repo = repository

    def _now(self) -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)

    def _next_version(self, subject_ref: str, current: int | None) -> int:
        if current is not None:
            return current + 1
        latest = self._repo.get_latest_version(subject_ref)
        if latest is not None:
            return latest[0] + 1
        return 1

    def write_entity(
        self,
        *,
        entity_type: str,
        entity_key: str,
        properties: dict[str, Any] | None,
        provenance: WriteProvenance,
        status: str = "active",
        force_update: bool = False,
    ) -> CanonicalWriteResult:
        provenance.validate()
        entity_type = normalize_entity_label(entity_type)
        _validate_key(entity_key, "entity_key")
        ref = entity_ref(entity_type, entity_key)
        existing = self._repo.get_entity(entity_type, entity_key)
        new_hash = compute_state_hash(properties, status)

        if existing and not force_update:
            if (
                provenance.source_event_id
                and existing.source_event_id == provenance.source_event_id
            ):
                return CanonicalWriteResult(ref, existing.version, "replay", False, True)
            if existing.state_hash == new_hash and existing.status == status:
                return CanonicalWriteResult(ref, existing.version, "noop", False)

        change = _change_type(existing is not None, status)
        version = self._next_version(ref, existing.version if existing else None)
        now = self._now()

        if existing is None:
            row = OntologyEntity(
                entity_type=entity_type,
                entity_key=entity_key,
                properties=properties,
                state_hash=new_hash,
                version=version,
                status=status,
                trace_id=provenance.trace_id,
                source_event_id=provenance.source_event_id,
                created_at=now,
                updated_at=now,
            )
        else:
            row = existing
            row.properties = properties
            row.state_hash = new_hash
            row.version = version
            row.status = status
            row.trace_id = provenance.trace_id
            row.source_event_id = provenance.source_event_id
            row.updated_at = now

        self._repo.save_entity(row)
        snapshot = {
            "entity_type": entity_type,
            "entity_key": entity_key,
            "properties": properties,
            "status": status,
            "version": version,
        }
        self._repo.append_version(
            subject_kind="entity",
            subject_ref=ref,
            version=version,
            change_type=change,
            snapshot=snapshot,
            trace_id=provenance.trace_id,
            source_event_id=provenance.source_event_id,
        )
        self._repo.upsert_projection_pending(
            subject_kind="entity",
            subject_ref=ref,
            canonical_version=version,
            change_type=change,
            trace_id=provenance.trace_id,
        )
        return CanonicalWriteResult(ref, version, change, True)

    def write_relationship(
        self,
        *,
        relationship_type: str,
        source_type: str,
        source_key: str,
        target_type: str,
        target_key: str,
        properties: dict[str, Any] | None,
        provenance: WriteProvenance,
        status: str = "active",
        force_update: bool = False,
    ) -> CanonicalWriteResult:
        provenance.validate()
        relationship_type = assert_relationship_type(relationship_type)
        source_type = normalize_entity_label(source_type)
        target_type = normalize_entity_label(target_type)
        for key, label in (
            (source_key, "source_key"),
            (target_key, "target_key"),
        ):
            _validate_key(key, label)
        ref = relationship_ref(
            relationship_type, source_type, source_key, target_type, target_key
        )
        existing = self._repo.get_relationship(
            relationship_type, source_type, source_key, target_type, target_key
        )
        new_hash = compute_state_hash(properties, status)

        if existing and not force_update:
            if (
                provenance.source_event_id
                and existing.source_event_id == provenance.source_event_id
            ):
                return CanonicalWriteResult(ref, existing.version, "replay", False, True)
            if existing.state_hash == new_hash and existing.status == status:
                return CanonicalWriteResult(ref, existing.version, "noop", False)

        change = _change_type(existing is not None, status)
        version = self._next_version(ref, existing.version if existing else None)
        now = self._now()

        if existing is None:
            row = OntologyRelationship(
                relationship_type=relationship_type,
                source_type=source_type,
                source_key=source_key,
                target_type=target_type,
                target_key=target_key,
                properties=properties,
                state_hash=new_hash,
                version=version,
                status=status,
                trace_id=provenance.trace_id,
                source_event_id=provenance.source_event_id,
                created_at=now,
                updated_at=now,
            )
        else:
            row = existing
            row.properties = properties
            row.state_hash = new_hash
            row.version = version
            row.status = status
            row.trace_id = provenance.trace_id
            row.source_event_id = provenance.source_event_id
            row.updated_at = now

        self._repo.save_relationship(row)
        snapshot = {
            "relationship_type": relationship_type,
            "source_type": source_type,
            "source_key": source_key,
            "target_type": target_type,
            "target_key": target_key,
            "properties": properties,
            "status": status,
            "version": version,
        }
        self._repo.append_version(
            subject_kind="relationship",
            subject_ref=ref,
            version=version,
            change_type=change,
            snapshot=snapshot,
            trace_id=provenance.trace_id,
            source_event_id=provenance.source_event_id,
        )
        self._repo.upsert_projection_pending(
            subject_kind="relationship",
            subject_ref=ref,
            canonical_version=version,
            change_type=change,
            trace_id=provenance.trace_id,
        )
        return CanonicalWriteResult(ref, version, change, True)
