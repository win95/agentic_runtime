"""OntologyWriteService — sync canonical writes + projection enqueue."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import IntegrityError

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

_MAX_KEY_LEN = 200
_STATUSES: frozenset[str] = frozenset({"active", "archived", "deleted"})


def compute_state_hash(properties: dict[str, Any] | None, status: str) -> str:
    canon = json.dumps({"p": properties or {}, "s": status}, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _validate_key(value: str, field: str) -> str:
    """Return the key stripped of surrounding whitespace.

    ``:`` is rejected: keys are embedded in ``subject_ref`` values
    (``entity:Asset:O21``, ``rel:PART_OF:Asset:O21:...``) whose parsers
    split on ``:``, so a colon in a key corrupts the ref and poisons
    the projection queue. Control characters are rejected too.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    cleaned = value.strip()
    if len(cleaned) > _MAX_KEY_LEN:
        raise ValueError(f"{field} exceeds {_MAX_KEY_LEN} chars")
    if ":" in cleaned:
        raise ValueError(f"{field} must not contain ':'")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cleaned):
        raise ValueError(f"{field} must not contain control characters")
    return cleaned


def _validate_status(status: str) -> str:
    if status not in _STATUSES:
        raise ValueError(f"status {status!r} must be one of {sorted(_STATUSES)}")
    return status


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
        # Next version is one past the highest version seen anywhere:
        # the row's own version and the latest version row. They agree
        # in the normal flow, but after a concurrent writer inserts a
        # version row the entity version alone would collide with it.
        latest = self._repo.get_latest_version(subject_ref)
        base = current if current is not None else 0
        if latest is not None:
            base = max(base, latest[0])
        return base + 1

    def _persist_with_retry(self, persist, retry):
        """Run ``persist`` (row insert/update + version append) in a
        SAVEPOINT; on IntegrityError roll back only to the savepoint and
        run ``retry`` once, which re-reads the row the concurrent writer
        inserted and applies the write as an update. The surrounding
        transaction is left intact either way.
        """
        try:
            with self._repo.session.begin_nested():
                persist()
        except IntegrityError:
            retry()

    def ensure_entity(
        self,
        *,
        entity_type: str,
        entity_key: str,
        properties: dict[str, Any] | None,
        provenance: WriteProvenance,
        status: str = "active",
    ) -> CanonicalWriteResult:
        """Create the entity only if it does not exist yet.

        Relation merges know only the endpoint identity, not its
        properties: writing ``properties=None`` over an existing endpoint
        would null out properties an earlier write stored, so an
        existing endpoint is left untouched (a noop, not a replay).
        """
        entity_type = normalize_entity_label(entity_type)
        entity_key = _validate_key(entity_key, "entity_key")
        existing = self._repo.get_entity(entity_type, entity_key)
        if existing is not None:
            return CanonicalWriteResult(
                entity_ref(entity_type, entity_key), existing.version, "noop", False
            )
        return self.write_entity(
            entity_type=entity_type,
            entity_key=entity_key,
            properties=properties,
            provenance=provenance,
            status=status,
        )

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
        status = _validate_status(status)
        entity_type = normalize_entity_label(entity_type)
        entity_key = _validate_key(entity_key, "entity_key")
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

        snapshot = {
            "entity_type": entity_type,
            "entity_key": entity_key,
            "properties": properties,
            "status": status,
            "version": version,
        }

        def _persist() -> None:
            self._repo.save_entity(row)
            self._repo.append_version(
                subject_kind="entity",
                subject_ref=ref,
                version=row.version,
                change_type=change,
                snapshot={**snapshot, "version": row.version},
                trace_id=provenance.trace_id,
                source_event_id=provenance.source_event_id,
            )

        def _retry() -> None:
            # The row we just tried to insert (or the version row) was
            # already written by a concurrent writer; drop our failed
            # attempt and re-apply as an update of the row now present.
            # (A savepoint rollback already detaches a failed pending
            # insert, so expunge only when it is still attached.)
            if self._repo.session.object_session(row) is self._repo.session:
                self._repo.session.expunge(row)
            current = self._repo.get_entity(entity_type, entity_key)
            if current is None:
                raise
            current.properties = properties
            current.state_hash = new_hash
            current.version = self._next_version(ref, current.version)
            current.status = status
            current.trace_id = provenance.trace_id
            current.source_event_id = provenance.source_event_id
            current.updated_at = self._now()
            self._repo.save_entity(current)
            self._repo.append_version(
                subject_kind="entity",
                subject_ref=ref,
                version=current.version,
                change_type=_change_type(True, status),
                snapshot={**snapshot, "version": current.version},
                trace_id=provenance.trace_id,
                source_event_id=provenance.source_event_id,
            )
            nonlocal_version[0] = current.version
            nonlocal_change[0] = _change_type(True, status)

        nonlocal_version = [version]
        nonlocal_change = [change]
        self._persist_with_retry(_persist, _retry)
        version = nonlocal_version[0]
        change = nonlocal_change[0]
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
        status = _validate_status(status)
        relationship_type = assert_relationship_type(relationship_type)
        source_type = normalize_entity_label(source_type)
        target_type = normalize_entity_label(target_type)
        source_key = _validate_key(source_key, "source_key")
        target_key = _validate_key(target_key, "target_key")
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

        def _persist() -> None:
            self._repo.save_relationship(row)
            self._repo.append_version(
                subject_kind="relationship",
                subject_ref=ref,
                version=row.version,
                change_type=change,
                snapshot={**snapshot, "version": row.version},
                trace_id=provenance.trace_id,
                source_event_id=provenance.source_event_id,
            )

        def _retry() -> None:
            if self._repo.session.object_session(row) is self._repo.session:
                self._repo.session.expunge(row)
            current = self._repo.get_relationship(
                relationship_type, source_type, source_key, target_type, target_key
            )
            if current is None:
                raise
            current.properties = properties
            current.state_hash = new_hash
            current.version = self._next_version(ref, current.version)
            current.status = status
            current.trace_id = provenance.trace_id
            current.source_event_id = provenance.source_event_id
            current.updated_at = self._now()
            self._repo.save_relationship(current)
            self._repo.append_version(
                subject_kind="relationship",
                subject_ref=ref,
                version=current.version,
                change_type=_change_type(True, status),
                snapshot={**snapshot, "version": current.version},
                trace_id=provenance.trace_id,
                source_event_id=provenance.source_event_id,
            )
            nonlocal_version[0] = current.version
            nonlocal_change[0] = _change_type(True, status)

        nonlocal_version = [version]
        nonlocal_change = [change]
        self._persist_with_retry(_persist, _retry)
        version = nonlocal_version[0]
        change = nonlocal_change[0]
        self._repo.upsert_projection_pending(
            subject_kind="relationship",
            subject_ref=ref,
            canonical_version=version,
            change_type=change,
            trace_id=provenance.trace_id,
        )
        return CanonicalWriteResult(ref, version, change, True)
