"""OntologyWriteService — dual write: Neo4j first, L1 state after (design §5).

Write order (chosen so a Neo4j failure leaves L1 untouched, and an
L1 failure after a Neo4j write self-heals on retry):

1. Structural validation (label regex, key rules, status enum).
   The class/relation allowlist is NOT structural: it lives in the
   metadata validator (B2, ``ontology.metadata.validate``), which
   checks the class/property against the Neo4j metadata and falls
   back to the v1 allowlists when no metadata is seeded yet. The
   validator runs here, before idempotency (design §5 order).
2. Idempotency / self-heal — look L1 up by ``source_event_id``:
   a hit whose object is still on Neo4j replays (no re-write); L1
   missing but the object present on Neo4j backfills L1; identical
   properties on Neo4j is a noop (this replaces ``state_hash``).
3. Write Neo4j: MERGE node/edge + SET all properties + ``status`` +
   ``version``. A base edge also carries ``base_ref`` (the shared
   ``ontology.reasoning.base_ref_for`` helper — never another format)
   and never a ``derived`` key; after the merge its derived edges are
   (re-)materialized, and before an overwrite/delete the old derived
   edges are removed by ``base_ref``. A Neo4j error raises at once:
   no L1 write, no queue (tran's decision 3).
4. Write L1 in one transaction: upsert the state row (status, new
   version = max(L1 version, Neo4j version) + 1, trace/source_event)
   + upsert the outbox row (state-sync intent). An L1 failure here
   raises — the object is already on Neo4j and the retry with the
   same ``source_event_id`` self-heals at step 2.

Delete (status archived/deleted) keeps the old behaviour: Neo4j
DETACH DELETE the node (derived edges go with it) / delete the edge
(after removing its derived edges by ``base_ref``) + L1 status update.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import IntegrityError

from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.schema import (
    ChangeType,
    EntityStatus,
    WriteOutcome,
)
from ontology.canonical.types import (
    CanonicalWriteResult,
    entity_ref,
    relationship_ref,
)
from ontology.graph.protocol import GraphStore
from ontology.models import OntologyEntity, OntologyRelationship
from ontology.reasoning import (
    base_ref_for,
    materialize_for_write,
    remove_derived_for_base,
)

logger = logging.getLogger(__name__)

_MAX_KEY_LEN = 200

_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _normalize_label(raw: str) -> str:
    """Structural label check only (strip + identifier regex).

    The allowlist is deliberately NOT checked here:
    ``schema.normalize_entity_label`` hardcodes the v1 classes, which
    would reject any class created through the metadata API before
    the metadata validator ever saw it. Membership is the
    validator's job (``ontology.metadata.validate`` — metadata when
    seeded, v1 fallback otherwise).
    """
    value = (raw or "").strip()
    if not _LABEL_RE.match(value):
        raise ValueError(f"Invalid entity_type identifier: {raw!r}")
    return value


def _normalize_relationship_type(raw: str) -> str:
    """Structural relation-type check only (strip + upper + regex);
    membership is the validator's job, as for labels."""
    value = (raw or "").strip().upper()
    if not _LABEL_RE.match(value):
        raise ValueError(f"Invalid relationship_type identifier: {raw!r}")
    return value

# Node/edge properties owned by the store layer, not by the caller:
# stripped before any user-property comparison so a write -> read ->
# compare round-trip is stable (merge_node also adds ``updated_at``).
_NODE_SYSTEM_KEYS = frozenset({"id", "status", "version", "updated_at"})
_EDGE_SYSTEM_KEYS = frozenset({"status", "version", "base_ref", "derived", "rule"})


def _user_props(
    props: dict[str, Any] | None, system_keys: frozenset[str]
) -> dict[str, Any]:
    """Caller-owned properties only (no store-managed keys)."""
    return {k: v for k, v in (props or {}).items() if k not in system_keys}


def _object_version(props: dict[str, Any] | None) -> int:
    """The ``version`` property on a Neo4j node/edge (0 when absent)."""
    if not props:
        return 0
    try:
        return int(props.get("version") or 0)
    except (TypeError, ValueError):
        return 0


def _object_status(props: dict[str, Any] | None) -> str:
    """The ``status`` property on a Neo4j node/edge (active when absent)."""
    if not props:
        return EntityStatus.ACTIVE.value
    return str(props.get("status") or EntityStatus.ACTIVE.value)


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


def _validate_status(status: str | EntityStatus) -> EntityStatus:
    try:
        return EntityStatus(status)
    except ValueError:
        raise ValueError(
            f"status {status!r} must be one of "
            f"{sorted(item.value for item in EntityStatus)}"
        ) from None


def _change_type(exists: bool, status: str | EntityStatus) -> ChangeType:
    if not exists:
        return ChangeType.CREATE
    status = EntityStatus(status)
    if status in EntityStatus.inactive():
        return ChangeType(status.value)
    return ChangeType.UPDATE


class OntologyWriteService:
    def __init__(
        self,
        repository: CanonicalOntologyRepository,
        *,
        store: GraphStore | None = None,
        validator: Any | None = None,
    ) -> None:
        self._repo = repository
        self._store = store
        # Metadata validator (B2): a caller-supplied callable is
        # called as validator(store, entity_type=..., properties=...)
        # / validator(store, relationship_type=..., source_type=...,
        # source_key=..., target_type=..., target_key=...). None means
        # the default metadata validator (lazy-imported in
        # _validate_entity/_validate_relationship to avoid a cycle).
        self._validator = validator

    def _get_store(self) -> GraphStore:
        if self._store is None:
            # NOTE: there is no ``ontology.graph.provider`` module —
            # the lazy graph-store singleton lives in
            # ``ontology.projection.provider``.
            from ontology.projection.provider import get_graph_store

            self._store = get_graph_store()
        return self._store

    def _validate_entity(
        self,
        store: GraphStore,
        *,
        entity_type: str,
        properties: dict[str, Any] | None,
    ) -> None:
        """Metadata validation for one entity write (B2).

        A caller-supplied validator gets the same keyword arguments
        as the default; with none supplied, the default metadata
        validator is lazy-imported (import inside the method to avoid
        a cycle) and called.
        """
        if self._validator is not None:
            self._validator(store, entity_type=entity_type, properties=properties)
            return
        from ontology.metadata.validate import default_entity_validator

        default_entity_validator(store, entity_type=entity_type, properties=properties)

    def _validate_relationship(
        self,
        store: GraphStore,
        *,
        relationship_type: str,
        source_type: str,
        source_key: str,
        target_type: str,
        target_key: str,
    ) -> None:
        """Metadata validation for one relationship write (B2)."""
        if self._validator is not None:
            self._validator(
                store,
                relationship_type=relationship_type,
                source_type=source_type,
                source_key=source_key,
                target_type=target_type,
                target_key=target_key,
            )
            return
        from ontology.metadata.validate import default_relationship_validator

        default_relationship_validator(
            store,
            relationship_type=relationship_type,
            source_type=source_type,
            source_key=source_key,
            target_type=target_type,
            target_key=target_key,
        )

    def _now(self) -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)

    def _persist_with_retry(self, persist, retry):
        """Run ``persist`` (state-row insert/update + outbox upsert)
        in a SAVEPOINT; on IntegrityError roll back only to the
        savepoint and run ``retry`` once, which re-reads the row the
        concurrent writer inserted and applies the write as an update.
        The surrounding transaction is left intact either way. (The
        old versions-table unique constraint that motivated the
        original helper is gone; what remains is the subject-key
        unique constraint on the state row itself.)
        """
        try:
            with self._repo.session.begin_nested():
                persist()
        except IntegrityError:
            retry()

    def _save_entity_state(
        self,
        *,
        entity_type: str,
        entity_key: str,
        status: EntityStatus | str,
        version: int,
        change: ChangeType | str,
        trace_id: str,
        source_event_id: str | None,
    ) -> OntologyEntity:
        ref = entity_ref(entity_type, entity_key)
        now = self._now()
        holder: dict[str, OntologyEntity] = {}

        def _apply(row: OntologyEntity | None) -> OntologyEntity:
            if row is None:
                row = OntologyEntity(
                    entity_type=entity_type,
                    entity_key=entity_key,
                    version=version,
                    status=status,
                    trace_id=trace_id,
                    source_event_id=source_event_id,
                    created_at=now,
                    updated_at=now,
                )
            else:
                row.version = version
                row.status = status
                row.trace_id = trace_id
                row.source_event_id = source_event_id
                row.updated_at = now
            self._repo.save_entity(row)
            self._repo.upsert_projection_pending(
                subject_kind="entity",
                subject_ref=ref,
                canonical_version=version,
                change_type=change,
                trace_id=trace_id,
            )
            return row

        def _persist() -> None:
            holder["row"] = _apply(self._repo.get_entity(entity_type, entity_key))

        def _retry() -> None:
            # The row we just tried to insert was already written by a
            # concurrent writer; drop our failed attempt and re-apply
            # as an update of the row now present.
            current = self._repo.get_entity(entity_type, entity_key)
            if current is None:
                raise RuntimeError(
                    f"Concurrent insert of {ref} vanished on retry"
                )
            holder["row"] = _apply(current)

        self._persist_with_retry(_persist, _retry)
        return holder["row"]

    def _save_relationship_state(
        self,
        *,
        relationship_type: str,
        source_type: str,
        source_key: str,
        target_type: str,
        target_key: str,
        status: EntityStatus | str,
        version: int,
        change: ChangeType | str,
        trace_id: str,
        source_event_id: str | None,
    ) -> OntologyRelationship:
        ref = relationship_ref(
            relationship_type, source_type, source_key, target_type, target_key
        )
        now = self._now()
        holder: dict[str, OntologyRelationship] = {}

        def _apply(row: OntologyRelationship | None) -> OntologyRelationship:
            if row is None:
                row = OntologyRelationship(
                    relationship_type=relationship_type,
                    source_type=source_type,
                    source_key=source_key,
                    target_type=target_type,
                    target_key=target_key,
                    version=version,
                    status=status,
                    trace_id=trace_id,
                    source_event_id=source_event_id,
                    created_at=now,
                    updated_at=now,
                )
            else:
                row.version = version
                row.status = status
                row.trace_id = trace_id
                row.source_event_id = source_event_id
                row.updated_at = now
            self._repo.save_relationship(row)
            self._repo.upsert_projection_pending(
                subject_kind="relationship",
                subject_ref=ref,
                canonical_version=version,
                change_type=change,
                trace_id=trace_id,
            )
            return row

        def _persist() -> None:
            holder["row"] = _apply(
                self._repo.get_relationship(
                    relationship_type, source_type, source_key, target_type, target_key
                )
            )

        def _retry() -> None:
            current = self._repo.get_relationship(
                relationship_type, source_type, source_key, target_type, target_key
            )
            if current is None:
                raise RuntimeError(
                    f"Concurrent insert of {ref} vanished on retry"
                )
            holder["row"] = _apply(current)

        self._persist_with_retry(_persist, _retry)
        return holder["row"]

    def ensure_entity(
        self,
        *,
        entity_type: str,
        entity_key: str,
        properties: dict[str, Any] | None,
        provenance: WriteProvenance,
        status: EntityStatus | str = EntityStatus.ACTIVE,
    ) -> CanonicalWriteResult:
        """Create the entity only if it does not exist yet.

        Relation merges know only the endpoint identity, not its
        properties: when the node already exists on Neo4j this is a
        noop (the node is left untouched), and when the node exists
        but L1 has no state row for it the row is backfilled from the
        node (design §5 step 2 self-heal).
        """
        entity_type = _normalize_label(entity_type)
        entity_key = _validate_key(entity_key, "entity_key")
        status = _validate_status(status)
        ref = entity_ref(entity_type, entity_key)
        existing = self._repo.get_entity(entity_type, entity_key)
        node = self._get_store().get_node(entity_type, entity_key)
        if node is not None:
            if existing is None:
                self._save_entity_state(
                    entity_type=entity_type,
                    entity_key=entity_key,
                    status=_object_status(node),
                    version=_object_version(node) or 1,
                    change=ChangeType.CREATE,
                    trace_id=provenance.trace_id,
                    source_event_id=provenance.source_event_id,
                )
                version = _object_version(node) or 1
            else:
                version = existing.version
            return CanonicalWriteResult(ref, version, WriteOutcome.NOOP, False)
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
        status: EntityStatus | str = EntityStatus.ACTIVE,
        force_update: bool = False,
    ) -> CanonicalWriteResult:
        provenance.validate()
        status = _validate_status(status)
        entity_type = _normalize_label(entity_type)
        entity_key = _validate_key(entity_key, "entity_key")
        ref = entity_ref(entity_type, entity_key)
        store = self._get_store()
        # (1) Metadata validation (B2) — before idempotency (design
        # §5 order). Deletes skip it: a class removed from metadata
        # must not block deleting its instances.
        if status not in EntityStatus.inactive():
            self._validate_entity(
                store, entity_type=entity_type, properties=properties
            )
        existing = self._repo.get_entity(entity_type, entity_key)
        node = store.get_node(entity_type, entity_key)

        # (2) Replay: this subject's last write already carried this
        # source_event_id and the object is still on Neo4j.
        if provenance.source_event_id and not force_update:
            replayed = self._repo.get_entity_by_source_event_id(
                entity_type, entity_key, provenance.source_event_id
            )
            if replayed is not None and node is not None:
                return CanonicalWriteResult(
                    ref,
                    _object_version(node) or replayed.version,
                    WriteOutcome.REPLAY,
                    False,
                    True,
                )
            # Replay hit but the Neo4j object is gone: fall through
            # and recreate it (the write below re-upserts L1).

        if status in EntityStatus.inactive():
            return self._delete_entity(
                store,
                existing=existing,
                node=node,
                entity_type=entity_type,
                entity_key=entity_key,
                ref=ref,
                status=status,
                provenance=provenance,
            )

        # (2) Noop / self-heal, both decided by comparing against the
        # Neo4j node (replaces the old state_hash comparison).
        # ``properties=None`` means "no payload" — keep what the node
        # already has (a relation-stub write must not wipe rich entity
        # properties). Only an explicit dict replaces them.
        current_user = _user_props(node, _NODE_SYSTEM_KEYS) if node is not None else {}
        effective = dict(properties) if properties is not None else current_user
        if node is not None and not force_update:
            if effective == current_user and _object_status(node) == status.value:
                if existing is None:
                    # Neo4j has the object, L1 has no state row: a
                    # previous write died after Neo4j. Backfill L1.
                    self._save_entity_state(
                        entity_type=entity_type,
                        entity_key=entity_key,
                        status=_object_status(node),
                        version=_object_version(node) or 1,
                        change=ChangeType.CREATE,
                        trace_id=provenance.trace_id,
                        source_event_id=provenance.source_event_id,
                    )
                version = (
                    existing.version
                    if existing is not None
                    else _object_version(node) or 1
                )
                return CanonicalWriteResult(ref, version, WriteOutcome.NOOP, False)

        # (3) Neo4j first: MERGE + SET all properties + status + version.
        version = max(existing.version if existing else 0, _object_version(node)) + 1
        change = _change_type(existing is not None, status)
        store.merge_node(
            entity_type,
            entity_key,
            {**effective, "status": status.value, "version": version},
        )

        # (4) L1 state + outbox. On failure the object is already on
        # Neo4j; the retry with the same source_event_id self-heals.
        self._save_entity_state(
            entity_type=entity_type,
            entity_key=entity_key,
            status=status,
            version=version,
            change=change,
            trace_id=provenance.trace_id,
            source_event_id=provenance.source_event_id,
        )
        return CanonicalWriteResult(ref, version, change, True)

    def _delete_entity(
        self,
        store: GraphStore,
        *,
        existing: OntologyEntity | None,
        node: dict[str, Any] | None,
        entity_type: str,
        entity_key: str,
        ref: str,
        status: EntityStatus,
        provenance: WriteProvenance,
    ) -> CanonicalWriteResult:
        # Replay: this subject was already deleted under this event id.
        if (
            provenance.source_event_id
            and existing is not None
            and existing.source_event_id == provenance.source_event_id
            and node is None
        ):
            return CanonicalWriteResult(
                ref, existing.version, WriteOutcome.REPLAY, False, True
            )
        # Already in this inactive status and nothing on Neo4j.
        if (
            node is None
            and existing is not None
            and existing.status == status.value
        ):
            return CanonicalWriteResult(ref, existing.version, WriteOutcome.NOOP, False)
        version = max(existing.version if existing else 0, _object_version(node)) + 1
        change = _change_type(existing is not None, status)
        if node is not None:
            # DETACH DELETE: derived edges attached to the node go with it.
            store.delete_node(entity_type, entity_key)
        self._save_entity_state(
            entity_type=entity_type,
            entity_key=entity_key,
            status=status,
            version=version,
            change=change,
            trace_id=provenance.trace_id,
            source_event_id=provenance.source_event_id,
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
        status: EntityStatus | str = EntityStatus.ACTIVE,
        force_update: bool = False,
    ) -> CanonicalWriteResult:
        provenance.validate()
        status = _validate_status(status)
        relationship_type = _normalize_relationship_type(relationship_type)
        source_type = _normalize_label(source_type)
        target_type = _normalize_label(target_type)
        source_key = _validate_key(source_key, "source_key")
        target_key = _validate_key(target_key, "target_key")
        ref = relationship_ref(
            relationship_type, source_type, source_key, target_type, target_key
        )
        store = self._get_store()
        # (1) Metadata validation (B2) — before idempotency (design
        # §5 order). Deletes skip it, as for entities.
        if status not in EntityStatus.inactive():
            self._validate_relationship(
                store,
                relationship_type=relationship_type,
                source_type=source_type,
                source_key=source_key,
                target_type=target_type,
                target_key=target_key,
            )
        existing = self._repo.get_relationship(
            relationship_type, source_type, source_key, target_type, target_key
        )
        edge = store.get_relationship(
            source_type, source_key, relationship_type, target_type, target_key
        )

        # (2) Replay: this edge's last write already carried this
        # source_event_id and the edge is still on Neo4j.
        if provenance.source_event_id and not force_update:
            replayed = self._repo.get_relationship_by_source_event_id(
                relationship_type,
                source_type,
                source_key,
                target_type,
                target_key,
                provenance.source_event_id,
            )
            if replayed is not None and edge is not None:
                return CanonicalWriteResult(
                    ref,
                    _object_version(edge) or replayed.version,
                    WriteOutcome.REPLAY,
                    False,
                    True,
                )
            # Replay hit but the edge is gone: fall through and recreate.

        if status in EntityStatus.inactive():
            return self._delete_relationship(
                store,
                existing=existing,
                edge=edge,
                relationship_type=relationship_type,
                source_type=source_type,
                source_key=source_key,
                target_type=target_type,
                target_key=target_key,
                ref=ref,
                status=status,
                provenance=provenance,
            )

        # A caller must never smuggle ``derived`` into a base edge:
        # the B4 recompute assumes base edges never carry derived=true.
        clean = (
            {k: v for k, v in properties.items() if k != "derived"}
            if properties is not None
            else None
        )
        # Same "no payload" rule as entities: None keeps the edge's
        # current properties instead of clearing them.
        current_user = _user_props(edge, _EDGE_SYSTEM_KEYS) if edge is not None else {}
        effective = dict(clean) if clean is not None else current_user
        if edge is not None and not force_update:
            if effective == current_user and _object_status(edge) == status.value:
                if existing is None:
                    # Edge on Neo4j, no L1 state row: backfill L1.
                    self._save_relationship_state(
                        relationship_type=relationship_type,
                        source_type=source_type,
                        source_key=source_key,
                        target_type=target_type,
                        target_key=target_key,
                        status=_object_status(edge),
                        version=_object_version(edge) or 1,
                        change=ChangeType.CREATE,
                        trace_id=provenance.trace_id,
                        source_event_id=provenance.source_event_id,
                    )
                version = (
                    existing.version
                    if existing is not None
                    else _object_version(edge) or 1
                )
                return CanonicalWriteResult(ref, version, WriteOutcome.NOOP, False)

        # Endpoints must exist on Neo4j before the edge can be written:
        # merge_relationship MATCHes them and silently writes nothing
        # when one is missing. An endpoint that is inactive (or
        # unknown) in L1 makes the edge unprojectable — record the L1
        # row and let the drain drop it (the O20 seed case: an archived
        # asset keeps its L1 edges but they never land on the graph);
        # an endpoint that is active in L1 but missing on Neo4j is a
        # partial failure — fail fast instead of writing a phantom.
        for endpoint_type, endpoint_key in (
            (source_type, source_key),
            (target_type, target_key),
        ):
            if store.get_node(endpoint_type, endpoint_key) is None:
                endpoint = self._repo.get_entity(endpoint_type, endpoint_key)
                if endpoint is None or endpoint.status in EntityStatus.inactive():
                    logger.warning(
                        "Skipping Neo4j write for %s: endpoint %s:%s is %s in L1",
                        ref,
                        endpoint_type,
                        endpoint_key,
                        "missing" if endpoint is None else endpoint.status,
                    )
                    if (
                        existing is not None
                        and existing.status == status.value
                    ):
                        return CanonicalWriteResult(
                            ref, existing.version, WriteOutcome.NOOP, False
                        )
                    version = (
                        max(existing.version if existing else 0, _object_version(edge))
                        + 1
                    )
                    change = _change_type(existing is not None, status)
                    self._save_relationship_state(
                        relationship_type=relationship_type,
                        source_type=source_type,
                        source_key=source_key,
                        target_type=target_type,
                        target_key=target_key,
                        status=status,
                        version=version,
                        change=change,
                        trace_id=provenance.trace_id,
                        source_event_id=provenance.source_event_id,
                    )
                    return CanonicalWriteResult(ref, version, change, True)
                raise RuntimeError(
                    f"Relationship endpoint {endpoint_type}:{endpoint_key} "
                    f"is active in L1 but missing on Neo4j; write the "
                    f"endpoint first (ref {ref})"
                )

        # (3) Neo4j first: drop the derived edges the previous write
        # materialized (same base_ref), MERGE the base edge with its
        # base_ref, then re-derive.
        version = max(existing.version if existing else 0, _object_version(edge)) + 1
        change = _change_type(existing is not None, status)
        new_base_ref = base_ref_for(
            source_type, source_key, relationship_type, target_type, target_key
        )
        old_base_ref = (edge or {}).get("base_ref") or new_base_ref
        if edge is not None:
            remove_derived_for_base(store, base_ref=old_base_ref)
        store.merge_relationship(
            source_type,
            source_key,
            relationship_type,
            target_type,
            target_key,
            {
                **effective,
                "status": status.value,
                "version": version,
                "base_ref": new_base_ref,
            },
        )
        materialize_for_write(
            store,
            relationship_type=relationship_type,
            source_type=source_type,
            source_key=source_key,
            target_type=target_type,
            target_key=target_key,
            base_ref=new_base_ref,
        )

        # (4) L1 state + outbox. On failure the edge is already on
        # Neo4j; the retry with the same source_event_id self-heals.
        self._save_relationship_state(
            relationship_type=relationship_type,
            source_type=source_type,
            source_key=source_key,
            target_type=target_type,
            target_key=target_key,
            status=status,
            version=version,
            change=change,
            trace_id=provenance.trace_id,
            source_event_id=provenance.source_event_id,
        )
        return CanonicalWriteResult(ref, version, change, True)

    def _delete_relationship(
        self,
        store: GraphStore,
        *,
        existing: OntologyRelationship | None,
        edge: dict[str, Any] | None,
        relationship_type: str,
        source_type: str,
        source_key: str,
        target_type: str,
        target_key: str,
        ref: str,
        status: EntityStatus,
        provenance: WriteProvenance,
    ) -> CanonicalWriteResult:
        # Replay: this edge was already deleted under this event id.
        if (
            provenance.source_event_id
            and existing is not None
            and existing.source_event_id == provenance.source_event_id
            and edge is None
        ):
            return CanonicalWriteResult(
                ref, existing.version, WriteOutcome.REPLAY, False, True
            )
        # Already in this inactive status and nothing on Neo4j.
        if (
            edge is None
            and existing is not None
            and existing.status == status.value
        ):
            return CanonicalWriteResult(ref, existing.version, WriteOutcome.NOOP, False)
        version = max(existing.version if existing else 0, _object_version(edge)) + 1
        change = _change_type(existing is not None, status)
        # Drop this base edge's derived edges before deleting it.
        base_ref = (edge or {}).get("base_ref") or base_ref_for(
            source_type, source_key, relationship_type, target_type, target_key
        )
        remove_derived_for_base(store, base_ref=base_ref)
        if edge is not None:
            store.delete_relationship(
                source_type, source_key, relationship_type, target_type, target_key
            )
        self._save_relationship_state(
            relationship_type=relationship_type,
            source_type=source_type,
            source_key=source_key,
            target_type=target_type,
            target_key=target_key,
            status=status,
            version=version,
            change=change,
            trace_id=provenance.trace_id,
            source_event_id=provenance.source_event_id,
        )
        return CanonicalWriteResult(ref, version, change, True)
