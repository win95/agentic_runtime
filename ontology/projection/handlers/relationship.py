"""Reconcile one canonical relationship state row against its Neo4j edge.

Same contract as ``project_entity``: converge status/version on the
edge, delete it when L1 says inactive, raise
``UnrecoverableDriftError`` when L1 says active but the edge is gone.
The missing-edge + endpoint-gone case (an endpoint was deleted in
L1, so the edge is invalid rather than drifted) is decided by the
engine, which owns the L1 reads — this handler only sees edges whose
endpoints are still valid.
"""

from __future__ import annotations

from ontology.canonical.schema import ChangeType, EntityStatus
from ontology.graph.protocol import GraphStore
from ontology.projection.handlers import UnrecoverableDriftError
from ontology.reasoning import base_ref_for

_EDGE_SYSTEM_KEYS = frozenset({"status", "version", "base_ref", "derived", "rule"})


def project_relationship(
    store: GraphStore,
    *,
    relationship_type: str,
    source_type: str,
    source_key: str,
    target_type: str,
    target_key: str,
    status: str | EntityStatus,
    version: int,
    change_type: str | ChangeType,
) -> None:
    if status in EntityStatus.inactive() or change_type in (
        ChangeType.ARCHIVED,
        ChangeType.DELETED,
    ):
        edge = store.get_relationship(
            source_type, source_key, relationship_type, target_type, target_key
        )
        if edge is not None:
            store.delete_relationship(
                source_type, source_key, relationship_type, target_type, target_key
            )
        return
    edge = store.get_relationship(
        source_type, source_key, relationship_type, target_type, target_key
    )
    if edge is None:
        raise UnrecoverableDriftError(
            f"Relationship {relationship_type} {source_type}:{source_key} -> "
            f"{target_type}:{target_key} is active in L1 (version "
            f"{version}) but missing on Neo4j; it cannot be recreated "
            f"from L1 state — restore from the Neo4j backup"
        )
    current_status = str(edge.get("status") or EntityStatus.ACTIVE.value)
    try:
        current_version = int(edge.get("version") or 0)
    except (TypeError, ValueError):
        current_version = 0
    if current_status == str(status) and current_version == version:
        return  # converged already — nothing to write
    props = {k: v for k, v in edge.items() if k not in _EDGE_SYSTEM_KEYS}
    props["status"] = str(status)
    props["version"] = version
    props["base_ref"] = edge.get("base_ref") or base_ref_for(
        source_type, source_key, relationship_type, target_type, target_key
    )
    store.merge_relationship(
        source_type,
        source_key,
        relationship_type,
        target_type,
        target_key,
        props,
    )
