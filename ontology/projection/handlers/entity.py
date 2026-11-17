"""Reconcile one canonical entity state row against its Neo4j node.

The drain no longer copies objects (L1 has none): it converges
state. L1 active + node present with a drifted status/version → SET
them back to the L1 values; L1 inactive + node present → delete the
node; L1 active + node missing → unrecoverable drift (the object
cannot be rebuilt from L1), raise ``UnrecoverableDriftError``.
"""

from __future__ import annotations

from ontology.canonical.schema import ChangeType, EntityStatus
from ontology.graph.protocol import GraphStore
from ontology.projection.handlers import UnrecoverableDriftError

_NODE_SYSTEM_KEYS = frozenset({"id", "status", "version", "updated_at"})


def project_entity(
    store: GraphStore,
    *,
    entity_type: str,
    entity_key: str,
    status: str | EntityStatus,
    version: int,
    change_type: str | ChangeType,
) -> None:
    if status in EntityStatus.inactive() or change_type in (
        ChangeType.ARCHIVED,
        ChangeType.DELETED,
    ):
        node = store.get_node(entity_type, entity_key)
        if node is not None:
            store.delete_node(entity_type, entity_key)
        return
    node = store.get_node(entity_type, entity_key)
    if node is None:
        raise UnrecoverableDriftError(
            f"Entity {entity_type}:{entity_key} is active in L1 "
            f"(version {version}) but missing on Neo4j; it cannot be "
            f"recreated from L1 state — restore from the Neo4j backup"
        )
    current_status = str(node.get("status") or EntityStatus.ACTIVE.value)
    try:
        current_version = int(node.get("version") or 0)
    except (TypeError, ValueError):
        current_version = 0
    if current_status == str(status) and current_version == version:
        return  # converged already — nothing to write
    props = {k: v for k, v in node.items() if k not in _NODE_SYSTEM_KEYS}
    props["status"] = str(status)
    props["version"] = version
    store.merge_node(entity_type, entity_key, props)
