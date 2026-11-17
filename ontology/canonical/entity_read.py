"""Read helpers for canonical entities — objects live on Neo4j now.

Design §6: the object (properties + status + version) is read from
the Neo4j node, not from L1 — L1 keeps only state and is consulted
for idempotency/audit, never for object content. A node that was
archived/deleted has been DETACH DELETEd, so it reads as missing
(404); a node that still carries an inactive ``status`` property
(e.g. written by the backfill before the drain converged it) is
hidden the same way.

``list_entities`` is gone: it had no callers left, and a list over
Neo4j belongs to the query API (B5), not to this helper.
"""

from __future__ import annotations

from typing import Any

from ontology.canonical.schema import EntityStatus
from ontology.graph.protocol import GraphStore

# Node properties owned by the store layer, not by the caller.
_SYSTEM_KEYS = frozenset({"id", "status", "version", "updated_at"})


def get_entity(store: GraphStore, entity_type: str, entity_key: str) -> dict[str, Any] | None:
    """Canonical read for the API: only ``active`` entities are visible.

    Returns ``{entity_type, entity_key, properties, status, version}``
    (no ``state_hash`` anymore) or None when the node does not exist
    or is inactive.
    """
    node = store.get_node(entity_type, entity_key)
    if node is None:
        return None
    status = str(node.get("status") or EntityStatus.ACTIVE.value)
    if status in EntityStatus.inactive():
        return None
    try:
        version = int(node.get("version") or 1)
    except (TypeError, ValueError):
        version = 1
    return {
        "entity_type": entity_type,
        "entity_key": entity_key,
        "properties": {k: v for k, v in node.items() if k not in _SYSTEM_KEYS},
        "status": status,
        "version": version,
    }
