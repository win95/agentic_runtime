"""Project canonical entity rows to Neo4j nodes."""

from __future__ import annotations

from typing import Any

from ontology.graph.protocol import GraphStore


def project_entity(
    store: GraphStore,
    *,
    entity_type: str,
    entity_key: str,
    properties: dict[str, Any] | None,
    status: str,
    change_type: str,
) -> None:
    if status in ("deleted", "archived") or change_type == "delete":
        store.delete_node(entity_type, entity_key)
        return
    store.merge_node(entity_type, entity_key, properties or {})
