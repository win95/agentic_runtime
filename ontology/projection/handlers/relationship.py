"""Project canonical relationship rows to Neo4j edges."""

from __future__ import annotations

from typing import Any

from ontology.graph.protocol import GraphStore


def project_relationship(
    store: GraphStore,
    *,
    relationship_type: str,
    source_type: str,
    source_key: str,
    target_type: str,
    target_key: str,
    properties: dict[str, Any] | None,
    status: str,
    change_type: str,
) -> None:
    if status in ("deleted", "archived") or change_type == "delete":
        store.delete_relationship(
            source_type, source_key, relationship_type, target_type, target_key
        )
        return
    store.merge_relationship(
        source_type,
        source_key,
        relationship_type,
        target_type,
        target_key,
        properties,
    )
