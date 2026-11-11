"""GraphStore protocol for ontology projection."""

from __future__ import annotations

from typing import Any, Protocol


class GraphStore(Protocol):
    def is_healthy(self) -> bool: ...

    def run_query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...

    def merge_node(self, label: str, node_id: str, properties: dict[str, Any]) -> None: ...

    def delete_node(self, label: str, node_id: str) -> None: ...

    def merge_relationship(
        self,
        source_label: str,
        source_id: str,
        rel_type: str,
        target_label: str,
        target_id: str,
        properties: dict[str, Any] | None,
    ) -> None: ...

    def delete_relationship(
        self,
        source_label: str,
        source_id: str,
        rel_type: str,
        target_label: str,
        target_id: str,
    ) -> None: ...

    def close(self) -> None: ...
