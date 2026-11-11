"""Build graph store from settings."""

from __future__ import annotations

from app.config import settings
from ontology.graph.neo4j_store import Neo4jGraphStore
from ontology.graph.protocol import GraphStore

_ALLOWED_BACKENDS = ("neo4j",)


def build_graph_store() -> GraphStore:
    backend = (settings.graph_database_backend or "").strip().lower()
    if backend == "neo4j":
        return Neo4jGraphStore()
    raise ValueError(
        f"Unsupported graph_database_backend {backend!r}; "
        f"allowed values: {', '.join(_ALLOWED_BACKENDS)}"
    )
