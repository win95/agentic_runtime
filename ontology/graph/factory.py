"""Build graph store from settings."""

from __future__ import annotations

from ontology.graph.neo4j_store import Neo4jGraphStore
from ontology.graph.protocol import GraphStore


def build_graph_store() -> GraphStore:
    return Neo4jGraphStore()
