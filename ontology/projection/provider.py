"""GraphProvider — lazy singleton around GraphStore."""

from __future__ import annotations

from ontology.graph.factory import build_graph_store
from ontology.graph.protocol import GraphStore

_store: GraphStore | None = None


def get_graph_store() -> GraphStore:
    global _store
    if _store is None:
        _store = build_graph_store()
    return _store


def reset_graph_store_for_tests() -> None:
    global _store
    _store = None
