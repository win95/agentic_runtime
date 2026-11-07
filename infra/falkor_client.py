"""FalkorDB client — knowledge graph / entity relationships (Layer 6)."""

from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger(__name__)

_graph = None


def get_graph():
    """Lazily select the `knowledge` graph, or raise if unreachable."""
    global _graph
    if _graph is not None:
        return _graph
    from falkordb import FalkorDB

    db = FalkorDB(host=settings.falkor_host, port=settings.falkor_port)
    _graph = db.select_graph("knowledge")
    return _graph


def query_asset_relationships(asset_id: str, limit: int = 20) -> list[dict]:
    """Return [{from, type, to}] relationships for one asset."""
    graph = get_graph()
    result = graph.query(
        "MATCH (a:Asset {id: $asset_id})-[r]-(b) "
        "RETURN a.id AS from_id, type(r) AS rel_type, b.id AS to_id "
        "LIMIT $limit",
        {"asset_id": asset_id, "limit": limit},
    )
    return [
        {"from": row[0], "type": row[1], "to": row[2]} for row in result.result_set
    ]
