"""Neo4j graph client — read path for missions and ingest."""

from __future__ import annotations

import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)

_CYPHER_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _assert_label(label: str) -> str:
    if not _CYPHER_IDENTIFIER_RE.match(label or ""):
        raise ValueError(f"Invalid graph label: {label!r}")
    return label


def query_entity_relationships(
    entity_type: str,
    entity_key: str,
    *,
    limit: int = 20,
) -> list[dict]:
    """Return [{from, type, to}] for one entity."""
    from ontology.knowledge.graph_query import neighbors

    try:
        rows = neighbors(entity_type, entity_key, depth=1, limit=limit)
    except Exception as exc:  # noqa: BLE001
        if settings.graph_features_enabled:
            raise
        logger.warning("Graph query skipped (graph disabled): %s", exc)
        return []
    return [
        {"from": r.get("from"), "type": r.get("type"), "to": r.get("to")}
        for r in rows
    ]


def query_asset_relationships(asset_id: str, limit: int = 20) -> list[dict]:
    """Backward-compatible Asset helper for mission runtime."""
    return query_entity_relationships("Asset", asset_id, limit=limit)


def is_graph_healthy() -> bool:
    if not settings.graph_features_enabled:
        return False
    try:
        from ontology.projection.provider import get_graph_store

        return get_graph_store().is_healthy()
    except Exception:  # noqa: BLE001
        return False
