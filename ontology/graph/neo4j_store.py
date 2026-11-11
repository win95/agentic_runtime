"""Neo4jGraphStore — sync projection adapter."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import settings
from ontology.canonical.schema import ENTITY_TYPES_V1
from ontology.graph import neo4j_driver

logger = logging.getLogger(__name__)

_CYPHER_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _assert_identifier(name: str, field: str) -> str:
    if not _CYPHER_IDENTIFIER_RE.match(name or ""):
        raise ValueError(f"Invalid Cypher identifier for {field}: {name!r}")
    return name


class Neo4jGraphStore:
    DB_MODE = "neo4j"

    def __init__(self, driver: Any | None = None) -> None:
        if driver is not None:
            self._driver = driver
            self._owns = False
        else:
            opened = neo4j_driver.connect_neo4j()
            if opened is None:
                raise RuntimeError("Neo4j driver unavailable")
            self._driver = opened
            self._owns = True
        self._database = settings.neo4j_database or "neo4j"
        try:
            self.ensure_constraints()
        except Exception as exc:  # noqa: BLE001
            # A Neo4j outage must never crash startup; constraints are
            # retried on the next store build.
            logger.warning("Neo4j ensure_constraints failed: %s", exc)

    def ensure_constraints(self) -> None:
        """Uniqueness constraint on (label, id) — one per v1 entity type.

        Uses the Neo4j 5 form ``CREATE CONSTRAINT IF NOT EXISTS FOR
        (n:Label) REQUIRE n.id IS UNIQUE``; IF NOT EXISTS makes repeat
        calls a no-op. Raises on failure — callers that must not crash
        (e.g. __init__) catch it.
        """
        for label in sorted(ENTITY_TYPES_V1):
            label = _assert_identifier(label, "label")
            cypher = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            self.run_query(cypher)

    @property
    def db_mode(self) -> str:
        return self.DB_MODE

    def is_healthy(self) -> bool:
        try:
            with self._driver.session(database=self._database) as session:
                row = session.run("RETURN 1 AS n").single()
                return row is not None and row["n"] == 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j health check failed: %s", exc)
            return False

    def run_query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = params or {}
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, params)
            return [dict(record) for record in result]

    def merge_node(self, label: str, node_id: str, properties: dict[str, Any]) -> None:
        label = _assert_identifier(label, "label")
        props = dict(properties or {})
        props["id"] = node_id
        # Full replace (SET n = $props): stale properties from an older
        # version are removed instead of lingering under +=.
        cypher = f"MERGE (n:{label} {{id: $id}}) SET n = $props, n.updated_at = timestamp()"
        self.run_query(cypher, {"id": node_id, "props": props})

    def node_exists(self, label: str, node_id: str) -> bool:
        label = _assert_identifier(label, "label")
        rows = self.run_query(
            f"MATCH (n:{label} {{id: $id}}) RETURN n.id AS id LIMIT 1",
            {"id": node_id},
        )
        return len(rows) > 0

    def delete_node(self, label: str, node_id: str) -> None:
        label = _assert_identifier(label, "label")
        cypher = f"MATCH (n:{label} {{id: $id}}) DETACH DELETE n"
        self.run_query(cypher, {"id": node_id})

    def merge_relationship(
        self,
        source_label: str,
        source_id: str,
        rel_type: str,
        target_label: str,
        target_id: str,
        properties: dict[str, Any] | None,
    ) -> None:
        source_label = _assert_identifier(source_label, "source_label")
        target_label = _assert_identifier(target_label, "target_label")
        rel_type = _assert_identifier(rel_type, "rel_type")
        props = dict(properties or {})
        # MATCH (not MERGE) the endpoints: a missing endpoint means no
        # rows, so the edge MERGE never runs and no stub node is created.
        # The projection engine checks endpoints first and decides
        # delete-vs-retry; this MATCH is the store-level guarantee.
        cypher = (
            f"MATCH (a:{source_label} {{id: $source_id}}) "
            f"MATCH (b:{target_label} {{id: $target_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r = $props"
        )
        self.run_query(
            cypher,
            {"source_id": source_id, "target_id": target_id, "props": props},
        )

    def delete_relationship(
        self,
        source_label: str,
        source_id: str,
        rel_type: str,
        target_label: str,
        target_id: str,
    ) -> None:
        source_label = _assert_identifier(source_label, "source_label")
        target_label = _assert_identifier(target_label, "target_label")
        rel_type = _assert_identifier(rel_type, "rel_type")
        cypher = (
            f"MATCH (a:{source_label} {{id: $source_id}})-[r:{rel_type}]->"
            f"(b:{target_label} {{id: $target_id}}) DELETE r"
        )
        self.run_query(cypher, {"source_id": source_id, "target_id": target_id})

    def close(self) -> None:
        if self._owns:
            neo4j_driver.close_neo4j()
