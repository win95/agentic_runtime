"""Neo4j driver singleton."""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_driver: Any | None = None


def connect_neo4j() -> Any | None:
    global _driver
    if _driver is not None:
        return _driver
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.warning("neo4j package not installed")
        return None
    uri = settings.neo4j_uri
    if not uri:
        return None
    auth = (settings.neo4j_username, settings.neo4j_password)
    _driver = GraphDatabase.driver(uri, auth=auth)
    return _driver


def close_neo4j() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
