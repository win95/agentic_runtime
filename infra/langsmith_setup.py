"""LangSmith setup — tracing is observability only (Layer 6).

Tracing is a no-op unless BOTH `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY` are set. LangSmith never stores execution state;
PostgreSQL remains the source of truth.
"""

from __future__ import annotations

import logging
import os

from app.config import settings

logger = logging.getLogger(__name__)


def setup_langsmith() -> bool:
    """Enable LangSmith tracing for this process. Returns True if enabled."""
    if not settings.langsmith_tracing:
        return False
    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty; tracing off.")
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    return True
