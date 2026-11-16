"""Best-effort observability — tracing/export must never fail a run."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_OBSERVABILITY_MARKERS = (
    "smith.langchain.com",
    "langsmith",
    "runs/multipart",
    "monthly unique traces",
    "mlflow",
    "tracking uri",
)


def is_observability_transport_error(exc: BaseException) -> bool:
    """True when the exception is only an observability transport failure."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if "429" in text and any(m in text for m in _OBSERVABILITY_MARKERS):
        return True
    return any(m in text for m in _OBSERVABILITY_MARKERS)


def is_observability_error_message(message: str) -> bool:
    text = (message or "").lower()
    if "429" in text and any(m in text for m in _OBSERVABILITY_MARKERS):
        return True
    return any(m in text for m in _OBSERVABILITY_MARKERS)


def config_without_callbacks(config: dict | None) -> dict:
    """Copy of RunnableConfig with callbacks cleared (local retry only)."""
    cfg = dict(config or {})
    cfg["callbacks"] = []
    return cfg


def invoke_with_observability_fallback(invoke, payload, config: dict | None = None):
    """Invoke a Runnable; retry once without callbacks on observability errors."""
    try:
        if config:
            return invoke(payload, config=config)
        return invoke(payload)
    except Exception as exc:  # noqa: BLE001 — narrow retry below
        if not is_observability_transport_error(exc):
            raise
        logger.warning(
            "Observability transport failed; retrying invoke without callbacks: %s",
            exc,
        )
        safe = config_without_callbacks(config)
        return invoke(payload, config=safe)


def filter_observability_errors(errors: list[str] | None) -> list[str]:
    """Drop tracing/export lines from execution errors."""
    if not errors:
        return []
    return [e for e in errors if not is_observability_error_message(e)]