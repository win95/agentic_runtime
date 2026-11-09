"""MLflow setup — primary observability: tracing for LangGraph/Deep Agents.

MLflow 3.x is the system of record for traces, evals and prompts. Tracing
is enabled via ``mlflow.langchain.autolog()`` (LangGraph graphs, including
the Deep Agents built on LangGraph, are traced as nested spans: nodes,
LLM calls, tool calls, retriever calls — with latency and token usage when
the upstream returns it).

LangSmith stays enabled separately for dev debugging (see
``infra/langsmith_setup.py``); MLflow does not replace it.

Safety rules for this module:
- Never crash the process: any failure (mlflow not installed, server
  down, bad URI) logs a warning and tracing stays off.
- Never hang the process either: MLflow's REST client retries dead servers
  with exponential backoff, so the HTTP retry/timeout env vars are bounded
  (unless the user set them) before the first server call.
- Idempotent: safe to call from every process entrypoint (API lifespan,
  Celery worker import, ``run_mission`` and the langgraph dev server,
  which imports the graph module). Autolog is installed only once.
- No cost estimation is done here. Cost via the LiteLLM price table is
  deliberately skipped (the user self-hosts LiteLLM calling vLLM and owns
  that setup); only token usage recorded by tracing is kept.
"""

from __future__ import annotations

import logging
import os

from app.config import settings

logger = logging.getLogger(__name__)

_setup_attempted = False
_tracing_enabled = False


def _apply_fail_fast_http_defaults() -> None:
    """Bound MLflow HTTP retries so a dead server fails fast, not after minutes.

    MLflow's REST client retries connection failures with exponential backoff
    (backoff factor 2 by default), so a plain ``set_experiment()`` against an
    unreachable server can block startup for minutes. These env vars are only
    set when the user has not set them already (``setdefault``), and they also
    bound eval/prompt-registry calls in the same process — failing fast is
    preferable to hanging the API/worker on observability.
    """

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")


def setup_mlflow_tracing() -> bool:
    """Enable MLflow tracing for this process. Returns True if enabled."""
    global _setup_attempted, _tracing_enabled
    if _setup_attempted:
        return _tracing_enabled
    _setup_attempted = True
    if not settings.mlflow_tracing_enabled:
        logger.info("MLflow tracing disabled (MLFLOW_TRACING_ENABLED=false).")
        return False
    _apply_fail_fast_http_defaults()
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        import mlflow.langchain

        # run_tracer_inline=True: traces async graph runs too (the
        # langgraph dev server runs graphs with ainvoke()).
        mlflow.langchain.autolog(log_traces=True, run_tracer_inline=True)
    except Exception as exc:  # noqa: BLE001 — tracing must never break the app
        logger.warning(
            "MLflow tracing unavailable (%s); continuing without tracing. "
            "Check MLFLOW_TRACKING_URI=%s and that the MLflow 3.x server is up.",
            exc,
            settings.mlflow_tracking_uri,
        )
        return False
    _tracing_enabled = True
    logger.info(
        "MLflow tracing enabled (uri=%s, experiment=%s)",
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment,
    )
    return True


def is_mlflow_tracing_enabled() -> bool:
    """True when MLflow tracing was successfully enabled in this process."""
    return _tracing_enabled


def reset_mlflow_setup_for_tests() -> None:
    """Reset the once-only guard (tests only)."""
    global _setup_attempted, _tracing_enabled
    _setup_attempted = False
    _tracing_enabled = False
