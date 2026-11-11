"""Execution tracing context: LangSmith metadata/tags + MLflow run tags.

Session traceability (docs/sessions.md): every execution carries its
``session_id`` into the observability layer, so one session's turns can
be followed across LangSmith and MLflow.

- LangSmith: the values built here go into the LangChain ``RunnableConfig``
  (``metadata`` + ``tags``) of both ``graph.invoke`` (``run_mission``) and
  the Deep Agent invoke (``run_deep_agent``). LangChain propagates config
  metadata/tags to child runs, so the whole trace tree of an execution
  carries them. Filter in the LangSmith UI by metadata ``session_id`` or
  by the tag ``session:<id>``.
- MLflow: ``mlflow.langchain.autolog`` traces carry no session tag of
  their own, but a trace created inside an active run is linked to it
  (``trace_metadata["mlflow.sourceRun"] == run_id`` — verified against
  MLflow 3.18.0 with a sqlite tracking store). ``mlflow_execution_run``
  therefore opens one nested run per execution, tagged with
  ``session_id`` / ``execution_id`` / ``mission_id`` / ``agent_id``.
  In the MLflow UI, search runs by ``tags.session_id``; the run's traces
  follow via the sourceRun link.

Tracing is observability only and must never break a run: the MLflow
wrapper is best-effort (tracing off, mlflow missing, or the tracking
server unreachable all degrade to a plain invoke with a warning).
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator

from infra.mlflow_setup import is_mlflow_tracing_enabled

logger = logging.getLogger(__name__)


def build_run_metadata(
    execution_id: str,
    mission_id: str,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, str]:
    """LangSmith run metadata for one execution.

    Only string values (LangSmith metadata must be JSON-serializable);
    absent values are omitted rather than sent as empty strings.
    """
    metadata: dict[str, str] = {"execution_id": execution_id}
    if mission_id:
        metadata["mission_id"] = mission_id
    if agent_id:
        metadata["agent_id"] = agent_id
    if session_id:
        metadata["session_id"] = session_id
    return metadata


def build_run_tags(
    mission_id: str,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> list[str]:
    """LangSmith run tags for one execution (``key:value`` form)."""
    tags: list[str] = []
    if mission_id:
        tags.append(f"mission:{mission_id}")
    if agent_id:
        tags.append(f"agent:{agent_id}")
    if session_id:
        tags.append(f"session:{session_id}")
    return tags


@contextlib.contextmanager
def mlflow_execution_run(
    execution_id: str,
    mission_id: str,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> Iterator[object | None]:
    """Open one nested MLflow run per execution, tagged for traceability.

    Yields the active run, or ``None`` when MLflow tracing is off or the
    run could not be started. Never raises: a tracing failure degrades
    to an untagged execution (autolog traces still land in the
    experiment, just without the session link).
    """
    if not is_mlflow_tracing_enabled():
        yield None
        return
    try:
        import mlflow
    except Exception as exc:  # noqa: BLE001 — tracing must never break the app
        logger.warning("MLflow unavailable (%s); running without run tags.", exc)
        yield None
        return
    active = None
    try:
        active = mlflow.start_run(
            run_name=f"execution-{execution_id}", nested=True
        )
        run = active.__enter__()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not start MLflow execution run for %s (%s); "
            "continuing without run tags.",
            execution_id,
            exc,
        )
        yield None
        return
    try:
        try:
            mlflow.set_tag("execution_id", execution_id)
            mlflow.set_tag("mission_id", mission_id)
            if agent_id:
                mlflow.set_tag("agent_id", agent_id)
            if session_id:
                mlflow.set_tag("session_id", session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not tag MLflow execution run for %s (%s); "
                "the run still exists under its name.",
                execution_id,
                exc,
            )
        yield run
    finally:
        # End the run by hand instead of `with mlflow.start_run(...)` so
        # a teardown failure (tracking server down) cannot mask the
        # execution's own error or result.
        try:
            active.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MLflow end_run failed for %s (%s); ignoring.", execution_id, exc
            )
