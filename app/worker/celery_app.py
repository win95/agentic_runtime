"""Celery application — Execution Dispatcher (Layer 2).

The dispatcher only moves execution ids through Redis. The worker that
receives a task loads the execution snapshot from PostgreSQL and runs the
LangGraph Mission Graph (Layer 3).
"""

from __future__ import annotations

from celery import Celery

from app.config import settings
from infra.langsmith_setup import setup_langsmith
from infra.mlflow_setup import setup_mlflow_tracing

setup_langsmith()
setup_mlflow_tracing()

celery_app = Celery(
    "agentic_runtime",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_time_limit=settings.execution_timeout_seconds + 300,
    task_soft_time_limit=settings.execution_timeout_seconds,
    result_expires=86400,
    # Beat: every 60s dispatch executions for due schedules (see
    # app.worker.tasks.run_due_schedules). The schedule lives in code;
    # beat's local state file only remembers last-run times, and losing
    # it is harmless — due checks live in PostgreSQL.
    beat_schedule={
        "run-due-schedules-every-minute": {
            "task": "app.worker.tasks.run_due_schedules",
            "schedule": 60.0,
        },
    },
)
