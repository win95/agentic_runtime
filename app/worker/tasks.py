"""Celery tasks — run one execution end to end.

`run_execution(execution_id)`:
1. Loads the Execution + Mission snapshot from PostgreSQL.
2. Runs the LangGraph Mission Graph (Layer 3).
3. Persists the result and final status.
4. Retries up to 3 times on transient failure (60s backoff); after the
   last retry the execution is marked FAILED with the error.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app import models
from app.core.db import SessionLocal
from app.services import mission_service
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="app.worker.tasks.run_execution",
    max_retries=3,
    default_retry_delay=60,
)
def run_execution(self, execution_id: str) -> dict:
    db = SessionLocal()
    try:
        execution = db.get(models.Execution, execution_id)
        if execution is None:
            logger.error("Execution %s not found", execution_id)
            return {"execution_id": execution_id, "status": "FAILED", "error": "not found"}
        if execution.status == "CANCELLED":
            logger.info("Execution %s was cancelled before start", execution_id)
            return {"execution_id": execution_id, "status": "CANCELLED"}
        if execution.status not in ("QUEUED", "RUNNING", "FAILED"):
            # Already finished (SUCCEEDED) — idempotent no-op.
            return {"execution_id": execution_id, "status": execution.status}

        execution.status = "RUNNING"
        execution.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()

        mission = db.get(models.Mission, execution.mission_id)
        if mission is None:
            raise RuntimeError(f"Mission {execution.mission_id} not found")
        mission_dict = mission_service.mission_to_dict(mission)

        # Imported here so the worker process (not the API process) pays the
        # LangGraph import cost.
        from infra.langsmith_setup import setup_langsmith
        from runtime.mission_graph import run_mission

        setup_langsmith()  # tracing for the worker process (no-op without key)
        final_state = run_mission(mission_dict, execution.input or {}, execution_id)

        # A cancellation that arrived mid-run wins over the graph result.
        db.refresh(execution)
        if execution.status == "CANCELLED":
            logger.info("Execution %s was cancelled during run", execution_id)
            return {"execution_id": execution_id, "status": "CANCELLED"}

        status = final_state.get("status", "FAILED")
        output = final_state.get("result") or {}
        evidence = final_state.get("evidence") or []
        errors = final_state.get("errors") or []
        if status != "SUCCEEDED":
            output = {**output, "errors": errors}

        # Save the Markdown report: MinIO `reports` bucket when reachable,
        # otherwise inline in the result (see runtime/report.py).
        from infra.minio_client import report_location, upload_report
        from runtime.report import build_markdown_report

        report_markdown = build_markdown_report(
            execution_id, mission_dict, execution.input or {}, final_state
        )
        report_key = f"executions/{execution.id}/report.md"
        try:
            upload_report(report_key, report_markdown)
            output["report_location"] = report_location(report_key)
            output["report_storage"] = "minio"
        except Exception as exc:  # noqa: BLE001 — report storage is best-effort
            logger.warning("Could not store report in MinIO, keeping inline: %s", exc)
            output["report_markdown"] = report_markdown
            output["report_storage"] = "inline"

        db.add(
            models.ExecutionResult(
                id=f"res_{uuid.uuid4().hex[:12]}",
                execution_id=execution.id,
                output=output,
                evidence=evidence,
            )
        )
        execution.status = status
        execution.error = "; ".join(errors) if errors else None
        execution.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.commit()
        logger.info("Execution %s finished with status %s", execution_id, status)
        return {"execution_id": execution_id, "status": status}
    except Exception as exc:
        db.rollback()
        if self.request.retries < self.max_retries:
            logger.warning(
                "Execution %s failed (attempt %d/%d), retrying: %s",
                execution_id,
                self.request.retries + 1,
                self.max_retries,
                exc,
            )
            raise self.retry(exc=exc, countdown=60) from exc
        # Retries exhausted — persist FAILED.
        try:
            execution = db.get(models.Execution, execution_id)
            if execution is not None and execution.status != "CANCELLED":
                execution.status = "FAILED"
                execution.error = str(exc)[:2000]
                execution.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
                db.commit()
        except Exception:  # noqa: BLE001 — last-resort persistence, never mask
            logger.exception("Could not persist FAILED status for %s", execution_id)
        raise
    finally:
        db.close()
