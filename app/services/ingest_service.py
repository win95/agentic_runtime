"""Batch ingest service (Option B) — thin wrapper over ``ingest.run``.

The Celery beat task calls this so the batch logic stays testable
without Celery, mirroring ``schedule_service.run_due_schedules``.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import settings
from ingest.run import run_batch
from ingest.sources import LocalDirSource, MinioInboxSource


def run_ingest_batch(db: Session, *, inbox: str | None = None,
                     use_minio: bool | None = None, bucket: str | None = None,
                     force: bool = False, dry_run: bool = False) -> dict:
    """Run one ingest batch from the configured inbox. Returns a summary.

    Defaults come from settings (``INGEST_INBOX_DIR`` /
    ``INGEST_USE_MINIO`` / ``INGEST_MINIO_BUCKET``); explicit arguments
    win. No retries here: beat re-fires on the schedule anyway, and the
    watermark makes a re-run skip what already succeeded.
    """
    if use_minio is None:
        use_minio = settings.ingest_use_minio
    if use_minio:
        source = MinioInboxSource(bucket or settings.ingest_minio_bucket)
    else:
        source = LocalDirSource(inbox or settings.ingest_inbox_dir)
    return run_batch(source=source, db=db, force=force, dry_run=dry_run)
