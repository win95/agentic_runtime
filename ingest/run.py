"""Batch runner (Option B).

One run processes every ``*.json`` file in the inbox: the file name
decides the row type (``faults.json``, ``faults-2026-11.json``, ...),
each row is validated (``ingest/schemas.py``), valid rows are loaded
idempotently (``ingest/loaders.py``), and failed rows go to quarantine
instead of being dropped.

Watermark: one ``IngestRun`` row per file. A re-run skips files whose
last run SUCCEEDED (``--force`` reprocesses them anyway). Quarantined
rows can be fixed in place and re-run with ``--retry``.

Exit code: 0 when every file SUCCEEDED or PARTIAL (quarantine is a
normal outcome); 1 when any file FAILED (unreadable / unparsable /
unknown type).

Usage:
    python -m ingest.run [--inbox DIR | --minio [--bucket NAME]]
                         [--file PATH] [--force] [--dry-run] [--retry]
    make ingest            # inbox ingest/inbox, local dir
    make ingest-dry-run    # validate only, touch nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import sys
from datetime import datetime, timezone

from pydantic import ValidationError

from app.config import settings
from app.core.db import SessionLocal
from app.models import IngestQuarantine, IngestRun
from ingest import FILE_TYPES
from ingest.loaders import LOADERS
from ingest.schemas import ROW_MODELS
from ingest.sources import LocalDirSource, MinioInboxSource

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def file_type(filename: str) -> str | None:
    """Row type from the file name: the stem before the first ``-``."""
    stem = filename.rsplit(".", 1)[0]
    candidate = stem.split("-", 1)[0]
    return candidate if candidate in FILE_TYPES else None


def _new_run_id() -> str:
    return f"ing_{secrets.token_hex(6)}"


def _already_succeeded(db, source: str) -> bool:
    return (
        db.query(IngestRun)
        .filter_by(source=source, status="SUCCEEDED")
        .first()
        is not None
    )


def _record_run(db, *, source: str, status: str, rows_ok: int, rows_failed: int,
                errors: list[str], started_at: datetime) -> IngestRun:
    run = IngestRun(
        id=_new_run_id(),
        source=source,
        status=status,
        rows_ok=rows_ok,
        rows_failed=rows_failed,
        errors=errors,
        started_at=started_at,
        finished_at=_utcnow(),
    )
    db.add(run)
    db.commit()
    return run


def _quarantine(db, run: IngestRun, source: str, failures: list[dict]) -> None:
    for failure in failures:
        db.add(IngestQuarantine(
            id=f"inq_{secrets.token_hex(6)}",
            run_id=run.id,
            source=source,
            row=failure["row"],
            error=str(failure["error"])[:2000],
            status="pending",
        ))
    if failures:
        db.commit()


def process_file(db, name: str, content: bytes, *, force: bool = False,
                 dry_run: bool = False) -> dict:
    """Process one inbox file. Returns its summary dict.

    Dry-run validates only (parse + schema) and never touches the DB —
    not even the watermark check — so it works with no store at all.
    """
    started_at = _utcnow()
    result = {"source": name, "status": "SKIPPED", "rows_ok": 0, "rows_failed": 0}

    row_type = file_type(name)
    if row_type is None:
        if dry_run:
            return {"source": name, "status": "FAILED", "rows_ok": 0, "rows_failed": 0}
        run = _record_run(db, source=name, status="FAILED", rows_ok=0,
                          rows_failed=0,
                          errors=[f"unknown file type for {name!r} "
                                  f"(expected one of {', '.join(FILE_TYPES)})"],
                          started_at=started_at)
        return {"source": name, "status": "FAILED", "rows_ok": 0,
                "rows_failed": 0, "run_id": run.id}

    try:
        parsed = json.loads(content)
    except (ValueError, UnicodeDecodeError) as exc:
        if dry_run:
            return {"source": name, "status": "FAILED", "rows_ok": 0, "rows_failed": 0}
        run = _record_run(db, source=name, status="FAILED", rows_ok=0,
                          rows_failed=0, errors=[f"invalid JSON: {exc}"],
                          started_at=started_at)
        return {"source": name, "status": "FAILED", "rows_ok": 0,
                "rows_failed": 0, "run_id": run.id}
    if not isinstance(parsed, list):
        if dry_run:
            return {"source": name, "status": "FAILED", "rows_ok": 0, "rows_failed": 0}
        run = _record_run(db, source=name, status="FAILED", rows_ok=0,
                          rows_failed=0,
                          errors=["file must contain a JSON array of rows"],
                          started_at=started_at)
        return {"source": name, "status": "FAILED", "rows_ok": 0,
                "rows_failed": 0, "run_id": run.id}

    model = ROW_MODELS[row_type]
    valid = []
    failures: list[dict] = []
    for raw in parsed:
        if not isinstance(raw, dict):
            failures.append({"row": raw, "error": "row must be a JSON object"})
            continue
        try:
            valid.append(model.model_validate(raw))
        except ValidationError as exc:
            failures.append({"row": raw, "error": str(exc)[:2000]})

    if dry_run:
        return {"source": name, "status": "DRY_RUN", "rows_ok": len(valid),
                "rows_failed": len(failures)}

    if not force and _already_succeeded(db, name):
        return result

    loader = LOADERS[row_type]
    if row_type in ("faults", "maintenance", "sensor_readings"):
        ok, load_failures = loader(db, valid)
    else:
        ok, load_failures = loader(valid)
    failures.extend(load_failures)

    status = "SUCCEEDED" if not failures else "PARTIAL"
    run = _record_run(db, source=name, status=status, rows_ok=ok,
                      rows_failed=len(failures), errors=[],
                      started_at=started_at)
    _quarantine(db, run, name, failures)
    logger.info("Ingest %s: %s (%d ok, %d failed)", name, status, ok, len(failures))
    return {"source": name, "status": status, "rows_ok": ok,
            "rows_failed": len(failures), "run_id": run.id}


def run_batch(*, source=None, db=None, force: bool = False,
              dry_run: bool = False) -> dict:
    """Process every file in the inbox. Returns a batch summary."""
    if source is None:
        source = LocalDirSource(settings.ingest_inbox_dir)
    own_db = db is None
    if own_db:
        db = SessionLocal()
    try:
        files = source.list_files()
        results = [process_file(db, name, content, force=force, dry_run=dry_run)
                   for name, content in files]
        summary = {
            "files": len(files),
            "processed": sum(1 for r in results if r["status"] != "SKIPPED"),
            "skipped": [r["source"] for r in results if r["status"] == "SKIPPED"],
            "failed": [r["source"] for r in results if r["status"] == "FAILED"],
            "rows_ok": sum(r["rows_ok"] for r in results),
            "rows_failed": sum(r["rows_failed"] for r in results),
            "runs": results,
        }
        return summary
    finally:
        if own_db:
            db.close()


def retry_quarantine(db, *, limit: int = 100) -> dict:
    """Re-run pending quarantined rows (after fixing them in place).

    A row that loads now is marked ``retried``; one that still fails
    stays ``pending`` with its error updated.
    """
    pending = (
        db.query(IngestQuarantine)
        .filter_by(status="pending")
        .order_by(IngestQuarantine.created_at)
        .limit(limit)
        .all()
    )
    retried = 0
    for item in pending:
        row_type = file_type(item.source)
        if row_type is None:
            item.error = f"unknown file type for {item.source!r}"
            continue
        try:
            row = ROW_MODELS[row_type].model_validate(item.row)
        except ValidationError as exc:
            item.error = str(exc)[:2000]
            continue
        loader = LOADERS[row_type]
        try:
            if row_type in ("faults", "maintenance", "sensor_readings"):
                ok, failures = loader(db, [row])
            else:
                ok, failures = loader([row])
        except Exception as exc:  # noqa: BLE001 — stays pending
            item.error = str(exc)[:2000]
            continue
        if failures:
            item.error = str(failures[0]["error"])[:2000]
            continue
        if ok:
            item.status = "retried"
            retried += 1
    db.commit()
    still_pending = db.query(IngestQuarantine).filter_by(status="pending").count()
    return {"retried": retried, "still_pending": still_pending}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch ingest (Option B).")
    parser.add_argument("--inbox", default=None,
                        help="local inbox directory (default: ingest/inbox)")
    parser.add_argument("--minio", action="store_true",
                        help="read the inbox from the MinIO ingest-inbox bucket")
    parser.add_argument("--bucket", default=None,
                        help="MinIO inbox bucket (default: ingest-inbox)")
    parser.add_argument("--file", default=None,
                        help="process one local file by path instead of the inbox")
    parser.add_argument("--force", action="store_true",
                        help="reprocess files that already succeeded")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate only; write nothing to any store")
    parser.add_argument("--retry", action="store_true",
                        help="re-run pending quarantined rows instead of the inbox")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.retry:
        db = SessionLocal()
        try:
            summary = retry_quarantine(db)
        finally:
            db.close()
        print(f"Retry: {summary['retried']} retried, "
              f"{summary['still_pending']} still pending.")
        return 0

    if args.file:
        from pathlib import Path

        path = Path(args.file)
        if not path.is_file():
            print(f"File not found: {path}")
            return 1
        db = SessionLocal()
        try:
            result = process_file(db, path.name, path.read_bytes(),
                                  force=args.force, dry_run=args.dry_run)
        finally:
            db.close()
        print(f"{result['source']}: {result['status']} "
              f"({result['rows_ok']} ok, {result['rows_failed']} failed).")
        return 0 if result["status"] != "FAILED" else 1

    if args.minio:
        source = MinioInboxSource(args.bucket or settings.ingest_minio_bucket)
    else:
        source = LocalDirSource(args.inbox or settings.ingest_inbox_dir)
    summary = run_batch(source=source, force=args.force, dry_run=args.dry_run)
    print(f"Ingest: {summary['processed']} file(s) processed, "
          f"{len(summary['skipped'])} skipped, "
          f"{summary['rows_ok']} rows ok, {summary['rows_failed']} rows failed.")
    for run in summary["runs"]:
        if run["status"] != "SKIPPED":
            print(f"  {run['source']}: {run['status']} "
                  f"({run['rows_ok']} ok, {run['rows_failed']} failed)")
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
