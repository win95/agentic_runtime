"""Inbox sources for batch ingest (Option B).

Two sources, one interface — ``list_files()`` returns
``[(name, content_bytes)]`` sorted by name:

- :class:`LocalDirSource`: ``*.json`` files in a directory. This is the
  default and what ``make ingest`` uses.
- :class:`MinioInboxSource`: objects in the ``ingest-inbox`` bucket.
  Files stay in the bucket after a run; the watermark (one
  ``IngestRun`` per file, see ``ingest/run.py``) is what stops a
  re-run from reprocessing them, so clean the bucket up on your own
  retention schedule.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalDirSource:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def list_files(self) -> list[tuple[str, bytes]]:
        if not self.directory.is_dir():
            logger.warning("Ingest inbox %s does not exist — nothing to do", self.directory)
            return []
        out = []
        for path in sorted(self.directory.glob("*.json")):
            out.append((path.name, path.read_bytes()))
        return out


class MinioInboxSource:
    def __init__(self, bucket: str) -> None:
        self.bucket = bucket

    def list_files(self) -> list[tuple[str, bytes]]:
        from infra.minio_client import get_client

        client = get_client()
        out = []
        try:
            objects = client.list_objects(self.bucket, recursive=True)
        except Exception as exc:  # noqa: BLE001 — no bucket, no files
            logger.warning("Could not list ingest bucket %s: %s", self.bucket, exc)
            return []
        for obj in sorted(objects, key=lambda o: o.object_name):
            if not obj.object_name.endswith(".json"):
                continue
            response = client.get_object(self.bucket, obj.object_name)
            try:
                out.append((obj.object_name, response.read()))
            finally:
                response.close()
                response.release_conn()
        return out
