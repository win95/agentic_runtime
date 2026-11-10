"""Batch ingest (Option B) — one package, no new API surface.

Data arrives as JSON files, one type per file, dropped into an inbox
(a local directory, or the MinIO ``ingest-inbox`` bucket). A Celery
beat task runs the batch on a schedule; ``python -m ingest.run`` runs
it by hand. Every row is validated here (Pydantic + the Layer 1 rules
from ``runtime/security.py``) before any loader touches a store.
"""

FILE_TYPES = ("faults", "maintenance", "documents", "relations", "sensor_readings")
