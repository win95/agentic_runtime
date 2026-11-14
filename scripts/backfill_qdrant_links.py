"""Backfill Qdrant payload node links for already-indexed documents.

``upsert_documents`` SKIPS existing points, so docs indexed before
Phase 1 never pick up ``node_ids``. This scrolls every point and
updates the payload only (``set_payload`` — no re-embedding).

Live usage (user's machine, Qdrant up):
    python -m scripts.backfill_qdrant_links
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from infra.qdrant_client import backfill_document_links

    result = backfill_document_links()
    print(
        f"Backfill: scanned={result['scanned']} updated={result['updated']} "
        f"unchanged={result['unchanged']} unknown={result['unknown']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
