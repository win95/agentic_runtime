"""Backfill Qdrant payload node links for already-indexed documents.

``upsert_documents`` SKIPS existing points, so docs indexed before
Phase 1 never pick up ``node_ids``. This scrolls every point and
updates the payload only (``set_payload`` — no re-embedding).

Live usage (user's machine, Qdrant up):
    python -m scripts.backfill_qdrant_links

Fake usage (this VM, no Qdrant):
    python -m scripts.backfill_qdrant_links --fake
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Qdrant payload node links.")
    parser.add_argument("--fake", action="store_true",
                        help="Run against the in-memory fake client (no Qdrant).")
    args = parser.parse_args()

    from infra.qdrant_client import backfill_document_links

    if args.fake:
        from runtime.retrieval_baseline import DeterministicFakeEmbeddings, FakeQdrantClient

        client = FakeQdrantClient(DeterministicFakeEmbeddings())
        result = backfill_document_links(client=client)
    else:
        result = backfill_document_links()
    print(
        f"Backfill: scanned={result['scanned']} updated={result['updated']} "
        f"unchanged={result['unchanged']} unknown={result['unknown']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
