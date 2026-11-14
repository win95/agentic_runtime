"""Deterministic plain-vector baseline retriever — Phase 0 (report §8).

A stand-in for the real vector path, built entirely from fakes so it
runs in this VM with no LiteLLM proxy, no Qdrant server and no live
embedding model:

- ``DeterministicFakeEmbeddings``: bag-of-words vectors over a fixed
  vocabulary taken from the seeded document texts only (never from
  the golden-set questions — the baseline must not be tuned to the
  questions it is scored on). L2-normalised, so cosine is a dot
  product. This is a lexical-overlap proxy for a real embedding
  model, not a measurement of one.
- ``FakeQdrantClient``: in-memory points for the 7 seeded documents,
  answering ``get_collection`` / ``query_points`` with cosine
  similarity, ties broken by document id for determinism.
- ``build_baseline_retriever``: wires the two fakes into the REAL
  ``infra.qdrant_client.search_documents`` (injected embeddings +
  client), so the baseline exercises the production search code path
  rather than reimplementing it.

The returned retriever is ``question -> (ranked doc ids, trace)``,
ready for ``runtime.retrieval_harness.evaluate_retriever``. It
retrieves ``limit=20`` (the whole 7-doc corpus) so Recall@20 is
measured on the full pool; Recall@6 / nDCG@6 / MRR are the
discriminating metrics on a corpus this small.
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

from seeds.demo_data import DOCUMENTS

# Small stopword list so the BoW vectors are about content words, not
# "the / and / is". Deliberately short and generic — not tuned to the
# golden set.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "at", "from", "it", "its",
    "this", "that", "what", "which", "how", "when", "should", "does",
    "do", "can", "if", "then", "than", "as", "after", "before", "all",
    "each", "every", "one", "two", "also", "only", "no", "not", "but",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (hyphenated codes like f-204 stay whole)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def build_vocabulary(docs: list[dict] | None = None) -> list[str]:
    """Fixed vocabulary from the seeded document texts only."""
    vocab: set[str] = set()
    for doc in docs or DOCUMENTS:
        vocab.update(tokenize(doc.get("title", "")))
        vocab.update(tokenize(doc.get("content", "")))
    return sorted(vocab)


class DeterministicFakeEmbeddings:
    """BoW vectors over the document-derived vocabulary (fake embeddings).

    Same interface as the real embedding model (``embed_query`` /
    ``embed_documents``), so it drops into ``infra.qdrant_client`` via
    the injected-``embeddings`` parameter.
    """

    def __init__(self, vocabulary: list[str] | None = None):
        self.vocabulary = vocabulary or build_vocabulary()
        self._index = {token: i for i, token in enumerate(self.vocabulary)}

    @property
    def dimension(self) -> int:
        return len(self.vocabulary)

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * len(self.vocabulary)
        for token in tokenize(text):
            idx = self._index.get(token)
            if idx is not None:
                vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


def _point_id(doc_id: str) -> str:
    """Point id for a doc — same uuid5 scheme as ``upsert_documents``."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{doc_id}"))


class FakeQdrantClient:
    """In-memory Qdrant stand-in holding the seeded documents.

    Implements what ``infra.qdrant_client.search_documents`` calls
    (``get_collection`` / ``query_points``) plus, for the Phase 1
    payload backfill, ``retrieve`` / ``scroll`` / ``set_payload`` /
    ``upsert``. Payloads are stored per doc and ``query_points``
    reflects ``set_payload`` updates, like the real store.
    """

    def __init__(self, embeddings: DeterministicFakeEmbeddings | None = None):
        self.embeddings = embeddings or DeterministicFakeEmbeddings()
        self._vectors = {
            doc["id"]: self.embeddings.embed_documents(
                [f"{doc.get('title', '')} {doc.get('content', '')}"]
            )[0]
            for doc in DOCUMENTS
        }
        self._payloads = {
            doc["id"]: {
                "doc_id": doc["id"],
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
            }
            for doc in DOCUMENTS
        }
        self._point_to_doc = {doc["id"]: _point_id(doc["id"]) for doc in DOCUMENTS}
        self._doc_by_point = {_point_id(doc["id"]): doc["id"] for doc in DOCUMENTS}

    def get_collection(self, name):  # noqa: ARG002 - name is part of the API
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=self.embeddings.dimension)
                )
            )
        )

    def query_points(self, collection_name, query, limit, with_payload=True):  # noqa: ARG002
        scored = []
        for doc in DOCUMENTS:
            vec = self._vectors[doc["id"]]
            score = sum(a * b for a, b in zip(query, vec))
            scored.append((doc, score))
        # Highest cosine first; doc id breaks ties deterministically
        # (e.g. the all-zero query vector, where every score is 0.0).
        scored.sort(key=lambda item: (-item[1], item[0]["id"]))
        points = [
            SimpleNamespace(
                id=doc["id"],
                score=score,
                payload=dict(self._payloads[doc["id"]]) if with_payload else {},
            )
            for doc, score in scored[:limit]
        ]
        return SimpleNamespace(points=points)

    def retrieve(self, collection_name, ids, with_payload=True):  # noqa: ARG002
        points = []
        for point_id in ids:
            doc_id = self._doc_by_point.get(str(point_id))
            if doc_id is None:
                continue
            points.append(SimpleNamespace(
                id=str(point_id),
                payload=dict(self._payloads[doc_id]) if with_payload else {},
            ))
        return points

    def scroll(self, collection_name, limit=100, offset=None, with_payload=True):  # noqa: ARG002
        """Paginate over all points; returns (points, next_offset)."""
        doc_ids = sorted(self._payloads)
        start = int(offset) if offset is not None else 0
        page = doc_ids[start:start + limit]
        points = [
            SimpleNamespace(
                id=self._point_to_doc[doc_id],
                payload=dict(self._payloads[doc_id]) if with_payload else {},
            )
            for doc_id in page
        ]
        next_offset = start + len(page) if start + len(page) < len(doc_ids) else None
        return points, next_offset

    def set_payload(self, collection_name, payload, points, wait=True):  # noqa: ARG002
        """Payload-only update (no re-embedding) for the given point ids."""
        for point_id in points:
            doc_id = self._doc_by_point.get(str(point_id))
            if doc_id is not None:
                self._payloads[doc_id].update(dict(payload))
        return SimpleNamespace(status="completed")

    def upsert(self, collection_name, points, wait=True):  # noqa: ARG002
        for point in points:
            payload = dict(getattr(point, "payload", {}) or {})
            doc_id = payload.get("doc_id")
            if doc_id:
                self._payloads[doc_id] = payload
                self._doc_by_point[str(point.id)] = doc_id
                self._point_to_doc[doc_id] = str(point.id)
        return SimpleNamespace(status="completed")


def baseline_search(
    question: str,
    limit: int = 20,
    embeddings: DeterministicFakeEmbeddings | None = None,
    client: FakeQdrantClient | None = None,
) -> list[dict]:
    """Plain-vector search over the seeded docs via the real search path."""
    from infra.qdrant_client import search_documents

    embeddings = embeddings or DeterministicFakeEmbeddings()
    client = client or FakeQdrantClient(embeddings)
    return search_documents(question, limit=limit, embeddings=embeddings, client=client)


def build_baseline_retriever(limit: int = 20):
    """Return ``question -> (ranked doc ids, trace)`` for the harness."""
    embeddings = DeterministicFakeEmbeddings()
    client = FakeQdrantClient(embeddings)

    def retrieve(question: str):
        docs = baseline_search(question, limit=limit, embeddings=embeddings, client=client)
        ranked_ids = [doc["doc_id"] for doc in docs]
        trace = {
            "scores": {doc["doc_id"]: round(doc.get("score", 0.0), 4) for doc in docs},
            "fake": True,
            "embedding": "deterministic-bow-over-seeded-doc-vocab",
            "limit": limit,
        }
        return ranked_ids, trace

    retrieve.__name__ = "plain_vector_baseline"
    return retrieve
