"""Qdrant client — document embeddings / vector retrieval (Layer 6).

Embeddings come from the user's LiteLLM proxy via `runtime.llm_factory`
(OpenAI-compatible). The vector dimension is NEVER hardcoded: it is
detected from a probe embedding at collection-creation time and stored on the
collection; a mismatch between the configured embedding model and an
existing collection raises a clear error instead of silently corrupting
results.

No invented documents: every function here raises a clear RuntimeError
when Qdrant or the embedding model is unreachable. Callers decide how to
surface that error — repositories never return fake documents.
"""

from __future__ import annotations

import logging
import uuid

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def get_client():
    """Lazily build the Qdrant client, or None if QDRANT_URL is empty."""
    global _client
    if _client is not None:
        return _client
    if not settings.qdrant_url:
        return None
    from qdrant_client import QdrantClient

    # trust_env=False: Qdrant is reached directly, never via an egress
    # proxy. httpx otherwise reads HTTP_PROXY/NO_PROXY from the environment,
    # and some environments export NO_PROXY entries (e.g. IPv6 literals)
    # that httpx cannot parse as URL patterns, which breaks client
    # construction before any request is made.
    _client = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, trust_env=False
    )
    return _client


def reset_client() -> None:
    """Drop the cached client (tests / reconnect)."""
    global _client
    _client = None


def embed_texts(texts: list[str], embeddings=None) -> list[list[float]]:
    """Embed texts with the configured embedding model (LiteLLM proxy)."""
    if embeddings is None:
        from runtime.llm_factory import build_embeddings

        embeddings = build_embeddings()
    return embeddings.embed_documents(list(texts))


def embed_query(text: str, embeddings=None) -> list[float]:
    """Embed one query string with the configured embedding model."""
    if embeddings is None:
        from runtime.llm_factory import build_embeddings

        embeddings = build_embeddings()
    return embeddings.embed_query(text)


def probe_dimension(embeddings=None) -> int:
    """Detect the embedding vector dimension from a probe embedding.

    The dimension depends on the user's embedding model (e.g. 1536 for
    text-embedding-3-small, 3072 for text-embedding-3-large), so it is
    measured, never assumed.
    """
    vector = embed_query("dimension probe", embeddings=embeddings)
    return len(vector)


def collection_exists(client=None, collection_name: str | None = None) -> bool:
    """True if the collection exists. Raises if Qdrant is unreachable."""
    client = client or get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    name = collection_name or settings.qdrant_collection
    try:
        client.get_collection(name)
        return True
    except Exception as exc:  # noqa: BLE001 — qdrant raises on missing collection
        if "not found" in str(exc).lower() or "404" in str(exc):
            return False
        raise RuntimeError(f"Could not check Qdrant collection {name!r}: {exc}") from exc


def get_collection_dim(client=None, collection_name: str | None = None) -> int | None:
    """Vector dimension of an existing collection, or None if it is missing."""
    client = client or get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    name = collection_name or settings.qdrant_collection
    try:
        info = client.get_collection(name)
    except Exception as exc:  # noqa: BLE001 — see collection_exists
        if "not found" in str(exc).lower() or "404" in str(exc):
            return None
        raise RuntimeError(f"Could not read Qdrant collection {name!r}: {exc}") from exc
    vectors = info.config.params.vectors
    size = getattr(vectors, "size", None)
    if size is None and isinstance(vectors, dict):
        first = next(iter(vectors.values()), None)
        size = getattr(first, "size", None) if first is not None else None
    return int(size) if size is not None else None


def ensure_collection(dim: int, client=None, collection_name: str | None = None) -> str:
    """Create the collection with `dim` if missing; verify dim if present.

    Returns "created" or "exists". Raises RuntimeError on a dimension
    mismatch (change EMBEDDING_MODEL or delete the collection first).
    """
    from qdrant_client.http.models import Distance, VectorParams

    client = client or get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    name = collection_name or settings.qdrant_collection
    existing_dim = get_collection_dim(client, name)
    if existing_dim is not None:
        if existing_dim != dim:
            raise RuntimeError(
                f"Qdrant collection {name!r} has vector size {existing_dim} but the "
                f"configured embedding model ({settings.embedding_model}) produces "
                f"size {dim}. Change EMBEDDING_MODEL back, or delete the "
                f"collection and re-index real documents."
            )
        return "exists"
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("Created Qdrant collection %r (dim=%d)", name, dim)
    return "created"


def upsert_documents(
    docs: list[dict],
    embeddings=None,
    client=None,
    collection_name: str | None = None,
) -> dict:
    """Embed + upsert documents. Idempotent: point ids are uuid5 of the
    document id, existing points are skipped, and re-running changes
    nothing. Each doc needs: id, title, content (optional: extra payload).

    Returns {"upserted": n, "skipped": n}.
    """
    from qdrant_client.http.models import PointStruct

    client = client or get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    name = collection_name or settings.qdrant_collection
    if not docs:
        return {"upserted": 0, "skipped": 0}
    point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{d['id']}")) for d in docs]
    existing = set()
    try:
        found = client.retrieve(collection_name=name, ids=point_ids, with_payload=False)
        existing = {str(p.id) for p in found}
    except Exception as exc:  # noqa: BLE001 — retrieve failure is fatal here
        raise RuntimeError(f"Could not check existing Qdrant points: {exc}") from exc
    todo = [(d, pid) for d, pid in zip(docs, point_ids) if pid not in existing]
    if not todo:
        return {"upserted": 0, "skipped": len(docs)}
    texts = [d.get("content", "") or d.get("title", "") for d, _ in todo]
    vectors = embed_texts(texts, embeddings=embeddings)
    points = [
        PointStruct(
            id=pid,
            vector=vector,
            payload={
                "doc_id": d["id"],
                "title": d.get("title", ""),
                "content": d.get("content", ""),
                **{k: v for k, v in d.items() if k not in ("id", "title", "content")},
            },
        )
        for (d, pid), vector in zip(todo, vectors)
    ]
    client.upsert(collection_name=name, points=points, wait=True)
    return {"upserted": len(points), "skipped": len(docs) - len(points)}


def search_documents(
    query_text: str, limit: int = 5, embeddings=None, client=None
) -> list[dict]:
    """Vector search over the technical-documents collection.

    Raises RuntimeError when Qdrant or the embedding model is unreachable,
    or when the collection does not exist yet (index real documents first).
    """
    client = client or get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    if get_collection_dim(client) is None:
        raise RuntimeError(
            f"Qdrant collection {settings.qdrant_collection!r} does not exist yet. "
            "Index real documents into it first (see README §3)."
        )
    vector = embed_query(query_text, embeddings=embeddings)
    try:
        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface Qdrant errors clearly
        raise RuntimeError(f"Qdrant search failed: {exc}") from exc
    docs = []
    for point in response.points:
        payload = dict(point.payload or {})
        payload.setdefault("score", point.score)
        docs.append(payload)
    return docs
