"""Qdrant client — document embeddings / vector retrieval (Layer 6).

Returns None / raises when Qdrant or the embedding model is not
configured; callers fall back to stub data. No silent fake vectors.
"""

from __future__ import annotations

import logging

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

    _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return _client


def embed_query(text: str) -> list[float]:
    """Embed one query string with the configured LLM provider's embeddings.

    Raises RuntimeError when no API key is configured.
    """
    if not settings.openai_api_key:
        raise RuntimeError("No OPENAI_API_KEY configured for embeddings")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(api_key=settings.openai_api_key).embed_query(text)


def search_documents(query_text: str, limit: int = 5) -> list[dict]:
    """Vector search over the technical-documents collection."""
    client = get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    vector = embed_query(query_text)
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=limit,
        with_payload=True,
    )
    docs = []
    for point in response.points:
        payload = dict(point.payload or {})
        payload.setdefault("score", point.score)
        docs.append(payload)
    return docs
