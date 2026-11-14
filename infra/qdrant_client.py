"""Qdrant client — document embeddings / vector retrieval (Layer 6).

Embeddings come from the user's LiteLLM proxy via `runtime.llm_factory`
(OpenAI-compatible). The vector dimension is NEVER hardcoded: it is
detected from a probe embedding at collection-creation time and stored on the
collection; a mismatch between the configured embedding model and an
existing collection raises a clear error instead of silently corrupting
results. Index-time and query-time embedding use separate models
(`EMBEDDING_MODEL_INDEX` / `EMBEDDING_MODEL_QUERY`, each falling back to
the shared `EMBEDDING_MODEL`) so either part can change later without
touching the other — but the two models must produce compatible vectors.

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
    """Embed texts with the index-time embedding model (LiteLLM proxy)."""
    if embeddings is None:
        from runtime.llm_factory import build_embeddings

        embeddings = build_embeddings(model=settings.resolved_embedding_model_index)
    return embeddings.embed_documents(list(texts))


def embed_query(text: str, embeddings=None) -> list[float]:
    """Embed one query string with the query-time embedding model."""
    if embeddings is None:
        from runtime.llm_factory import build_embeddings

        embeddings = build_embeddings(model=settings.resolved_embedding_model_query)
    return embeddings.embed_query(text)


def probe_dimension(embeddings=None) -> int:
    """Detect the embedding vector dimension from a probe embedding.

    The dimension depends on the user's embedding model (e.g. 1536 for
    text-embedding-3-small, 3072 for text-embedding-3-large), so it is
    measured, never assumed. The probe uses the INDEX-time model: the
    collection stores index-time vectors, so its dimension is the index
    model's dimension.
    """
    if embeddings is None:
        from runtime.llm_factory import build_embeddings

        embeddings = build_embeddings(model=settings.resolved_embedding_model_index)
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
                f"index-time embedding model ({settings.resolved_embedding_model_index}) produces "
                f"size {dim}. Change EMBEDDING_MODEL_INDEX back, or delete the "
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


def set_document_payloads(
    payloads_by_point_id: dict[str, dict], client=None, collection_name: str | None = None
) -> dict:
    """Payload-only update for existing points (no re-embedding).

    ``payloads_by_point_id`` maps point id (uuid5 of the doc id, as
    in ``upsert_documents``) to the payload fields to set. Returns
    ``{"updated": n}``.
    """
    client = client or get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    name = collection_name or settings.qdrant_collection
    updated = 0
    for point_id, payload in payloads_by_point_id.items():
        client.set_payload(
            collection_name=name, payload=dict(payload), points=[point_id], wait=True
        )
        updated += 1
    return {"updated": updated}


def backfill_document_links(client=None, collection_name: str | None = None) -> dict:
    """Backfill ``node_ids`` (+ asset/component/building) payloads.

    ``upsert_documents`` SKIPS existing points, so already-indexed
    docs never pick up new payload fields. This scrolls every point,
    computes its links from ``runtime.retrieval_linking`` (grounded
    in ``seeds/demo_data.py``), and updates the payload only — no
    re-embedding, no new points. Idempotent: a second run finds the
    payloads already equal and updates nothing.

    Returns ``{"scanned", "updated", "unchanged", "unknown"}`` where
    ``unknown`` counts points whose doc id has no grounded link.
    """
    from runtime.retrieval_linking import compute_document_links

    client = client or get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    name = collection_name or settings.qdrant_collection
    scanned = updated = unchanged = unknown = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name, limit=100, offset=offset, with_payload=True
        )
        for point in points:
            payload = dict(getattr(point, "payload", {}) or {})
            doc_id = payload.get("doc_id")
            scanned += 1
            if not doc_id:
                unknown += 1
                continue
            links = compute_document_links({"id": doc_id})
            if not links["node_ids"]:
                unknown += 1
                continue
            wanted = {
                "node_ids": links["node_ids"],
                "asset_id": links["asset_id"],
                "component_id": links["component_id"],
                "building_id": links["building_id"],
            }
            if all(payload.get(k) == v for k, v in wanted.items()):
                unchanged += 1
                continue
            client.set_payload(
                collection_name=name, payload=wanted, points=[point.id], wait=True
            )
            updated += 1
        if offset is None:
            break
    logger.info(
        "Qdrant link backfill: scanned=%d updated=%d unchanged=%d unknown=%d",
        scanned, updated, unchanged, unknown,
    )
    return {
        "scanned": scanned, "updated": updated,
        "unchanged": unchanged, "unknown": unknown,
    }


def _scroll_payloads(client, name):
    """Yield every point payload in the collection (scroll, limit 100).

    Shared by the Phase 5 live-retrieval helpers below; a scroll
    failure surfaces as a RuntimeError naming Qdrant, never a raw
    client error and never a silent empty result.
    """
    offset = None
    while True:
        try:
            points, offset = client.scroll(
                collection_name=name, limit=100, offset=offset, with_payload=True
            )
        except Exception as exc:  # noqa: BLE001 — surface Qdrant errors clearly
            raise RuntimeError(f"Qdrant scroll failed: {exc}") from exc
        for point in points:
            yield dict(getattr(point, "payload", {}) or {})
        if offset is None:
            break


def _live_client(client):
    """Resolve an explicit client, or the configured one; fail fast."""
    client = client or get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    return client


def count_documents_per_node(client=None, collection_name: str | None = None) -> dict[str, int]:
    """Live corpus counts: node key -> number of docs linked to it.

    Scrolls every point and counts its payload ``node_ids`` — the
    live replacement for the demo ``DOCUMENT_LINKS`` table in the
    hybrid retriever's seed-specificity (``runtime/retrieval_ppr.py::
    seed_specificity`` treats a missing/empty count as maximally
    specific, so an empty store degrades to ``{}`` here, never a
    crash). Raises RuntimeError when Qdrant is unreachable.
    """
    client = _live_client(client)
    name = collection_name or settings.qdrant_collection
    counts: dict[str, int] = {}
    for payload in _scroll_payloads(client, name):
        for node_key in payload.get("node_ids") or []:
            counts[node_key] = counts.get(node_key, 0) + 1
    return counts


def get_document_links(client=None, collection_name: str | None = None) -> dict[str, list[str]]:
    """Live doc -> node ids mapping, from Qdrant payload ``node_ids``.

    The live replacement for the demo ``DOCUMENT_LINKS`` table in the
    hybrid retriever's graph arm: a doc is a graph candidate when
    its linked nodes touch a seed or a 1-hop neighbour. Docs with no
    payload ``node_ids`` are omitted (an unlinked doc scores
    ``graph_score = 0`` and the vector arm carries it). An empty
    store returns ``{}``. Raises RuntimeError when Qdrant is
    unreachable.
    """
    client = _live_client(client)
    name = collection_name or settings.qdrant_collection
    links: dict[str, list[str]] = {}
    for payload in _scroll_payloads(client, name):
        doc_id = payload.get("doc_id")
        node_ids = payload.get("node_ids") or []
        if doc_id and node_ids:
            links[doc_id] = list(node_ids)
    return links


def get_document_payloads(
    doc_ids: list[str], client=None, collection_name: str | None = None
) -> dict[str, dict]:
    """Payloads (title/content/node_ids/...) for the given doc ids.

    Vector-search results already carry title/content in their
    payload; this is the fetch for graph-only candidates (docs the
    graph arm found that the vector arm did not), so the exact
    feature and the title/content in the candidate row come from
    real payloads, not the demo corpus. Point ids are the same
    uuid5 scheme as ``upsert_documents``. Unknown ids are simply
    absent from the result (a doc with no text contributes no text
    feature, never a crash). Raises RuntimeError when Qdrant is
    unreachable.
    """
    if not doc_ids:
        return {}
    client = _live_client(client)
    name = collection_name or settings.qdrant_collection
    point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{d}")) for d in doc_ids]
    try:
        found = client.retrieve(collection_name=name, ids=point_ids, with_payload=True)
    except Exception as exc:  # noqa: BLE001 — surface Qdrant errors clearly
        raise RuntimeError(f"Qdrant payload fetch failed: {exc}") from exc
    by_point = {str(getattr(point, "id", "")): point for point in found}
    out: dict[str, dict] = {}
    for doc_id, point_id in zip(doc_ids, point_ids):
        point = by_point.get(point_id)
        if point is None:
            continue
        payload = dict(getattr(point, "payload", {}) or {})
        out[payload.get("doc_id") or doc_id] = payload
    return out


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
