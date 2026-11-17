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
import time
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
    """Embed texts with the index-time embedding model (LiteLLM proxy).

    When the embeddings object is built here it is wrapped with the
    process embedding cache (``runtime/embedding_cache.py``): the
    same text embedded twice in one process reuses its vector.
    Explicitly injected embeddings are used as-is.
    """
    if embeddings is None:
        from runtime.embedding_cache import wrap_embeddings
        from runtime.llm_factory import build_embeddings

        embeddings = wrap_embeddings(
            build_embeddings(model=settings.resolved_embedding_model_index)
        )
    return embeddings.embed_documents(list(texts))


def embed_query(text: str, embeddings=None) -> list[float]:
    """Embed one query string with the query-time embedding model.

    Same caching rule as ``embed_texts`` (see above).
    """
    if embeddings is None:
        from runtime.embedding_cache import wrap_embeddings
        from runtime.llm_factory import build_embeddings

        embeddings = wrap_embeddings(
            build_embeddings(model=settings.resolved_embedding_model_query)
        )
    return embeddings.embed_query(text)


# B10: probed dimensions, cached per index model (see probe_dimension).
_probe_dimension_cache: dict[str, int] = {}


def probe_dimension(embeddings=None) -> int:
    """Detect the embedding vector dimension from a probe embedding.

    The dimension depends on the user's embedding model (e.g. 1536 for
    text-embedding-3-small, 3072 for text-embedding-3-large), so it is
    measured, never assumed. The probe uses the INDEX-time model: the
    collection stores index-time vectors, so its dimension is the index
    model's dimension.

    B10: the dimension is a property of the model, so when the
    embeddings object is built here (``embeddings=None``) the probed
    value is cached per process keyed by the index model name — one
    probe per model, not one per documents file. An explicitly
    injected embeddings object is always probed fresh (test fakes
    keep exact call counts).
    """
    built_here = embeddings is None
    model = settings.resolved_embedding_model_index if built_here else None
    if built_here:
        cached = _probe_dimension_cache.get(model)
        if cached is not None:
            return cached
        from runtime.llm_factory import build_embeddings

        embeddings = build_embeddings(model=model)
    vector = embed_query("dimension probe", embeddings=embeddings)
    dim = len(vector)
    if built_here:
        if len(_probe_dimension_cache) >= 8:
            _probe_dimension_cache.pop(next(iter(_probe_dimension_cache)))
        _probe_dimension_cache[model] = dim
    return dim


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


# B10: (collection, dim) pairs already verified against a given
# client object, so ``ensure_collection`` does not re-read the
# collection metadata on every documents file. Keyed by client
# identity with a strong reference held (a different client — e.g.
# another test's mock — never gets a stale hit; a deleted
# collection would surface loudly at upsert time, not silently).
_collection_verified: dict[tuple[int, str, int], object] = {}


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
    verified_key = (id(client), name, dim)
    if verified_key in _collection_verified:
        return "exists"
    existing_dim = get_collection_dim(client, name)
    if existing_dim is not None:
        if existing_dim != dim:
            raise RuntimeError(
                f"Qdrant collection {name!r} has vector size {existing_dim} but the "
                f"index-time embedding model ({settings.resolved_embedding_model_index}) produces "
                f"size {dim}. Change EMBEDDING_MODEL_INDEX back, or delete the "
                f"collection and re-index real documents."
            )
        if len(_collection_verified) >= 16:
            _collection_verified.pop(next(iter(_collection_verified)))
        _collection_verified[verified_key] = client
        return "exists"
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("Created Qdrant collection %r (dim=%d)", name, dim)
    if len(_collection_verified) >= 16:
        _collection_verified.pop(next(iter(_collection_verified)))
    _collection_verified[verified_key] = client
    return "created"


def reset_qdrant_caches() -> None:
    """Drop the process caches in this module (tests / reconnect).

    Substrate (counts/links), probed dimensions and verified
    collections are all recomputed on next use.
    """
    _substrate_cache.clear()
    _probe_dimension_cache.clear()
    _collection_verified.clear()


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
    invalidate_qdrant_substrate_cache(name)
    return {"upserted": len(points), "skipped": len(docs) - len(points)}


_PAYLOAD_BATCH_SIZE = 100


def _apply_payload_updates(
    client, name: str, updates: list[tuple[str, dict]]
) -> int:
    """Apply (point id, payload) updates, batched when the client allows.

    One ``batch_update_points`` call carries up to
    ``_PAYLOAD_BATCH_SIZE`` SetPayload operations (distinct payloads
    per point — grouping by identical payload would not help, the
    payloads differ per doc). Clients without ``batch_update_points``
    (older fakes) fall back to one ``set_payload`` per point. Returns
    the number of points updated.
    """
    if not updates:
        return 0
    batch = getattr(client, "batch_update_points", None)
    if batch is None:
        for point_id, payload in updates:
            client.set_payload(
                collection_name=name, payload=dict(payload), points=[point_id], wait=True
            )
        return len(updates)
    from qdrant_client.http.models import SetPayload, SetPayloadOperation

    for start in range(0, len(updates), _PAYLOAD_BATCH_SIZE):
        chunk = updates[start:start + _PAYLOAD_BATCH_SIZE]
        batch(
            collection_name=name,
            update_operations=[
                SetPayloadOperation(
                    set_payload=SetPayload(payload=dict(payload), points=[point_id])
                )
                for point_id, payload in chunk
            ],
            wait=True,
        )
    return len(updates)


def set_document_payloads(
    payloads_by_point_id: dict[str, dict], client=None, collection_name: str | None = None
) -> dict:
    """Payload-only update for existing points (no re-embedding).

    ``payloads_by_point_id`` maps point id (uuid5 of the doc id, as
    in ``upsert_documents``) to the payload fields to set. Updates
    are sent batched (see ``_apply_payload_updates``). Returns
    ``{"updated": n}``.
    """
    client = client or get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    name = collection_name or settings.qdrant_collection
    updated = _apply_payload_updates(
        client, name, list(payloads_by_point_id.items())
    )
    if updated:
        invalidate_qdrant_substrate_cache(name)
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
        page_updates: list[tuple[str, dict]] = []
        for point in points:
            payload = dict(getattr(point, "payload", {}) or {})
            doc_id = payload.get("doc_id")
            scanned += 1
            if not doc_id:
                unknown += 1
                continue
            links = compute_document_links(
                {
                    "id": doc_id,
                    "asset_id": payload.get("asset_id"),
                    "component_id": payload.get("component_id"),
                    "building_id": payload.get("building_id"),
                }
            )
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
            page_updates.append((point.id, wanted))
        updated += _apply_payload_updates(client, name, page_updates)
        if offset is None:
            break
    if updated:
        invalidate_qdrant_substrate_cache(name)
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


# --- Retrieval substrate cache (B6) --------------------------------------
# ``count_documents_per_node`` + ``get_document_links`` both scroll the
# WHOLE collection (limit 100/page) — and a HybridRetriever used to be
# rebuilt per ``hybrid_documents`` call, so one execution paid for the
# same two full scrolls twice. The counts and the links come from the
# same payloads, so one scroll computes both; the result is cached
# per process, keyed by collection, with the client object held for
# an identity check (a different client — e.g. another test's mock —
# never gets a stale hit). TTL 60 s (the drain cadence) is the
# fallback; every Qdrant write in this module invalidates the cache
# immediately, so invalidation-on-write is the primary freshness
# rule and the TTL only bounds a missed invalidation. NEVER
# unbounded/uninvalidated: the cache is one entry per collection.
_SUBSTRATE_TTL_SECONDS = 60
_substrate_cache: dict[str, dict] = {}


def invalidate_qdrant_substrate_cache(collection_name: str | None = None) -> None:
    """Drop cached Qdrant substrate (counts + links).

    Called after every Qdrant write in this module. ``None`` drops
    all collections; a name drops just that one.
    """
    if collection_name is None:
        _substrate_cache.clear()
    else:
        _substrate_cache.pop(collection_name, None)


def get_qdrant_substrate(
    client=None, collection_name: str | None = None
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """(doc counts per node, doc -> node ids) from ONE full scroll.

    Cached per process (see above): a fresh entry is computed by a
    single scroll and shared by both readers. Returns COPIES — the
    caller mutating the result must never poison the cache.
    """
    client = _live_client(client)
    name = collection_name or settings.qdrant_collection
    now = time.monotonic()
    entry = _substrate_cache.get(name)
    if (
        entry is not None
        and entry["client"] is client
        and now - entry["at"] < _SUBSTRATE_TTL_SECONDS
    ):
        return dict(entry["counts"]), {
            doc_id: list(node_ids) for doc_id, node_ids in entry["links"].items()
        }
    counts: dict[str, int] = {}
    links: dict[str, list[str]] = {}
    for payload in _scroll_payloads(client, name):
        doc_id = payload.get("doc_id")
        node_ids = list(payload.get("node_ids") or [])
        for node_key in node_ids:
            counts[node_key] = counts.get(node_key, 0) + 1
        if doc_id and node_ids:
            links[doc_id] = node_ids
    _substrate_cache[name] = {
        "client": client,
        "at": now,
        "counts": counts,
        "links": links,
    }
    return dict(counts), {
        doc_id: list(node_ids) for doc_id, node_ids in links.items()
    }


def _live_client(client):
    """Resolve an explicit client, or the configured one; fail fast."""
    client = client or get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    return client


def count_documents_per_node(client=None, collection_name: str | None = None) -> dict[str, int]:
    """Live corpus counts: node key -> number of docs linked to it.

    Read from the shared substrate cache (one scroll computes counts
    + links together — see ``get_qdrant_substrate``). A missing/empty
    count is maximally specific in ``runtime/retrieval_ppr.py::
    seed_specificity``, so an empty store degrades to ``{}`` here,
    never a crash. Raises RuntimeError when Qdrant is unreachable.
    """
    counts, _links = get_qdrant_substrate(client, collection_name)
    return counts


def get_document_links(client=None, collection_name: str | None = None) -> dict[str, list[str]]:
    """Live doc -> node ids mapping, from Qdrant payload ``node_ids``.

    Read from the shared substrate cache (see
    ``get_qdrant_substrate``). A doc is a graph candidate when its
    linked nodes touch a seed or a 1-hop neighbour. Docs with no
    payload ``node_ids`` are omitted (an unlinked doc scores
    ``graph_score = 0`` and the vector arm carries it). An empty
    store returns ``{}``. Raises RuntimeError when Qdrant is
    unreachable.
    """
    _counts, links = get_qdrant_substrate(client, collection_name)
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
    query_text: str, limit: int = 5, embeddings=None, client=None, *, slim: bool = False
) -> list[dict]:
    """Vector search over the technical-documents collection.

    Raises RuntimeError when Qdrant or the embedding model is unreachable,
    or when the collection does not exist yet (index real documents first).

    B14: ``slim=True`` fetches only ``doc_id`` + ``node_ids`` (+ short
    ``title``) — no ``content``. The hybrid path uses it: the vector
    arm ranks 40 docs but fusion keeps 6, so shipping up to 10,000
    chars of content per doc for all 40 is pure waste. Content is
    fetched only for the docs that survive (see
    ``HybridRetriever._candidate_rows`` / ``_fill_top_contents``).
    The plain path (``repositories.search_documents``) keeps the
    default full payload + its 4,000-char cut.
    """
    client = client or get_client()
    if client is None:
        raise RuntimeError("Qdrant not configured (QDRANT_URL empty)")
    if get_collection_dim(client) is None:
        raise RuntimeError(
            f"Qdrant collection {settings.qdrant_collection!r} does not exist yet. "
            "Index real documents into it first (see README §3)."
        )
    # B4: the question embedding goes through the shared B13
    # process cache keyed by (model, text) — the same cached
    # vector the plan path's intent scoring reuses, so the
    # question is embedded once per (model, text), not once per
    # call site. An embeddings object with no readable model name
    # is not cached (test fakes take this path — call counts
    # stay exact). ``embeddings=None`` keeps the existing path
    # (build + wrap with the cache).
    if embeddings is None:
        vector = embed_query(query_text)
    else:
        from runtime.embedding_cache import cached_embed_query

        vector = cached_embed_query(embeddings, query_text)
    try:
        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            limit=limit,
            with_payload=["doc_id", "node_ids", "title"] if slim else True,
        )
    except Exception as exc:  # noqa: BLE001 — surface Qdrant errors clearly
        raise RuntimeError(f"Qdrant search failed: {exc}") from exc
    docs = []
    for point in response.points:
        payload = dict(point.payload or {})
        payload.setdefault("score", point.score)
        docs.append(payload)
    return docs
