"""Process-level embedding cache (B13).

Embeddings are a deterministic function of (model, text), but the
same text is embedded several times per execution through different
paths — the hook query vs the agent re-searching the same question,
the plan path's intent scoring vs the vector search. Each repeat is
a LiteLLM call (latency + tokens) for a vector already computed.

This module caches query/document vectors in-process, keyed by
(model, text), with a short TTL and a bounded size. The cache store
is global so different wrapper objects built from the same model
share hits; the wrapper itself is thin.

Rules:
- Key includes the model name: vectors are NEVER served across a
  model change. An embeddings object with no readable model name
  is not cached at all (test fakes take this path — their call
  counts stay exact).
- TTL 300 s, max 512 entries (LRU). Expired entries are purged
  lazily on access; the cache never grows without bound.
- Thread-safe (one lock): the worker serves concurrent executions
  in-process.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

_TTL_SECONDS = 300
_MAX_ENTRIES = 512

_lock = threading.Lock()
# (model, text) -> (expires_at_monotonic, vector)
_CACHE: OrderedDict[tuple[str, str], tuple[float, list[float]]] = {}


def _model_name(embeddings, model: str | None) -> str | None:
    if model:
        return model
    return getattr(embeddings, "model", None)


def cached_embed_query(
    embeddings, text: str, *, model: str | None = None
) -> list[float]:
    """Embed one query string, reusing a cached vector when fresh."""
    name = _model_name(embeddings, model)
    if name is None:
        return embeddings.embed_query(text)
    key = (name, text)
    now = time.monotonic()
    with _lock:
        entry = _CACHE.get(key)
        if entry is not None and entry[0] > now:
            _CACHE.move_to_end(key)
            return list(entry[1])
        if entry is not None:
            del _CACHE[key]
    vector = embeddings.embed_query(text)
    with _lock:
        _CACHE[key] = (now + _TTL_SECONDS, list(vector))
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return vector


def cached_embed_documents(
    embeddings, texts: list[str], *, model: str | None = None
) -> list[list[float]]:
    """Embed documents, reusing cached vectors per text.

    Only the cache-missing texts are embedded, in one call, in
    order — the result order always matches the input order.
    """
    name = _model_name(embeddings, model)
    if name is None:
        return embeddings.embed_documents(list(texts))
    now = time.monotonic()
    out: list[list[float] | None] = [None] * len(texts)
    missing: list[int] = []
    with _lock:
        for i, text in enumerate(texts):
            entry = _CACHE.get((name, text))
            if entry is not None and entry[0] > now:
                _CACHE.move_to_end((name, text))
                out[i] = list(entry[1])
            else:
                if entry is not None:
                    del _CACHE[(name, text)]
                missing.append(i)
    if missing:
        vectors = embeddings.embed_documents([texts[i] for i in missing])
        with _lock:
            for i, vector in zip(missing, vectors):
                out[i] = list(vector)
                _CACHE[(name, texts[i])] = (now + _TTL_SECONDS, list(vector))
            while len(_CACHE) > _MAX_ENTRIES:
                _CACHE.popitem(last=False)
    return [list(v) for v in out]  # type: ignore[list-item]


class CachedEmbeddings:
    """Thin caching proxy over any embeddings object.

    ``embed_query`` / ``embed_documents`` go through the shared
    process cache; every other attribute passes through unchanged
    (``__getattr__``), so the proxy is a drop-in replacement.
    """

    def __init__(self, embeddings, *, model: str | None = None) -> None:
        object.__setattr__(self, "_embeddings", embeddings)
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_embeddings"), name)

    def embed_query(self, text: str) -> list[float]:
        return cached_embed_query(
            object.__getattribute__(self, "_embeddings"),
            text,
            model=object.__getattribute__(self, "_model"),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return cached_embed_documents(
            object.__getattribute__(self, "_embeddings"),
            list(texts),
            model=object.__getattribute__(self, "_model"),
        )


def wrap_embeddings(embeddings, *, model: str | None = None) -> CachedEmbeddings:
    """Wrap an embeddings object with the process embedding cache."""
    if isinstance(embeddings, CachedEmbeddings):
        return embeddings
    return CachedEmbeddings(embeddings, model=model)


def clear_embedding_cache() -> None:
    """Drop all cached vectors (tests / model change out-of-band)."""
    with _lock:
        _CACHE.clear()


def embedding_cache_size() -> int:
    """Current cache entries (observability / tests)."""
    with _lock:
        return len(_CACHE)
