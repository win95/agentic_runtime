"""Phase 5 live retrieval eval — golden set against the REAL stores.

Unlike the Phase 0–4 evals (fake BoW / code-blurred setups, no live
servers), this script runs the golden set (scripts/
golden_set_fault_analysis.json, 36 cases) against the real Qdrant
collection + the real Neo4j graph, with the real LiteLLM query
embeddings. It reports the same aggregate + sibling-confusion tables
as the Phase 1 eval (``runtime/retrieval_harness.format_results_table``)
plus a store-inventory preamble (Qdrant collection + point count,
Neo4j reachable + a probe result, embedding model names resolved
from settings).

Preflight, fail fast (exit code 2), in order:
  (i)   embedding config present (``require_embedding_config``) —
        the fix names ``EMBEDDING_MODEL`` in ``.env``;
  (ii)  Qdrant reachable AND the collection exists AND holds >= 1
        point — the fix names Qdrant and the seed step;
  (iii) Neo4j reachable — BUT this one does NOT hard-fail: a missing
        graph is a legitimate live state (guardrail 3), so the run
        degrades to a vector-only live run with a clear warning and
        ``graph_missing`` reported throughout.

No grid search: live grid re-runs are the user's call (report §11.5).
The PASS/FAIL verdict uses the same exit criterion as Phase 1
(hybrid nDCG@6 > baseline, Recall@6 >= baseline, sibling nDCG@6 >
baseline) but is INFORMATIONAL live — the corpus is small and real
embeddings retain code signal, so a live FAIL is printed, not treated
as a code failure, and the script still exits 0 once the eval
completes. Exit code 2 means the preflight failed, before any
retrieval.

Usage:
    python -m scripts.eval_retrieval_live [--tracking-uri URI] [--output FILE]
    make eval-retrieval-live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from runtime.retrieval_harness import (  # noqa: E402
    format_results_table,
    load_golden_set,
    run_retrieval_eval,
)
from runtime.retrieval_hybrid import build_hybrid_retriever  # noqa: E402

PREFLIGHT_EXIT_CODE = 2


class LivePreflightError(RuntimeError):
    """A live store is missing/unreachable. Message names the store + fix."""


def _count_points(client, collection_name: str) -> int:
    """Point count for the collection.

    Prefers the client's ``count`` (real QdrantClient); falls back to
    a full scroll for clients without it (test stand-ins). Raises
    RuntimeError naming Qdrant when the store cannot be read.
    """
    count_fn = getattr(client, "count", None)
    if count_fn is not None:
        try:
            return int(count_fn(collection_name=collection_name).count)
        except Exception as exc:  # noqa: BLE001 — reword, never raw
            raise RuntimeError(f"Qdrant count failed: {exc}") from exc
    total = 0
    offset = None
    while True:
        try:
            points, offset = client.scroll(
                collection_name=collection_name, limit=100,
                offset=offset, with_payload=False,
            )
        except Exception as exc:  # noqa: BLE001 — reword, never raw
            raise RuntimeError(f"Qdrant scroll failed: {exc}") from exc
        total += len(points)
        if offset is None:
            break
    return total


def _probe_graph(store) -> tuple[bool, int | None]:
    """Neo4j health probe. Returns ``(reachable, node_count_or_None)``.

    ``is_healthy()`` decides reachability when the store has it;
    otherwise a one-row ``run_query`` probe does. The node count is
    best-effort inventory — a failed count never flips reachability.
    Never raises: an unreachable graph degrades, it does not fail.
    """
    if hasattr(store, "is_healthy"):
        try:
            reachable = bool(store.is_healthy())
        except Exception:  # noqa: BLE001 — down is a degraded run, not a crash
            return False, None
        if not reachable:
            return False, None
    else:
        try:
            store.run_query("MATCH (n) RETURN n LIMIT 1")
        except Exception:  # noqa: BLE001 — see above
            return False, None
        reachable = True
    node_count = None
    try:
        rows = store.run_query("MATCH (n) RETURN count(n) AS node_count")
        if rows:
            node_count = rows[0].get("node_count")
    except Exception:  # noqa: BLE001 — inventory only, never fatal
        node_count = None
    return reachable, node_count


def preflight_live_stores(*, client=None, graph_store=None) -> dict:
    """Check the live stores in order; fail fast on (i)/(ii), degrade on (iii).

    Returns the store inventory: Qdrant URL/collection/dim/point count,
    resolved embedding model names, Neo4j reachability + probe result,
    and ``graph_missing`` (True when the run must go vector-only).
    Raises ``LivePreflightError`` (exit code 2) when the embedding
    config or Qdrant is missing — the message names the store and the
    fix. Accepts injected ``client`` / ``graph_store`` so tests can
    call it directly with mocked stores.
    """
    from infra.qdrant_client import get_client, get_collection_dim
    from runtime.llm_factory import require_embedding_config

    # (i) Embedding config — the model name the LiteLLM proxy serves.
    try:
        require_embedding_config()
    except ValueError as exc:
        raise LivePreflightError(
            f"{exc} Set EMBEDDING_MODEL in .env to the model name your "
            "LiteLLM proxy serves."
        ) from exc

    # (ii) Qdrant reachable, collection exists, >= 1 point.
    qdrant_client = client or get_client()
    if qdrant_client is None:
        raise LivePreflightError(
            "Qdrant is not configured: QDRANT_URL is empty. "
            "Set QDRANT_URL in .env to your Qdrant server URL."
        )
    collection_name = settings.qdrant_collection
    try:
        dim = get_collection_dim(qdrant_client, collection_name)
    except RuntimeError as exc:
        raise LivePreflightError(
            f"Qdrant is unreachable at {settings.qdrant_url}: {exc}. "
            "Start Qdrant (docker compose up) and check QDRANT_URL in .env."
        ) from exc
    if dim is None:
        raise LivePreflightError(
            f"Qdrant collection {collection_name!r} does not exist yet. "
            "Seed it first: `make seed` (demo docs, with payload links) "
            "or `make ingest` (real technical documents)."
        )
    try:
        points = _count_points(qdrant_client, collection_name)
    except RuntimeError as exc:
        raise LivePreflightError(
            f"Qdrant is unreachable at {settings.qdrant_url}: {exc}. "
            "Start Qdrant (docker compose up) and check QDRANT_URL in .env."
        ) from exc
    if points < 1:
        raise LivePreflightError(
            f"Qdrant collection {collection_name!r} is empty (0 points). "
            "Seed it first: `make seed` (demo docs, with payload links) "
            "or `make ingest` (real technical documents)."
        )

    # (iii) Neo4j reachable — degrade to vector-only, never hard-fail.
    store = graph_store
    if store is None:
        from ontology.projection.provider import get_graph_store

        store = get_graph_store()
    graph_reachable, node_count = _probe_graph(store)

    return {
        "embedding_model_index": settings.resolved_embedding_model_index,
        "embedding_model_query": settings.resolved_embedding_model_query,
        "qdrant_url": settings.qdrant_url,
        "qdrant_collection": collection_name,
        "qdrant_dim": dim,
        "qdrant_points": points,
        "graph_missing": not graph_reachable,
        "graph_node_count": node_count,
    }


def build_live_baseline_retriever(limit: int = 20, *, embeddings=None, client=None):
    """Plain-vector live baseline — ``search_documents``, no graph arm.

    A small ``retrieve(question)`` closure over the real Qdrant
    client and the real query-time embeddings. Accepts injected
    ``embeddings`` / ``client`` so tests can build it against
    mocked stores.
    """
    from infra.qdrant_client import get_client, search_documents
    from runtime.llm_factory import build_embeddings

    live_embeddings = embeddings or build_embeddings(
        model=settings.resolved_embedding_model_query
    )
    live_client = client or get_client()
    if live_client is None:
        raise RuntimeError(
            "Qdrant is not configured: QDRANT_URL is empty. "
            "Set QDRANT_URL in .env to your Qdrant server URL."
        )

    def retrieve(question: str):
        docs = search_documents(
            question, limit=limit, embeddings=live_embeddings, client=live_client
        )
        ranked_ids = [doc["doc_id"] for doc in docs]
        trace = {
            "scores": {doc["doc_id"]: round(doc.get("score", 0.0), 4) for doc in docs},
            "live": True,
            "embedding": settings.resolved_embedding_model_query,
            "limit": limit,
        }
        return ranked_ids, trace

    retrieve.__name__ = "plain_vector_live_baseline"
    return retrieve


def format_inventory(inventory: dict) -> str:
    """One-screen store-inventory preamble for the live run."""
    graph_line = (
        "Neo4j: reachable"
        + (f" ({inventory['graph_node_count']} nodes)"
           if inventory["graph_node_count"] is not None else "")
        if not inventory["graph_missing"]
        else "Neo4j: UNREACHABLE — degrading to vector-only live run "
             "(guardrail 3: graph-missing is a legitimate live state)"
    )
    return "\n".join([
        "Live store inventory:",
        f"  Qdrant: {inventory['qdrant_url']} collection "
        f"{inventory['qdrant_collection']!r} dim={inventory['qdrant_dim']} "
        f"points={inventory['qdrant_points']}",
        f"  Embeddings: index={inventory['embedding_model_index']!r} "
        f"query={inventory['embedding_model_query']!r}",
        f"  {graph_line}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 live retrieval eval.")
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking URI (default: skip MLflow).")
    parser.add_argument("--output", default=None, help="Write full results JSON.")
    parser.add_argument("--golden-set", default=None,
                        help="Path to the golden set JSON.")
    args = parser.parse_args()

    try:
        inventory = preflight_live_stores()
    except LivePreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}")
        return PREFLIGHT_EXIT_CODE
    print(format_inventory(inventory))
    print()
    if inventory["graph_missing"]:
        print("WARNING: Neo4j unreachable — the hybrid below runs "
              "vector-only (graph_missing throughout); fix Neo4j and "
              "re-run for the full hybrid numbers.")
        print()

    golden = load_golden_set(args.golden_set)
    cases = golden["cases"]
    version = golden.get("version", "unknown")
    params = {
        "embedding_index": inventory["embedding_model_index"],
        "embedding_query": inventory["embedding_model_query"],
        "graph_missing": inventory["graph_missing"],
        "qdrant_collection": inventory["qdrant_collection"],
        "qdrant_points": inventory["qdrant_points"],
    }

    baseline = build_live_baseline_retriever()
    hybrid = build_hybrid_retriever()
    results: dict = {}
    for label, retriever in (("baseline", baseline), ("hybrid", hybrid)):
        key = f"live / {label}"
        results[key] = run_retrieval_eval(
            retriever, cases,
            retriever_name=f"{label}_live",
            golden_set_version=version,
            tracking_uri=args.tracking_uri,
            params=params,
        )
        print(f"--- {key} ---")
        print(format_results_table(results[key]))
        print()

    base = results["live / baseline"]["aggregate"]
    hyb = results["live / hybrid"]["aggregate"]
    sib_base = results["live / baseline"]["sibling_confusion"]
    sib_hyb = results["live / hybrid"]["sibling_confusion"]
    exit_ok = (
        hyb["ndcg@6"] > base["ndcg@6"]
        and hyb["recall@6"] >= base["recall@6"]
        and sib_hyb["ndcg@6"] > sib_base["ndcg@6"]
    )
    print("EXIT CRITERION (live, INFORMATIONAL — small corpus, real "
          "embeddings retain code signal; a FAIL here is not a code "
          "failure): "
          f"nDCG@6 {base['ndcg@6']:.3f} -> {hyb['ndcg@6']:.3f}, "
          f"Recall@6 {base['recall@6']:.3f} -> {hyb['recall@6']:.3f}, "
          f"sibling nDCG@6 {sib_base['ndcg@6']:.3f} -> "
          f"{sib_hyb['ndcg@6']:.3f} — "
          + ("PASS" if exit_ok else "FAIL"))
    print("NOTE: no grid search on the live run — live grid re-runs are "
          "the user's call (report §11.5).")

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"inventory": inventory, "results": results,
             "exit_criterion_pass": exit_ok,
             "exit_criterion_informational": True},
            indent=2, ensure_ascii=False, default=str,
        ))
        print(f"Full results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
