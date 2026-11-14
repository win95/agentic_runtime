"""Phase 5 live eval tests — preflight fail-fast + mocked live runs.

Mocked stores only: no LiteLLM proxy, no Qdrant/Neo4j server.
Main features only — no parametrized explosion.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from runtime.retrieval_graph import clear_gazetteer_codes
from runtime.retrieval_harness import run_retrieval_eval
from runtime.retrieval_hybrid import build_live_hybrid_retriever
from scripts.eval_retrieval_live import (
    LivePreflightError,
    build_live_baseline_retriever,
    preflight_live_stores,
)

ROOT = Path(__file__).resolve().parents[1]

LIVE_PAYLOADS = {
    "doc-live-motor": {
        "doc_id": "doc-live-motor",
        "title": "Motor overheat F-204",
        "content": "O21 motor-M2 overheats, fault F-204",
        "node_ids": ["O21", "motor-M2", "doc-live-motor"],
    },
    "doc-live-brake": {
        "doc_id": "doc-live-brake",
        "title": "Brake inspection",
        "content": "brake-BK1 inspection",
        "node_ids": ["O21", "brake-BK1"],
    },
}

# Tiny golden subset: gold ids are the mocked live doc ids.
TINY_CASES = [
    {
        "id": "t1", "question": "F-204 motor overheating O21",
        "expected_answer": "Motor overheat F-204",
        "must_include_facts": [], "gold_chunk_ids": ["doc-live-motor"],
        "gold_node_ids": ["O21"], "query_type": "lookup",
    },
    {
        "id": "t2", "question": "brake-BK1 inspection O21",
        "expected_answer": "Brake inspection",
        "must_include_facts": [], "gold_chunk_ids": ["doc-live-brake"],
        "gold_node_ids": ["O21"], "query_type": "lookup",
    },
]


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{doc_id}"))


class StubEmbeddings:
    """Embeddings that return a fixed vector — no LiteLLM call."""

    dimension = 4

    def embed_query(self, text):  # noqa: ARG002
        return [1.0, 0.0, 0.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class MockQdrantClient:
    """In-memory Qdrant stand-in (no ``count`` — exercises scroll fallback)."""

    def __init__(self, payloads, *, vector_ids=None, fail_get_collection=False,
                 missing_collection=False):
        self._payloads = {k: dict(v) for k, v in payloads.items()}
        self._vector_ids = list(vector_ids) if vector_ids is not None else sorted(payloads)
        self._fail_get_collection = fail_get_collection
        self._missing_collection = missing_collection

    def get_collection(self, name):  # noqa: ARG002
        if self._fail_get_collection:
            raise ConnectionError("connection refused")
        if self._missing_collection:
            raise RuntimeError(f"Collection {name} not found")
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=4))
            )
        )

    def query_points(self, collection_name, query, limit, with_payload=True):  # noqa: ARG002
        points = [
            SimpleNamespace(
                id=doc_id,
                score=0.9 - 0.01 * i,
                payload=dict(self._payloads[doc_id]) if with_payload else {},
            )
            for i, doc_id in enumerate(self._vector_ids[:limit])
        ]
        return SimpleNamespace(points=points)

    def scroll(self, collection_name, limit=100, offset=None, with_payload=True):  # noqa: ARG002
        doc_ids = sorted(self._payloads)
        start = int(offset) if offset is not None else 0
        page = doc_ids[start:start + limit]
        points = [
            SimpleNamespace(
                id=_point_id(doc_id),
                payload=dict(self._payloads[doc_id]) if with_payload else {},
            )
            for doc_id in page
        ]
        next_offset = start + len(page) if start + len(page) < len(doc_ids) else None
        return points, next_offset

    def retrieve(self, collection_name, ids, with_payload=True):  # noqa: ARG002
        by_point = {_point_id(doc_id): doc_id for doc_id in self._payloads}
        points = []
        for point_id in ids:
            doc_id = by_point.get(str(point_id))
            if doc_id is None:
                continue
            points.append(SimpleNamespace(
                id=str(point_id),
                payload=dict(self._payloads[doc_id]) if with_payload else {},
            ))
        return points


class MockGraphStore:
    """Neo4j stand-in: ``is_healthy`` + ``run_query`` rows, or down."""

    def __init__(self, rows=None, *, fail=False):
        self._rows = list(rows or [])
        self._fail = fail

    def is_healthy(self):
        return not self._fail

    def run_query(self, cypher, params=None):  # noqa: ARG002
        if self._fail:
            raise RuntimeError("Neo4j down")
        if "count(n)" in cypher:
            return [{"node_count": 2}]
        return [dict(row) for row in self._rows]


@pytest.fixture(autouse=True)
def _live_settings(monkeypatch):
    """Dummy embedding credentials (no .env in this VM) + stubbed gazetteer.

    The preflight fails fast on missing embedding config, so tests
    supply a placeholder key; the gazetteer refresh is stubbed out
    (the real one flips global live-mode state). Demo mode is
    restored after every test.
    """
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "qdrant_url", "http://localhost:6333")
    monkeypatch.setattr(
        "runtime.retrieval_graph.refresh_gazetteer_codes",
        lambda store=None: {"assets": 0, "components": 0, "buildings": 0,
                            "documents": 0, "source": "stubbed"},
    )
    yield
    clear_gazetteer_codes()


def _up_client(**kwargs):
    return MockQdrantClient(LIVE_PAYLOADS, **kwargs)


def _up_store():
    return MockGraphStore(rows=[
        {"from_id": "O21", "rel_type": "DOCUMENTED_BY",
         "to_id": "doc-live-motor", "to_label": "Document"},
    ])


# --- Preflight: fail fast ---------------------------------------------------

def test_preflight_fails_fast_embedding_config_missing(monkeypatch):
    monkeypatch.setattr(settings, "embedding_model", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    with pytest.raises(LivePreflightError, match="EMBEDDING_MODEL"):
        preflight_live_stores(client=_up_client(), graph_store=_up_store())
    with pytest.raises(LivePreflightError, match=r"\.env"):
        preflight_live_stores(client=_up_client(), graph_store=_up_store())


def test_preflight_fails_fast_qdrant_unreachable():
    client = _up_client(fail_get_collection=True)
    with pytest.raises(LivePreflightError, match="Qdrant"):
        preflight_live_stores(client=client, graph_store=_up_store())


def test_preflight_fails_fast_qdrant_collection_missing():
    client = _up_client(missing_collection=True)
    with pytest.raises(LivePreflightError, match="Qdrant"):
        preflight_live_stores(client=client, graph_store=_up_store())
    with pytest.raises(LivePreflightError, match="seed"):
        preflight_live_stores(client=client, graph_store=_up_store())


def test_preflight_fails_fast_qdrant_collection_empty():
    client = MockQdrantClient({})
    with pytest.raises(LivePreflightError, match="Qdrant"):
        preflight_live_stores(client=client, graph_store=_up_store())
    with pytest.raises(LivePreflightError, match="empty"):
        preflight_live_stores(client=client, graph_store=_up_store())


def test_preflight_ok_with_healthy_stores():
    inventory = preflight_live_stores(client=_up_client(), graph_store=_up_store())
    assert inventory["qdrant_collection"] == settings.qdrant_collection
    assert inventory["qdrant_dim"] == 4
    assert inventory["qdrant_points"] == 2
    assert inventory["embedding_model_query"] == "text-embedding-3-small"
    assert inventory["graph_missing"] is False
    assert inventory["graph_node_count"] == 2


# --- Preflight: Neo4j degradation -------------------------------------------

def test_preflight_degrades_when_neo4j_down():
    store = MockGraphStore(fail=True)
    inventory = preflight_live_stores(client=_up_client(), graph_store=store)
    assert inventory["graph_missing"] is True
    assert inventory["qdrant_points"] == 2
    assert inventory["graph_node_count"] is None


def test_hybrid_runs_vector_only_when_graph_missing():
    """Degraded run: hybrid still ranks through the harness, flagged."""
    client = _up_client()
    store = MockGraphStore(fail=True)
    inventory = preflight_live_stores(client=client, graph_store=store)
    assert inventory["graph_missing"] is True
    hybrid = build_live_hybrid_retriever(
        embeddings=StubEmbeddings(), client=client, graph_store=store
    )
    result = run_retrieval_eval(
        hybrid, TINY_CASES, retriever_name="hybrid_live_degraded",
        golden_set_version="test",
    )
    assert result["aggregate"]["n_cases"] == 2
    assert all(row["trace"]["graph_missing"] for row in result["per_query"])


# --- Live baseline + hybrid through the harness ------------------------------

def test_live_baseline_and_hybrid_rank_through_harness():
    client = _up_client()
    store = _up_store()
    baseline = build_live_baseline_retriever(embeddings=StubEmbeddings(), client=client)
    hybrid = build_live_hybrid_retriever(
        embeddings=StubEmbeddings(), client=client, graph_store=store
    )
    for label, retriever in (("baseline", baseline), ("hybrid", hybrid)):
        result = run_retrieval_eval(
            retriever, TINY_CASES, retriever_name=f"{label}_live",
            golden_set_version="test",
        )
        assert result["aggregate"]["n_cases"] == 2
        assert result["aggregate"]["n_answerable"] == 2
        ranked = result["per_query"][0]["ranked_ids"]
        assert "doc-live-motor" in ranked
        assert result["retriever"] == f"{label}_live"


# --- Script as a subprocess: real fail-fast in this VM ------------------------

def _run_live_eval_subprocess(extra_env=None):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home())}
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "scripts.eval_retrieval_live"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


def test_script_fails_fast_naming_qdrant_in_this_vm():
    """Real preflight in this VM: embedding config satisfied with a
    dummy key, so the run reaches Qdrant — which is unreachable here.
    Exit 2, Qdrant named, no retrieval attempted."""
    proc = _run_live_eval_subprocess({"LLM_API_KEY": "dummy-key"})
    assert proc.returncode == 2
    output = proc.stdout + proc.stderr
    assert "PREFLIGHT FAILED" in output
    assert "Qdrant" in output


def test_script_fails_fast_naming_embedding_config_without_key():
    """Default VM env (no LiteLLM key): the first preflight check
    fails, naming EMBEDDING_MODEL and the .env step."""
    proc = _run_live_eval_subprocess()
    assert proc.returncode == 2
    output = proc.stdout + proc.stderr
    assert "PREFLIGHT FAILED" in output
    assert "EMBEDDING_MODEL" in output
    assert ".env" in output
