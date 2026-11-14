"""Test-local doubles for the retrieval tests — NOTHING here is real.

After the fake removal (option B) there are no fake embeddings,
no fake Qdrant client and no demo adjacency in ``runtime/``. The
doubles in this module live in ``tests/`` only and inject CANNED
inputs into the real code: a small adjacency built from the real
seed data (``seeds/demo_data.py::RELATIONS``), real per-asset
ground-truth counts, the real payload link table, and a stub
Qdrant client over the 7 demo-document payloads. No canned number
here is a claim about live behaviour — the golden-set regression
numbers the old fakes produced (nDCG 0.906 -> 0.942 and friends)
are gone with them and are not reproduced here.
"""

from __future__ import annotations

import re
import uuid
from types import SimpleNamespace

from runtime.retrieval_graph import refresh_gazetteer_codes
from runtime.retrieval_intent import PROTOTYPES, _tokenize
from runtime.retrieval_linking import DOCUMENT_LINKS
from seeds.demo_data import DOCUMENTS, RELATIONS

# --- Canned gazetteer keys (the demo ontology's keys) ---------------------
# Served to ``refresh_gazetteer_codes`` by ``GazetteerKeyStore`` so
# ``gazetteer_seeds`` links the demo codes in tests. Fault codes are
# deliberately absent: there is no Fault entity type, so a fault
# code links nothing.

DEMO_GAZETTEER_KEYS: dict[str, list[str]] = {
    "Asset": ["O20", "O21", "O22", "O23", "O24"],
    "Component": [
        "motor-M0",
        "motor-M2",
        "motor-M3",
        "door-sensor-DS1",
        "controller-C1",
        "controller-C2",
        "brake-BK1",
    ],
    "Building": ["building-B1", "building-B2"],
    "Document": [
        "doc-door-sensor",
        "doc-motor-overheat",
        "doc-brake-inspection",
        "doc-leveling",
        "doc-rope-inspection",
        "doc-o23-overview",
    ],
}


class GazetteerKeyStore:
    """Minimal graph store serving canned entity keys per label.

    Only what ``ontology/knowledge/graph_query.py::list_entity_keys``
    needs: ``run_query`` returning ``{"id": key}`` rows for the
    label in the MATCH clause.
    """

    def __init__(self, keys_by_label=None):
        self.keys_by_label = keys_by_label or DEMO_GAZETTEER_KEYS
        self.calls: list[tuple[str, dict]] = []

    def run_query(self, cypher, params=None):
        self.calls.append((cypher, params or {}))
        match = re.search(r"\(n:([A-Za-z][A-Za-z0-9_]*)\)", cypher)
        label = match.group(1) if match else ""
        return [{"id": key} for key in self.keys_by_label.get(label, [])]


def refresh_demo_gazetteer() -> dict:
    """Populate the gazetteer cache with the demo ontology's keys."""
    return refresh_gazetteer_codes(store=GazetteerKeyStore())


# --- Canned adjacency (from the real seed RELATIONS) -----------------------
# Both directions per relation (the shape the graph code expects:
# edges stored directed as modelled, used undirected). Neighbours
# sorted by key, deterministic.

def canned_adjacency() -> dict:
    """Adjacency built from ``seeds/demo_data.py::RELATIONS``."""
    adjacency: dict[str, dict] = {}

    def _ensure(key: str, node_type: str) -> None:
        adjacency.setdefault(key, {"type": node_type, "neighbors": []})

    for frm, frm_type, rel, to, to_type in RELATIONS:
        _ensure(frm, frm_type)
        _ensure(to, to_type)
        if (rel, to, to_type) not in adjacency[frm]["neighbors"]:
            adjacency[frm]["neighbors"].append((rel, to, to_type))
        if (rel, frm, frm_type) not in adjacency[to]["neighbors"]:
            adjacency[to]["neighbors"].append((rel, frm, frm_type))
    for entry in adjacency.values():
        entry["neighbors"] = sorted(set(entry["neighbors"]), key=lambda e: (e[1], e[0]))
    return adjacency


# --- Canned per-asset ground truth (counted off seeds/demo_data.py) --------
# faults / sensor readings / maintenance rows per asset: O21 8/62/6,
# O22 2/6/1, O23 0/4/0, O24 0/4/2, O20 0/0/0.

CANNED_ASSET_DATA: dict[str, dict[str, int]] = {
    "O21": {"faults": 8, "sensors": 62, "maintenance": 6},
    "O22": {"faults": 2, "sensors": 6, "maintenance": 1},
    "O23": {"faults": 0, "sensors": 4, "maintenance": 0},
    "O24": {"faults": 0, "sensors": 4, "maintenance": 2},
    "O20": {"faults": 0, "sensors": 0, "maintenance": 0},
}

# Doc -> node ids, from the real payload link table.
CANNED_DOC_LINKS: dict[str, list[str]] = {
    doc_id: list(link["node_ids"]) for doc_id, link in DOCUMENT_LINKS.items()
}

# Doc counts per node, from the real payload link table.
CANNED_DOC_COUNTS: dict[str, int] = {}
for _node_ids in CANNED_DOC_LINKS.values():
    for _key in _node_ids:
        CANNED_DOC_COUNTS[_key] = CANNED_DOC_COUNTS.get(_key, 0) + 1


def canned_doc_texts() -> dict[str, dict]:
    """Demo-document payloads: {doc_id: {doc_id, title, content, node_ids}}."""
    return {
        doc["id"]: {
            "doc_id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "node_ids": list(CANNED_DOC_LINKS.get(doc["id"], [])),
        }
        for doc in DOCUMENTS
    }


def demo_vocabulary() -> set[str]:
    """Content words of the demo documents (intent rule (a) input)."""
    words: set[str] = set()
    for doc in DOCUMENTS:
        words |= set(_tokenize(doc["title"] + " " + doc["content"]))
    return words


# --- Stub embeddings ---------------------------------------------------------

class StubEmbeddings:
    """Embeddings returning one fixed vector — no LiteLLM call."""

    dimension = 4

    def embed_query(self, text):  # noqa: ARG002
        return [1.0, 0.0, 0.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class KeywordEmbeddings:
    """Canned keyword-overlap embeddings for intent scoring.

    One dimension per word in the intent prototypes: a text embeds
    to 1.0 on every prototype word it contains, 0.0 elsewhere, so
    cosine similarity is exactly the prototype-word overlap. This
    stands in for a real embedding model in tests only — it knows
    nothing about the documents and must never leave ``tests/``.
    """

    VOCABULARY = sorted({w for text in PROTOTYPES.values() for w in text.split()})
    _TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")

    def embed_query(self, text):
        tokens = set(self._TOKEN_RE.findall((text or "").lower()))
        return [1.0 if word in tokens else 0.0 for word in self.VOCABULARY]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


# --- Stub Qdrant client over canned payloads ----------------------------------

def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{doc_id}"))


class CannedQdrantClient:
    """In-memory Qdrant stand-in over canned payloads.

    ``vector_scores`` maps doc id -> the vector arm's score (a
    fixed order when omitted: 0.9, 0.89, ... over sorted doc ids).
    ``scroll`` / ``retrieve`` / ``set_payload`` implement the real
    payload helpers' surface (backfill + counts/links/texts).
    """

    def __init__(self, payloads=None, *, vector_scores=None, fail_search=False):
        default_payloads = payloads if payloads is not None else canned_doc_texts()
        self._payloads = {k: dict(v) for k, v in default_payloads.items()}
        if vector_scores is None:
            vector_scores = {
                doc_id: 0.9 - 0.01 * i
                for i, doc_id in enumerate(sorted(self._payloads))
            }
        self._vector_scores = dict(vector_scores)
        self._fail_search = fail_search
        self.scroll_calls = 0
        self.retrieve_calls = 0

    @property
    def payloads(self) -> dict[str, dict]:
        """Live view of the stored payloads (backfill assertions read this)."""
        return self._payloads

    def get_collection(self, name):  # noqa: ARG002
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=4))
            )
        )

    def query_points(self, collection_name, query, limit, with_payload=True):  # noqa: ARG002
        if self._fail_search:
            raise RuntimeError("connection refused")
        ranked = sorted(
            self._payloads, key=lambda d: self._vector_scores.get(d, 0.0), reverse=True
        )
        points = [
            SimpleNamespace(
                id=doc_id,
                score=self._vector_scores.get(doc_id, 0.0),
                payload=dict(self._payloads[doc_id]) if with_payload else {},
            )
            for doc_id in ranked[:limit]
        ]
        return SimpleNamespace(points=points)

    def scroll(self, collection_name, limit=100, offset=None, with_payload=True):  # noqa: ARG002
        self.scroll_calls += 1
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
        self.retrieve_calls += 1
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

    def set_payload(self, collection_name, payload, points, wait=True):  # noqa: ARG002
        by_point = {_point_id(doc_id): doc_id for doc_id in self._payloads}
        for point_id in points:
            doc_id = by_point.get(str(point_id))
            if doc_id is not None:
                self._payloads[doc_id].update(dict(payload))


# --- Canned retriever builders -------------------------------------------------

def make_hybrid_retriever(*, vector_scores=None, phase2: bool = True, w_ppr=None, **kwargs):
    """HybridRetriever over canned data (no live store, no LiteLLM).

    Injected: stub embeddings, canned Qdrant client, canned
    adjacency, canned doc counts/links/texts. ``vector_scores``
    controls the vector arm's fixed ranking (see
    ``CannedQdrantClient``); ``**kwargs`` pass through to
    ``HybridRetriever``.
    """
    from runtime.retrieval_hybrid import HybridRetriever

    build_kwargs: dict = {"phase2": phase2}
    if w_ppr is not None:
        build_kwargs["w_ppr"] = w_ppr
    build_kwargs.update(kwargs)
    return HybridRetriever(
        embeddings=StubEmbeddings(),
        client=CannedQdrantClient(vector_scores=vector_scores),
        adjacency=canned_adjacency(),
        doc_counts=dict(CANNED_DOC_COUNTS),
        doc_links={k: list(v) for k, v in CANNED_DOC_LINKS.items()},
        doc_texts=canned_doc_texts(),
        **build_kwargs,
    )


def make_expanded_retriever(*, vector_scores=None, phase2: bool = True, **kwargs):
    """ExpandedRetriever over canned data, with the stub generators."""
    from runtime.retrieval_expansion import (
        ExpandedRetriever,
        stub_decompose_question,
        stub_paraphrase_variants,
    )

    return ExpandedRetriever(
        make_hybrid_retriever(vector_scores=vector_scores, phase2=phase2, **kwargs),
        paraphrase_generator=stub_paraphrase_variants,
        decomposition_generator=stub_decompose_question,
    )
