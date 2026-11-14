"""Qdrant payload node linking — Phase 1 (report §3.1, §8 Phase 1).

Every Qdrant point carries ``node_ids[]`` (+ ``asset_id`` /
``component_id`` / ``building_id`` where known) in its payload, so a
vector hit is already an entity link — step 4 of the pipeline is a
payload read, not a model call.

Grounding: the links below are read off ``seeds/demo_data.py`` —
which asset/component each document is about, plus the
``DOCUMENTED_BY`` relations in ``RELATIONS``. The demo corpus has
7 documents; only 3 are ontology ``Document`` nodes (``doc-door-sensor``,
``doc-motor-overheat``, ``doc-brake-inspection``, all
``DOCUMENTED_BY`` from O21). The other 4 have no Document node, so
their links name the Asset/Component/Building they are about instead:

- ``doc-controller-errors`` — O21's controller C1 (F-410 / F-103 in
  the text); no Document node.
- ``doc-leveling`` — O21 (F-102); no leveling component node exists.
- ``doc-rope-inspection`` — O21's traction system (the doc is one of
  O21's 6 seeded technical documents; no rope component node exists).
- ``doc-o23-overview`` — O23 in building B2 (title + text name it);
  no Document node.

Deliberate asymmetry (golden-set case M05): ``doc-motor-overheat``
names F-204, O21's fault, so it links to O21 / motor-M2 only — NOT
to O22 / motor-M3, even though the remedy text also fits O22's F-206.
Linking reflects what the document is *about* / linked to in the
ontology, never a forced fit to a question. A graph-only retriever
that follows O22's links therefore misses this doc; the fusion
(``runtime/retrieval_fusion.py``) keeps the vector arm so it still
surfaces.
"""

from __future__ import annotations

# doc_id -> payload link fields. ``node_ids`` always includes the
# document's own id when it is an ontology Document node, plus every
# Asset/Component/Building the chunk is about.
DOCUMENT_LINKS: dict[str, dict] = {
    "doc-door-sensor": {
        "node_ids": ["O21", "door-sensor-DS1", "building-B1", "doc-door-sensor"],
        "asset_id": "O21",
        "component_id": "door-sensor-DS1",
        "building_id": "building-B1",
    },
    "doc-motor-overheat": {
        "node_ids": ["O21", "motor-M2", "building-B1", "doc-motor-overheat"],
        "asset_id": "O21",
        "component_id": "motor-M2",
        "building_id": "building-B1",
    },
    "doc-brake-inspection": {
        "node_ids": ["O21", "brake-BK1", "building-B1", "doc-brake-inspection"],
        "asset_id": "O21",
        "component_id": "brake-BK1",
        "building_id": "building-B1",
    },
    "doc-controller-errors": {
        "node_ids": ["O21", "controller-C1", "building-B1"],
        "asset_id": "O21",
        "component_id": "controller-C1",
        "building_id": "building-B1",
    },
    "doc-leveling": {
        "node_ids": ["O21", "building-B1"],
        "asset_id": "O21",
        "component_id": None,
        "building_id": "building-B1",
    },
    "doc-rope-inspection": {
        "node_ids": ["O21", "building-B1"],
        "asset_id": "O21",
        "component_id": None,
        "building_id": "building-B1",
    },
    "doc-o23-overview": {
        "node_ids": ["O23", "building-B2"],
        "asset_id": "O23",
        "component_id": None,
        "building_id": "building-B2",
    },
}


def compute_document_links(doc: dict) -> dict:
    """Payload link fields for one document dict (``id``/``title``/``content``).

    Known demo docs return the grounded table above. Unknown docs get
    no links (``node_ids: []``) — the caller must not invent a link;
    an unlinked doc simply scores ``graph_score = 0`` and the vector
    arm carries it (guardrail 3 fallback).
    """
    links = DOCUMENT_LINKS.get(doc.get("id", ""))
    if links is None:
        return {"node_ids": [], "asset_id": None, "component_id": None, "building_id": None}
    return {
        "node_ids": list(links["node_ids"]),
        "asset_id": links["asset_id"],
        "component_id": links["component_id"],
        "building_id": links["building_id"],
    }


def with_document_links(doc: dict) -> dict:
    """Return a copy of ``doc`` with payload link fields attached.

    Existing explicit link fields win over the table, so a caller that
    already knows a doc's links is never overwritten.
    """
    out = dict(doc)
    computed = compute_document_links(doc)
    for key, value in computed.items():
        if key not in out or out[key] in (None, [], ""):
            out[key] = value
    return out


def linking_coverage(docs: list[dict]) -> dict:
    """Linking coverage over a doc list: share with >=1 ``node_id``.

    Phase 1 target (locked decision): >= 80% of chunks with >=1 node_id.
    """
    if not docs:
        return {"n_docs": 0, "n_linked": 0, "coverage": 0.0}
    linked = 0
    for doc in docs:
        node_ids = doc.get("node_ids")
        if node_ids is None:
            node_ids = compute_document_links(doc)["node_ids"]
        if node_ids:
            linked += 1
    return {
        "n_docs": len(docs),
        "n_linked": linked,
        "coverage": linked / len(docs),
    }
