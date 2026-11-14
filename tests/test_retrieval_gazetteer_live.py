"""Phase 5 dynamic gazetteer tests — live ontology keys, no Neo4j.

Covers ``ontology/knowledge/graph_query.py::list_entity_keys`` and
the dynamic gazetteer in ``runtime/retrieval_graph.py``
(``refresh_gazetteer_codes`` / ``clear_gazetteer_codes`` /
``gazetteer_seeds`` live branch) against a fake GraphStore whose
``run_query`` returns canned keys. The fake store holds NON-demo
keys (``AHU-7``, ``pump-P5``) whose shapes the demo patterns do
not cover, so a link for them proves the patterns came from the
ontology list, not the demo tables. Demo-mode behaviour itself is
covered by the existing ``test_query_plan_linking.py`` /
``test_retrieval_phase1.py`` files, which must pass unmodified.
Main features only.
"""

from __future__ import annotations

import re

import pytest

from ontology.knowledge.graph_query import list_entity_keys
from runtime.retrieval_graph import (
    clear_gazetteer_codes,
    gazetteer_seeds,
    refresh_gazetteer_codes,
)


class FakeGraphStore:
    """Minimal GraphStore: canned ``id`` rows per label, or raises."""

    def __init__(self, keys_by_label=None, error=None):
        self.keys_by_label = keys_by_label or {}
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def run_query(self, cypher, params=None):
        self.calls.append((cypher, params or {}))
        if self.error is not None:
            raise self.error
        match = re.search(r"\(n:([A-Za-z][A-Za-z0-9_]*)\)", cypher)
        label = match.group(1) if match else ""
        return [{"id": key} for key in self.keys_by_label.get(label, [])]


LIVE_KEYS = {
    "Asset": ["O99", "AHU-7"],
    "Component": ["motor-M9", "pump-P5"],
    "Building": ["building-B9"],
    "Document": ["doc-new-guide"],
}


@pytest.fixture(autouse=True)
def _demo_mode():
    clear_gazetteer_codes()
    yield
    clear_gazetteer_codes()


def _seed_map(question: str) -> dict[str, float]:
    return {s["key"]: s["confidence"] for s in gazetteer_seeds(question)}


def _refresh(store=None):
    return refresh_gazetteer_codes(store=store or FakeGraphStore(LIVE_KEYS))


# --- list_entity_keys -----------------------------------------------------

def test_list_entity_keys_sorted_distinct_and_filters_bad_rows():
    store = FakeGraphStore({"Asset": ["O22", "O21", "O21", None, 7, ""]})
    assert list_entity_keys("Asset", store=store) == ["O21", "O22"]
    assert "MATCH (n:Asset)" in store.calls[0][0]
    assert "RETURN DISTINCT n.id AS id" in store.calls[0][0]


def test_list_entity_keys_limit_capped_to_200_and_floored_to_1():
    store = FakeGraphStore({"Asset": ["O21"]})
    list_entity_keys("Asset", limit=500, store=store)
    assert store.calls[-1][1]["limit"] == 200
    list_entity_keys("Asset", limit=0, store=store)
    assert store.calls[-1][1]["limit"] == 1


def test_list_entity_keys_rejects_invalid_label():
    store = FakeGraphStore({"Asset": ["O21"]})
    with pytest.raises(ValueError):
        list_entity_keys("Asset; DROP TABLE x", store=store)
    with pytest.raises(ValueError):
        list_entity_keys("123Asset", store=store)
    assert store.calls == []


# --- refresh ---------------------------------------------------------------

def test_refresh_returns_counts_and_source_ontology():
    assert _refresh() == {
        "assets": 2,
        "components": 2,
        "buildings": 1,
        "documents": 1,
        "source": "ontology",
    }


def test_refresh_never_raises_and_reports_unreachable():
    store = FakeGraphStore(error=RuntimeError("neo4j down"))
    assert refresh_gazetteer_codes(store=store) == {
        "assets": 0,
        "components": 0,
        "buildings": 0,
        "documents": 0,
        "source": "unreachable",
    }


# --- dynamic matching (live mode) -------------------------------------------

def test_dynamic_verbatim_match_of_non_demo_keys():
    _refresh()
    assert _seed_map("Check AHU-7.") == {"AHU-7": 1.0}
    assert _seed_map("Check pump-P5.") == {"pump-P5": 1.0}
    assert _seed_map("See doc-new-guide.") == {"doc-new-guide": 1.0}
    assert _seed_map("Elevator O99 is stuck.") == {"O99": 1.0}
    assert _seed_map("The motor-M9 overheated.") == {"motor-M9": 1.0}


def test_dynamic_variant_match_scores_0_8():
    _refresh()
    assert _seed_map("Check AHU 7.") == {"AHU-7": 0.8}
    assert _seed_map("Check ahu7.") == {"AHU-7": 0.8}
    assert _seed_map("Check pump P5.") == {"pump-P5": 0.8}
    assert _seed_map("Elevator O-99 is stuck.") == {"O99": 0.8}
    assert _seed_map("The motor M9 overheated.") == {"motor-M9": 0.8}
    assert _seed_map("See doc new guide.") == {"doc-new-guide": 0.8}
    assert _seed_map("In building B9.") == {"building-B9": 0.8}


def test_dynamic_verbatim_and_variant_twin_dedup_keeps_1():
    _refresh()
    assert _seed_map("AHU-7 and AHU 7") == {"AHU-7": 1.0}
    assert _seed_map("O99 and O-99") == {"O99": 1.0}


def test_dynamic_no_false_positive_on_embedded_keys():
    _refresh()
    assert gazetteer_seeds("XAHU-7 is not an asset.") == []
    assert gazetteer_seeds("AHU-7x is not an asset.") == []
    assert gazetteer_seeds("The pumpP5x unit.") == []


def test_dynamic_keys_do_not_link_in_demo_mode():
    # AHU-7 / pump-P5 shapes are outside the demo vocabulary, so
    # they link only after a refresh — proof the live patterns
    # come from the ontology list, not the demo tables.
    assert gazetteer_seeds("Check AHU-7.") == []
    assert gazetteer_seeds("Check pump P5.") == []


# --- degradation: live mode links nothing unlisted --------------------------

def test_live_mode_unlisted_demo_codes_link_nothing():
    _refresh()  # LIVE_KEYS has no O21/O22 and no Fault entities.
    assert gazetteer_seeds("Elevator O21 is stuck.") == []
    assert gazetteer_seeds("Elevator O22's motor M3 overheated.") == []
    assert gazetteer_seeds("Fault F-204 reported.") == []
    assert gazetteer_seeds("Fault F204 reported.") == []
    assert gazetteer_seeds("The door sensor triggers spuriously.") == []


def test_unreachable_refresh_links_nothing_in_live_mode():
    store = FakeGraphStore(error=RuntimeError("neo4j down"))
    assert refresh_gazetteer_codes(store=store)["source"] == "unreachable"
    assert gazetteer_seeds("Elevator O99 is stuck.") == []
    assert gazetteer_seeds("Elevator O21 is stuck.") == []
    assert gazetteer_seeds("Check AHU-7.") == []


def test_failed_refresh_clears_an_earlier_good_cache():
    _refresh()
    assert _seed_map("Check AHU-7.") == {"AHU-7": 1.0}
    store = FakeGraphStore(error=RuntimeError("neo4j down"))
    assert refresh_gazetteer_codes(store=store)["source"] == "unreachable"
    assert gazetteer_seeds("Check AHU-7.") == []


# --- refresh twice / clear ---------------------------------------------------

def test_second_refresh_replaces_cache_no_stale_keys():
    _refresh()
    assert _seed_map("Check AHU-7.") == {"AHU-7": 1.0}
    summary = refresh_gazetteer_codes(store=FakeGraphStore({"Asset": ["O77"]}))
    assert summary == {
        "assets": 1,
        "components": 0,
        "buildings": 0,
        "documents": 0,
        "source": "ontology",
    }
    assert gazetteer_seeds("Check AHU-7.") == []
    assert _seed_map("Elevator O77 is stuck.") == {"O77": 1.0}


def test_clear_returns_to_demo_mode():
    _refresh()
    assert gazetteer_seeds("Elevator O21 is stuck.") == []
    clear_gazetteer_codes()
    assert _seed_map("Elevator O21 is stuck.") == {"O21": 1.0}
    assert _seed_map("Elevator O-21 is stuck.") == {"O21": 0.8}
    assert gazetteer_seeds("Check AHU-7.") == []
