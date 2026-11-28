"""Tests for ontology reasoning v1 (R1–R3) — fake store only, no live Neo4j.

The FakeStore records every (cypher, params) call and returns canned rows
keyed by the shape of the query, so each test pins the exact Cypher the
reasoner emits: MERGE conditioned on (source, type, target, base_ref),
DELETE by base_ref, and the BFS SUBCLASS_OF walk.
"""

from __future__ import annotations

import pytest

from ontology.reasoning import (
    INVERSE_RELATIONS,
    RULES,
    materialize_for_write,
    recompute_all,
    recompute_for_entity,
    remove_derived_for_base,
    subclass_ancestors,
)


class FakeStore:
    """Records (cypher, params); returns canned rows per query shape."""

    def __init__(
        self,
        *,
        subclass_map=None,
        transitive_out=None,
        transitive_in=None,
        base_edges=None,
        deleted_count=0,
    ):
        self.queries: list[tuple[str, dict]] = []
        self.subclass_map = subclass_map or {}
        self.transitive_out = transitive_out or {}
        self.transitive_in = transitive_in or {}
        self.base_edges = base_edges or []
        self.deleted_count = deleted_count

    def run_query(self, cypher, params=None):
        params = dict(params or {})
        self.queries.append((cypher, params))
        if "DELETE r" in cypher:
            return [{"deleted": self.deleted_count}]
        if "SUBCLASS_OF" in cypher:
            return [{"id": p} for p in self.subclass_map.get(params.get("class_id"), [])]
        if "[:PART_OF]->(c)" in cypher:
            return list(self.transitive_out.get(params.get("target_key"), []))
        if "(x)-[:PART_OF]->(s" in cypher:
            return list(self.transitive_in.get(params.get("source_key"), []))
        if "startNode(r)" in cypher:
            return list(self.base_edges)
        if "[r:PART_OF|HAS_COMPONENT]" in cypher:
            return list(self.base_edges)
        return []


def _merges(store):
    return [(c, p) for c, p in store.queries if "MERGE" in c]


def test_rules_catalog():
    assert {r["id"] for r in RULES} == {"R1", "R2", "R3"}
    assert all({"id", "name", "description"} <= set(r) for r in RULES)
    assert INVERSE_RELATIONS == {"PART_OF": "HAS_COMPONENT", "HAS_COMPONENT": "PART_OF"}


def test_materialize_part_of_inverse_and_transitive_both_directions():
    store = FakeStore(
        transitive_out={"B": [{"id": "C", "label": "Component"}]},
        transitive_in={"A": [{"id": "X", "label": "Component"}]},
    )
    created = materialize_for_write(
        store,
        relationship_type="PART_OF",
        source_type="Component",
        source_key="A",
        target_type="Component",
        target_key="B",
        base_ref="base-1",
    )
    merges = _merges(store)
    assert len(merges) == 3
    # Every MERGE carries the derived marker + rule + base_ref condition.
    for cypher, params in merges:
        assert "derived: true" in cypher
        assert "base_ref: $base_ref" in cypher
        assert params["base_ref"] == "base-1"
        assert params["rule"] in ("inverse:PART_OF", "transitive:PART_OF")
    # R2 inverse: B -HAS_COMPONENT-> A.
    inv_cypher, inv_params = merges[0]
    assert "HAS_COMPONENT" in inv_cypher
    assert inv_params == {
        "source_key": "B",
        "target_key": "A",
        "rule": "inverse:PART_OF",
        "base_ref": "base-1",
    }
    # R3 (a): A -PART_OF-> C; R3 (b): X -PART_OF-> B.
    assert {m[1]["rule"] for m in merges[1:]} == {"transitive:PART_OF"}
    assert created == [
        {
            "type": "HAS_COMPONENT",
            "source_type": "Component",
            "source_key": "B",
            "target_type": "Component",
            "target_key": "A",
            "rule": "inverse:PART_OF",
        },
        {
            "type": "PART_OF",
            "source_type": "Component",
            "source_key": "A",
            "target_type": "Component",
            "target_key": "C",
            "rule": "transitive:PART_OF",
        },
        {
            "type": "PART_OF",
            "source_type": "Component",
            "source_key": "X",
            "target_type": "Component",
            "target_key": "B",
            "rule": "transitive:PART_OF",
        },
    ]


def test_materialize_has_component_only_inverse_no_transitive():
    store = FakeStore()
    created = materialize_for_write(
        store,
        relationship_type="HAS_COMPONENT",
        source_type="Component",
        source_key="A",
        target_type="Component",
        target_key="B",
        base_ref="base-2",
    )
    merges = _merges(store)
    assert len(merges) == 1
    cypher, params = merges[0]
    assert "PART_OF" in cypher
    assert params["rule"] == "inverse:HAS_COMPONENT"
    assert params["source_key"] == "B" and params["target_key"] == "A"
    # No transitive neighbour lookups were issued at all.
    assert not any("[:PART_OF]->(c)" in c or "(x)-[:PART_OF]->(s" in c for c, _ in store.queries)
    assert created == [
        {
            "type": "PART_OF",
            "source_type": "Component",
            "source_key": "B",
            "target_type": "Component",
            "target_key": "A",
            "rule": "inverse:HAS_COMPONENT",
        }
    ]


def test_materialize_unrelated_type_derives_nothing():
    store = FakeStore()
    created = materialize_for_write(
        store,
        relationship_type="RELATED_TO",
        source_type="Document",
        source_key="D1",
        target_type="Asset",
        target_key="O21",
        base_ref="base-3",
    )
    assert created == []
    assert store.queries == []


def test_materialize_idempotent_same_base_ref():
    kwargs = dict(
        relationship_type="PART_OF",
        source_type="Component",
        source_key="A",
        target_type="Component",
        target_key="B",
        base_ref="base-1",
    )
    store = FakeStore(
        transitive_out={"B": [{"id": "C", "label": "Component"}]},
        transitive_in={"A": [{"id": "X", "label": "Component"}]},
    )
    first = materialize_for_write(store, **kwargs)
    second = materialize_for_write(store, **kwargs)
    assert first == second
    merges = _merges(store)
    # Same MERGE emitted twice (3 per call), each conditioned on base_ref
    # — so the second run matches the existing derived edges instead of
    # creating parallel duplicates.
    assert len(merges) == 6
    for cypher, params in merges:
        assert "base_ref: $base_ref" in cypher
        assert params["base_ref"] == "base-1"


def test_remove_derived_for_base():
    store = FakeStore(deleted_count=3)
    removed = remove_derived_for_base(store, base_ref="base-9")
    assert removed == 3
    assert len(store.queries) == 1
    cypher, params = store.queries[0]
    assert "DELETE r" in cypher
    assert "derived: true" in cypher
    assert "base_ref: $base_ref" in cypher
    assert params == {"base_ref": "base-9"}


def test_subclass_ancestors_cycle_terminates():
    store = FakeStore(subclass_map={"A": ["B"], "B": ["A", "C"], "C": []})
    ancestors = subclass_ancestors(store, "A")
    assert ancestors == ["B", "C"]
    assert "A" not in ancestors
    # One query per visited node (A, B, C) — the A->B->A cycle was not re-walked.
    assert len(store.queries) == 3
    assert all("SUBCLASS_OF" in c for c, _ in store.queries)


def test_recompute_all_deletes_before_rederiving():
    store = FakeStore(
        deleted_count=5,
        base_edges=[
            {
                "source_key": "A",
                "source_type": "Component",
                "rel_type": "PART_OF",
                "target_key": "B",
                "target_type": "Component",
                "base_ref": "ref-1",
            }
        ],
    )
    result = recompute_all(store)
    kinds = [
        "DELETE" if "DELETE r" in c else "MERGE" if "MERGE" in c else "READ"
        for c, _ in store.queries
    ]
    assert kinds[0] == "DELETE"
    assert kinds.index("MERGE") > kinds.index("DELETE")
    # The re-derived inverse edge reuses the base edge's stored base_ref.
    merges = _merges(store)
    assert len(merges) == 1
    assert merges[0][1]["base_ref"] == "ref-1"
    assert merges[0][1]["rule"] == "inverse:PART_OF"
    assert result == {
        "removed": 5,
        "created": [
            {
                "type": "HAS_COMPONENT",
                "source_type": "Component",
                "source_key": "B",
                "target_type": "Component",
                "target_key": "A",
                "rule": "inverse:PART_OF",
            }
        ],
    }


def test_recompute_for_entity():
    store = FakeStore(
        deleted_count=2,
        base_edges=[
            {
                "source_key": "A",
                "source_type": "Component",
                "rel_type": "PART_OF",
                "target_key": "B",
                "target_type": "Component",
                "base_ref": "ref-9",
            },
            {
                "source_key": "X",
                "source_type": "Component",
                "rel_type": "HAS_COMPONENT",
                "target_key": "A",
                "target_type": "Component",
                "base_ref": "ref-8",
            },
        ],
    )
    result = recompute_for_entity(store, entity_type="Component", entity_key="A")
    first_cypher, first_params = store.queries[0]
    assert "DELETE r" in first_cypher
    assert "derived: true" in first_cypher
    assert first_params == {"entity_key": "A"}
    assert result["removed"] == 2
    assert result["created"] == [
        {
            "type": "HAS_COMPONENT",
            "source_type": "Component",
            "source_key": "B",
            "target_type": "Component",
            "target_key": "A",
            "rule": "inverse:PART_OF",
        },
        {
            "type": "PART_OF",
            "source_type": "Component",
            "source_key": "A",
            "target_type": "Component",
            "target_key": "X",
            "rule": "inverse:HAS_COMPONENT",
        },
    ]


def test_invalid_identifier_raises_and_issues_no_query():
    store = FakeStore()
    with pytest.raises(ValueError):
        materialize_for_write(
            store,
            relationship_type="PART_OF",
            source_type="Bad Label",
            source_key="A",
            target_type="Component",
            target_key="B",
            base_ref="base-1",
        )
    with pytest.raises(ValueError):
        materialize_for_write(
            store,
            relationship_type="PART OF",
            source_type="Component",
            source_key="A",
            target_type="Component",
            target_key="B",
            base_ref="base-1",
        )
    with pytest.raises(ValueError):
        remove_derived_for_base(store, base_ref="")
    with pytest.raises(ValueError):
        recompute_for_entity(store, entity_type="Component; DROP", entity_key="A")
    assert store.queries == []


# ----------------------------------------------------------------------
# subclass_descendants (B5) — the downward mirror of
# subclass_ancestors, used by entities_by_class.
# ----------------------------------------------------------------------


def test_subclass_descendants_basic():
    from ontology.reasoning import subclass_descendants

    store = FakeStore(
        subclass_map={"A": ["B", "C"], "B": ["D"], "C": [], "D": []}
    )
    assert subclass_descendants(store, "A") == ["B", "C", "D"]
    assert subclass_descendants(store, "B") == ["D"]
    assert subclass_descendants(store, "D") == []


def test_subclass_descendants_cycle_terminates():
    from ontology.reasoning import subclass_descendants

    store = FakeStore(subclass_map={"A": ["B"], "B": ["A", "C"], "C": []})
    descendants = subclass_descendants(store, "A")
    assert descendants == ["B", "C"]
    assert "A" not in descendants
    # One query per visited node (A, B, C) — the A->B->A cycle was not re-walked.
    assert len(store.queries) == 3
    assert all("SUBCLASS_OF" in c for c, _ in store.queries)
