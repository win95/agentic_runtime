"""Ontology metadata (B2): CRUD + cache + validation, against an
in-memory fake graph store.

``MetadataFakeGraphStore`` extends the projection fake with a small
Cypher interpreter for the metadata queries in
``ontology/metadata/store.py`` (and the ``SUBCLASS_OF`` walk in
``ontology/reasoning/materialize.py``). It records every query, so
cache behaviour is assertable. Anything it does not recognise
returns [] — the same as the plain fake.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from app.core.db import SessionLocal, init_db
from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.write_service import OntologyWriteService
from ontology.metadata import store as metadata_store
from ontology.metadata.validate import validate_entity, validate_relationship
from tests.ontology.test_projection_fake_store import FakeGraphStore


class MetadataFakeGraphStore(FakeGraphStore):
    """FakeGraphStore + in-memory OntologyClass / ObjectProperty /
    DataProperty nodes and SUBCLASS_OF edges, driven by run_query."""

    def __init__(self, *, healthy: bool = True) -> None:
        super().__init__(healthy=healthy)
        self.queries: list[tuple[str, dict[str, Any] | None]] = []
        self.meta_classes: dict[str, dict[str, Any]] = {}
        self.meta_objprops: dict[str, dict[str, Any]] = {}
        self.meta_dataprops: dict[tuple[str, str], dict[str, Any]] = {}
        self.subclasses: set[tuple[str, str]] = set()

    # -- helpers ------------------------------------------------------
    def _class_row(self, class_id: str) -> dict[str, Any] | None:
        row = self.meta_classes.get(class_id)
        return dict(row) if row is not None else None

    def _objprop_row(self, property_id: str) -> dict[str, Any] | None:
        row = self.meta_objprops.get(property_id)
        return dict(row) if row is not None else None

    def _dataprop_rows(self, property_id: str) -> list[dict[str, Any]]:
        rows = [
            dict(row)
            for (key, _), row in sorted(self.meta_dataprops.items())
            if key == property_id
        ]
        return sorted(rows, key=lambda r: str(r.get("domain") or ""))

    # -- run_query interpreter ----------------------------------------
    def run_query(self, cypher: str, params: dict[str, Any] | None = None) -> list:
        self.queries.append((cypher, params))
        params = params or {}

        if "SUBCLASS_OF" in cypher:
            if "RETURN p.id AS id" in cypher:
                # subclass_ancestors: parents of params["class_id"].
                return [
                    {"id": parent}
                    for child, parent in sorted(self.subclasses)
                    if child == params.get("class_id")
                ]
            if "count(child)" in cypher:
                return [
                    {
                        "n": sum(
                            1 for _, parent in self.subclasses
                            if parent == params.get("id")
                        )
                    }
                ]
            if "DELETE r" in cypher:
                self.subclasses.discard(
                    (params.get("child_id"), params.get("parent_id"))
                )
                return []
            if "MERGE (c)-[:SUBCLASS_OF]->(p)" in cypher:
                self.subclasses.add(
                    (params.get("child_id"), params.get("parent_id"))
                )
                return []
            return []

        if "OntologyClass" in cypher:
            if "DETACH DELETE" in cypher:
                class_id = params.get("id")
                self.meta_classes.pop(class_id, None)
                self.subclasses = {
                    (c, p) for c, p in self.subclasses
                    if c != class_id and p != class_id
                }
                return []
            if "MERGE (c:OntologyClass" in cypher:
                row = {
                    "id": params.get("id"),
                    "name": params.get("name"),
                    "description": params.get("description"),
                }
                self.meta_classes[row["id"]] = row
                return [dict(row)]
            if "SET" in cypher:
                row = self._class_row(params.get("id"))
                if row is None:
                    return []
                if "name" in params:
                    row["name"] = params["name"]
                if "description" in params:
                    row["description"] = params["description"]
                self.meta_classes[row["id"]] = row
                return [dict(row)]
            if "{id: $id}" in cypher:
                row = self._class_row(params.get("id"))
                return [row] if row is not None else []
            return [dict(row) for _, row in sorted(self.meta_classes.items())]

        if "ObjectProperty" in cypher:
            if "MERGE (p:ObjectProperty" in cypher:
                row = {
                    "id": params.get("id"),
                    "domain": params.get("domain"),
                    "range": params.get("range"),
                    "inverse_of": params.get("inverse_of"),
                }
                self.meta_objprops[row["id"]] = row
                return [dict(row)]
            if "SET" in cypher:
                row = self._objprop_row(params.get("id"))
                if row is None:
                    return []
                for key in ("domain", "range", "inverse_of"):
                    if key in params:
                        row[key] = params[key]
                self.meta_objprops[row["id"]] = row
                return [dict(row)]
            if "DELETE p" in cypher:
                row = self._objprop_row(params.get("id"))
                self.meta_objprops.pop(params.get("id"), None)
                return [row] if row is not None else []
            if "{id: $id}" in cypher:
                row = self._objprop_row(params.get("id"))
                return [row] if row is not None else []
            return [dict(row) for _, row in sorted(self.meta_objprops.items())]

        if "DataProperty" in cypher:
            if "MERGE (d:DataProperty" in cypher:
                row = {
                    "id": params.get("id"),
                    "domain": params.get("domain"),
                    "datatype": params.get("datatype"),
                    "required": bool(params.get("required")),
                }
                self.meta_dataprops[(row["id"], row["domain"])] = row
                return [dict(row)]
            if "SET" in cypher:
                rows = self._dataprop_rows(params.get("id"))
                if "domain" in params and "WHERE d.domain" in cypher:
                    rows = [r for r in rows if r.get("domain") == params["domain"]]
                if not rows:
                    return []
                for row in rows:
                    if "datatype" in params:
                        row["datatype"] = params["datatype"]
                    if "required" in params:
                        row["required"] = bool(params["required"])
                    self.meta_dataprops[(row["id"], row["domain"])] = row
                return [dict(r) for r in rows]
            if "DELETE d" in cypher:
                rows = self._dataprop_rows(params.get("id"))
                for row in rows:
                    self.meta_dataprops.pop((row["id"], row["domain"]), None)
                return []
            if "{id: $id, domain: $domain}" in cypher:
                row = self.meta_dataprops.get(
                    (params.get("id"), params.get("domain"))
                )
                return [dict(row)] if row is not None else []
            if "{id: $id}" in cypher:
                return self._dataprop_rows(params.get("id"))
            return [
                dict(row)
                for _, row in sorted(
                    self.meta_dataprops.items(),
                    key=lambda item: (str(item[1].get("domain") or ""), item[0][0]),
                )
            ]

        if cypher.startswith("MATCH ()-[r:") and "count(r)" in cypher:
            match = re.search(r"\[r:([A-Za-z][A-Za-z0-9_]*)\]", cypher)
            rel_type = match.group(1) if match else ""
            return [
                {
                    "n": sum(
                        1 for key in self.edges if key[2] == rel_type
                    )
                }
            ]

        if "RETURN count(n)" in cypher:
            match = re.search(r"MATCH \(n:([A-Za-z][A-Za-z0-9_]*)\)", cypher)
            label = match.group(1) if match else ""
            return [
                {"n": sum(1 for (node_label, _) in self.nodes if node_label == label)}
            ]

        return []


@pytest.fixture(autouse=True)
def _clear_metadata_cache():
    metadata_store.clear_metadata_cache()
    yield
    metadata_store.clear_metadata_cache()


@pytest.fixture()
def db():
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def store():
    return MetadataFakeGraphStore()


def _seed_v1_metadata(store: MetadataFakeGraphStore) -> None:
    for class_id in ("Asset", "Building", "Component", "Document"):
        metadata_store.create_class(store, class_id=class_id, name=class_id)
    metadata_store.create_object_property(
        store, property_id="PART_OF", domain="Asset", range="Building",
        inverse_of="HAS_COMPONENT",
    )
    metadata_store.create_object_property(
        store, property_id="HAS_COMPONENT", domain="Asset", range="Component",
        inverse_of="PART_OF",
    )
    metadata_store.create_object_property(
        store, property_id="RELATED_TO", domain="any", range="any",
    )
    metadata_store.create_object_property(
        store, property_id="DOCUMENTED_BY", domain="Asset", range="Document",
    )


# ----------------------------------------------------------------------
# Metadata CRUD
# ----------------------------------------------------------------------


def test_metadata_class_crud(store):
    assert metadata_store.list_classes(store) == []
    created = metadata_store.create_class(
        store, class_id="Asset", name="Asset", description="elevator"
    )
    assert created == {"id": "Asset", "name": "Asset", "description": "elevator"}
    assert metadata_store.get_class(store, "Asset") == created
    assert metadata_store.get_class(store, "Missing") is None
    # Re-create is idempotent (MERGE): still one node, fields updated.
    metadata_store.create_class(store, class_id="Asset", name="Asset v2")
    assert len(metadata_store.list_classes(store)) == 1
    assert metadata_store.get_class(store, "Asset")["name"] == "Asset v2"
    updated = metadata_store.update_class(store, class_id="Asset", name="Asset v3")
    assert updated["name"] == "Asset v3"
    assert updated["description"] is None  # create overwrote it above
    with pytest.raises(ValueError, match="not found"):
        metadata_store.update_class(store, class_id="Missing", name="x")
    deleted = metadata_store.delete_class(store, "Asset")
    assert deleted["id"] == "Asset"
    assert metadata_store.list_classes(store) == []
    with pytest.raises(ValueError, match="not found"):
        metadata_store.delete_class(store, "Asset")
    with pytest.raises(ValueError, match="Invalid Cypher identifier"):
        metadata_store.create_class(store, class_id="Bad Name!", name="x")


def test_metadata_object_property_crud(store):
    created = metadata_store.create_object_property(
        store, property_id="PART_OF", domain="Asset", range="Building",
        inverse_of="HAS_COMPONENT",
    )
    assert created["inverse_of"] == "HAS_COMPONENT"
    assert metadata_store.get_object_property(store, "PART_OF") == created
    assert metadata_store.get_object_property(store, "Missing") is None
    assert len(metadata_store.list_object_properties(store)) == 1
    updated = metadata_store.update_object_property(
        store, property_id="PART_OF", range="Component"
    )
    assert updated["range"] == "Component"
    assert updated["domain"] == "Asset"  # untouched field kept
    with pytest.raises(ValueError, match="not found"):
        metadata_store.update_object_property(store, property_id="Missing")
    deleted = metadata_store.delete_object_property(store, "PART_OF")
    assert deleted["id"] == "PART_OF"
    assert metadata_store.list_object_properties(store) == []
    with pytest.raises(ValueError, match="not found"):
        metadata_store.delete_object_property(store, "PART_OF")


def test_metadata_data_property_crud_same_key_several_domains(store):
    # Same key for two classes: two definitions, MERGE key (id, domain).
    metadata_store.create_data_property(
        store, property_id="name", domain="Asset", datatype="str"
    )
    metadata_store.create_data_property(
        store, property_id="name", domain="Building", datatype="str"
    )
    assert len(metadata_store.list_data_properties(store)) == 2
    # Update narrowed to one domain leaves the other alone.
    updated = metadata_store.update_data_property(
        store, property_id="name", domain="Building", datatype="int"
    )
    assert updated == {
        "id": "name", "domain": "Building", "datatype": "int", "required": False,
    }
    rows = metadata_store.list_data_properties(store)
    assert next(r for r in rows if r["domain"] == "Asset")["datatype"] == "str"
    with pytest.raises(ValueError, match="not found"):
        metadata_store.update_data_property(
            store, property_id="name", domain="Document", datatype="int"
        )
    with pytest.raises(ValueError, match="Invalid datatype"):
        metadata_store.create_data_property(
            store, property_id="x", domain="Asset", datatype="date"
        )
    # Delete removes every definition of the key.
    metadata_store.delete_data_property(store, "name")
    assert metadata_store.list_data_properties(store) == []
    with pytest.raises(ValueError, match="not found"):
        metadata_store.delete_data_property(store, "name")


def test_add_subclass_rejects_direct_cycle(store):
    metadata_store.create_class(store, class_id="A", name="A")
    metadata_store.create_class(store, class_id="B", name="B")
    metadata_store.add_subclass(store, child_id="B", parent_id="A")
    with pytest.raises(ValueError, match="cycle"):
        metadata_store.add_subclass(store, child_id="A", parent_id="B")
    with pytest.raises(ValueError, match="itself"):
        metadata_store.add_subclass(store, child_id="A", parent_id="A")
    with pytest.raises(ValueError, match="not found"):
        metadata_store.add_subclass(store, child_id="A", parent_id="Missing")


def test_add_subclass_rejects_indirect_cycle_and_is_idempotent(store):
    for class_id in ("A", "B", "C"):
        metadata_store.create_class(store, class_id=class_id, name=class_id)
    metadata_store.add_subclass(store, child_id="B", parent_id="A")
    metadata_store.add_subclass(store, child_id="C", parent_id="B")
    # A -> C would close the loop C -> B -> A.
    with pytest.raises(ValueError, match="cycle"):
        metadata_store.add_subclass(store, child_id="A", parent_id="C")
    # Re-adding the same edge is a noop, not an error.
    metadata_store.add_subclass(store, child_id="C", parent_id="B")
    assert store.subclasses == {("B", "A"), ("C", "B")}
    removed = metadata_store.remove_subclass(store, child_id="C", parent_id="B")
    assert removed == {"child_id": "C", "parent_id": "B"}
    assert store.subclasses == {("B", "A")}


def test_delete_class_refused_with_subclass(store):
    metadata_store.create_class(store, class_id="P", name="P")
    metadata_store.create_class(store, class_id="C", name="C")
    metadata_store.add_subclass(store, child_id="C", parent_id="P")
    with pytest.raises(ValueError, match="subclass"):
        metadata_store.delete_class(store, "P")
    metadata_store.remove_subclass(store, child_id="C", parent_id="P")
    metadata_store.delete_class(store, "P")
    assert metadata_store.get_class(store, "P") is None


def test_delete_class_refused_with_instance(store):
    metadata_store.create_class(store, class_id="Asset", name="Asset")
    store.merge_node("Asset", "O21", {"name": "lift"})
    with pytest.raises(ValueError, match="instance"):
        metadata_store.delete_class(store, "Asset")
    store.delete_node("Asset", "O21")
    metadata_store.delete_class(store, "Asset")
    assert metadata_store.get_class(store, "Asset") is None


def test_delete_object_property_refused_when_in_use(store):
    metadata_store.create_object_property(
        store, property_id="PART_OF", domain="Asset", range="Building"
    )
    store.merge_node("Asset", "O21", {"name": "lift"})
    store.merge_node("Building", "B1", {"name": "tower"})
    store.merge_relationship("Asset", "O21", "PART_OF", "Building", "B1", {})
    with pytest.raises(ValueError, match="still used"):
        metadata_store.delete_object_property(store, "PART_OF")
    store.delete_relationship("Asset", "O21", "PART_OF", "Building", "B1")
    metadata_store.delete_object_property(store, "PART_OF")
    assert metadata_store.get_object_property(store, "PART_OF") is None


def test_metadata_cache_hit_and_invalidate(store):
    metadata_store.get_metadata(store)
    assert len(store.queries) == 3  # one list query per metadata kind
    metadata_store.get_metadata(store)
    assert len(store.queries) == 3  # cache hit: no new queries
    metadata_store.create_class(store, class_id="Asset", name="Asset")
    metadata_store.get_metadata(store)
    assert len(store.queries) == 3 + 1 + 3  # create + re-list after invalidate
    metadata_store.clear_metadata_cache()
    metadata_store.get_metadata(store, force_refresh=True)
    assert len(store.queries) == 3 + 1 + 3 + 3


def test_metadata_cache_is_per_store():
    first = MetadataFakeGraphStore()
    second = MetadataFakeGraphStore()
    metadata_store.create_class(first, class_id="Asset", name="Asset")
    assert [c["id"] for c in metadata_store.get_metadata(first)["classes"]] == [
        "Asset"
    ]
    # The second store never saw that class: no leak across stores.
    assert metadata_store.get_metadata(second)["classes"] == []


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_validate_entity_unknown_class_rejected(store):
    _seed_v1_metadata(store)
    with pytest.raises(ValueError, match="not a defined class"):
        validate_entity(store, entity_type="Spaceship", properties={})
    validate_entity(store, entity_type="Asset", properties={})


def test_validate_entity_datatype_and_required(store):
    _seed_v1_metadata(store)
    metadata_store.create_data_property(
        store, property_id="name", domain="Asset", datatype="str", required=True
    )
    metadata_store.create_data_property(
        store, property_id="floors", domain="Building", datatype="int"
    )
    with pytest.raises(ValueError, match="must be str"):
        validate_entity(store, entity_type="Asset", properties={"name": 123})
    with pytest.raises(ValueError, match="required data property 'name'"):
        validate_entity(store, entity_type="Asset", properties={})
    validate_entity(store, entity_type="Asset", properties={"name": "lift"})
    # "floors" is defined, but for Building — for Asset it is unknown.
    with pytest.raises(ValueError, match="unknown data propert"):
        validate_entity(
            store, entity_type="Asset", properties={"name": "lift", "floors": 2}
        )
    validate_entity(store, entity_type="Building", properties={"floors": 20})
    with pytest.raises(ValueError, match="must be int"):
        validate_entity(store, entity_type="Building", properties={"floors": "20"})


def test_validate_entity_fallback_v1_when_no_metadata():
    plain = FakeGraphStore()
    # No OntologyClass node anywhere: v1 allowlist applies, and no
    # data-property checks exist to reject unknown keys.
    validate_entity(plain, entity_type="Asset", properties={"anything": 1})
    with pytest.raises(ValueError, match="not allowed in v1"):
        validate_entity(plain, entity_type="Spaceship", properties={})
    validate_relationship(
        plain,
        relationship_type="PART_OF",
        source_type="Asset",
        source_key="O21",
        target_type="Building",
        target_key="B1",
    )
    with pytest.raises(ValueError, match="not allowed in v1"):
        validate_relationship(
            plain,
            relationship_type="FOO",
            source_type="Asset",
            source_key="O21",
            target_type="Building",
            target_key="B1",
        )
    with pytest.raises(ValueError, match="not allowed in v1"):
        validate_relationship(
            plain,
            relationship_type="PART_OF",
            source_type="Asset",
            source_key="O21",
            target_type="Spaceship",
            target_key="S1",
        )


def test_validate_entity_unknown_key_rule_only_when_defined(store):
    _seed_v1_metadata(store)
    metadata_store.create_data_property(
        store, property_id="name", domain="Asset", datatype="str"
    )
    # Asset has a defined data property: stray keys are rejected.
    with pytest.raises(ValueError, match="unknown data propert"):
        validate_entity(store, entity_type="Asset", properties={"bogus": 1})
    # Document has none defined: legacy keys are not rejected.
    validate_entity(store, entity_type="Document", properties={"anything": 1})
    # System keys are never unknown (motor-M0 carries status=retired).
    validate_entity(
        store, entity_type="Asset", properties={"name": "x", "status": "retired"}
    )
    # properties=None (relation-stub write): only the class is checked.
    validate_entity(store, entity_type="Asset", properties=None)


def test_validate_entity_subclass_data_property(store):
    _seed_v1_metadata(store)
    metadata_store.create_class(store, class_id="SubComp", name="SubComp")
    metadata_store.add_subclass(store, child_id="SubComp", parent_id="Component")
    metadata_store.create_data_property(
        store, property_id="temperature", domain="Component", datatype="float"
    )
    # Domain is an ancestor of the class: the property applies.
    validate_entity(store, entity_type="SubComp", properties={"temperature": 1.5})
    validate_entity(store, entity_type="SubComp", properties={"temperature": 2})
    with pytest.raises(ValueError, match="must be float"):
        validate_entity(
            store, entity_type="SubComp", properties={"temperature": "hot"}
        )
    # bool is not an int/float, even though bool subclasses int.
    with pytest.raises(ValueError, match="must be float"):
        validate_entity(store, entity_type="SubComp", properties={"temperature": True})


def test_validate_relationship_domain_range(store):
    _seed_v1_metadata(store)
    store.merge_node("Asset", "O21", {"name": "lift"})
    store.merge_node("Building", "B1", {"name": "tower"})
    store.merge_node("Component", "M1", {"name": "motor"})
    validate_relationship(
        store,
        relationship_type="PART_OF",
        source_type="Asset",
        source_key="O21",
        target_type="Building",
        target_key="B1",
    )
    with pytest.raises(ValueError, match="not a Asset"):
        validate_relationship(
            store,
            relationship_type="PART_OF",
            source_type="Component",
            source_key="M1",
            target_type="Building",
            target_key="B1",
        )
    with pytest.raises(ValueError, match="not a Building"):
        validate_relationship(
            store,
            relationship_type="PART_OF",
            source_type="Asset",
            source_key="O21",
            target_type="Component",
            target_key="M1",
        )
    # Missing endpoint: that side is skipped, not verified.
    validate_relationship(
        store,
        relationship_type="PART_OF",
        source_type="Asset",
        source_key="O21",
        target_type="Building",
        target_key="B-missing",
    )
    # RELATED_TO is any/any: always passes.
    validate_relationship(
        store,
        relationship_type="RELATED_TO",
        source_type="Component",
        source_key="M1",
        target_type="Asset",
        target_key="O21",
    )


def test_validate_relationship_unknown_type_rejected(store):
    _seed_v1_metadata(store)
    with pytest.raises(ValueError, match="not a defined object property"):
        validate_relationship(
            store,
            relationship_type="FOO",
            source_type="Asset",
            source_key="O21",
            target_type="Building",
            target_key="B1",
        )


# ----------------------------------------------------------------------
# Validation wired into the write service
# ----------------------------------------------------------------------


def test_write_service_validates_against_metadata(db, store):
    _seed_v1_metadata(store)
    metadata_store.create_data_property(
        store, property_id="name", domain="Asset", datatype="str"
    )
    writer = OntologyWriteService(CanonicalOntologyRepository(db), store=store)
    prov = WriteProvenance(trace_id="meta-write-test")
    with pytest.raises(ValueError, match="not a defined class"):
        writer.write_entity(
            entity_type="Spaceship",
            entity_key="S1",
            properties={},
            provenance=prov,
        )
    with pytest.raises(ValueError, match="must be str"):
        writer.write_entity(
            entity_type="Asset",
            entity_key="O21",
            properties={"name": 123},
            provenance=prov,
        )
    result = writer.write_entity(
        entity_type="Asset",
        entity_key="O21",
        properties={"name": "lift"},
        provenance=prov,
    )
    assert result.applied
    db.commit()
    # Domain/range is enforced on relationships too.
    writer.write_entity(
        entity_type="Component",
        entity_key="M1",
        properties={},
        provenance=prov,
    )
    writer.write_entity(
        entity_type="Building",
        entity_key="B1",
        properties={},
        provenance=prov,
    )
    db.commit()
    with pytest.raises(ValueError, match="not a Asset"):
        writer.write_relationship(
            relationship_type="PART_OF",
            source_type="Component",
            source_key="M1",
            target_type="Building",
            target_key="B1",
            properties=None,
            provenance=prov,
        )


def test_write_service_falls_back_without_metadata(db):
    plain = FakeGraphStore()
    writer = OntologyWriteService(CanonicalOntologyRepository(db), store=plain)
    prov = WriteProvenance(trace_id="meta-fallback-test")
    result = writer.write_entity(
        entity_type="Asset",
        entity_key="O21",
        properties={"name": "lift"},
        provenance=prov,
    )
    assert result.applied
    with pytest.raises(ValueError, match="not allowed in v1"):
        writer.write_entity(
            entity_type="Spaceship",
            entity_key="S1",
            properties={},
            provenance=prov,
        )
