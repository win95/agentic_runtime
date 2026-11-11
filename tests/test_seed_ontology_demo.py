"""Ontology demo seed tests — endpoint properties, new assets, archived
O20, Document nodes, and the bare-endpoint upgrade path.

No live Neo4j: the projection drain runs against the in-memory fake
graph store from tests/ontology/test_projection_fake_store.py.
"""

from app.core.db import SessionLocal
from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.write_service import OntologyWriteService
from ontology.models import OntologyRelationship, OntologyVersion
from ontology.projection import provider as provider_module
from seeds.demo_data import RELATIONS
from seeds.seed_ontology_demo import (
    ENDPOINT_PROPERTIES,
    O21_PROPERTIES,
    seed_ontology_demo,
)
from tests.ontology.test_projection_fake_store import FakeGraphStore


def _repo(db):
    return CanonicalOntologyRepository(db)


def test_seed_ontology_demo_entities_and_relations(db_session, monkeypatch):
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)

    result = seed_ontology_demo()
    # O21 + 5 endpoints + 11 extra (B2/O22/M3/C2/O23/O24/O20/M0 + 3 docs).
    assert result["entities"] == 17
    assert result["relations"] == 15

    db = SessionLocal()
    try:
        repo = _repo(db)
        # Endpoint properties were filled in.
        b1 = repo.get_entity("Building", "building-B1")
        assert b1 is not None
        assert b1.properties == ENDPOINT_PROPERTIES[("Building", "building-B1")]
        motor = repo.get_entity("Component", "motor-M2")
        assert motor.properties["manufacturer"] == "Otis"
        # New building + assets.
        assert repo.get_entity("Building", "building-B2") is not None
        assert repo.get_entity("Asset", "O22") is not None
        assert repo.get_entity("Asset", "O23") is not None
        assert repo.get_entity("Asset", "O24") is not None
        # Archived asset + its component.
        o20 = repo.get_entity("Asset", "O20")
        assert o20 is not None
        assert o20.status == "archived"
        assert repo.get_entity("Component", "motor-M0") is not None
        # Document nodes + DOCUMENTED_BY edges (keys are Qdrant doc ids).
        doc = repo.get_entity("Document", "doc-door-sensor")
        assert doc is not None
        assert doc.properties["title"].startswith("Door sensor troubleshooting")
        assert (
            db.query(OntologyRelationship)
            .filter_by(relationship_type="DOCUMENTED_BY")
            .count()
            == 3
        )
        assert db.query(OntologyRelationship).count() == 15
    finally:
        db.close()

    # The archived asset never lands in the graph, and neither do its
    # relations (their endpoint is gone). Edge keys are
    # (source_label, source_id, rel_type, target_label, target_id).
    assert ("Asset", "O20") not in fake.nodes
    assert not any(e[1] == "O20" or e[4] == "O20" for e in fake.edges)
    assert ("Asset", "O22") in fake.nodes
    assert ("Document", "doc-door-sensor") in fake.nodes


def test_seed_ontology_demo_rerun_is_noop(db_session, monkeypatch):
    monkeypatch.setattr(provider_module, "_store", FakeGraphStore())
    seed_ontology_demo()

    db = SessionLocal()
    try:
        versions_after_first = db.query(OntologyVersion).count()
        entity = _repo(db).get_entity("Asset", "O21")
        assert entity.properties == O21_PROPERTIES
    finally:
        db.close()

    seed_ontology_demo()

    db = SessionLocal()
    try:
        assert db.query(OntologyVersion).count() == versions_after_first
        assert _repo(db).get_entity("Asset", "O21").properties == O21_PROPERTIES
    finally:
        db.close()


def test_endpoint_properties_upgrade_path(db_session, monkeypatch):
    """Old behavior: endpoints seeded bare (properties=None). Re-seed
    with the new module must FILL the properties in; a third run is a
    noop."""
    fake = FakeGraphStore()
    monkeypatch.setattr(provider_module, "_store", fake)

    # Simulate the old seed exactly: O21 with properties, endpoints bare.
    db = SessionLocal()
    try:
        writer = OntologyWriteService(_repo(db))
        prov = WriteProvenance(trace_id="seed-ontology-demo-old")
        writer.write_entity(
            entity_type="Asset", entity_key="O21",
            properties=O21_PROPERTIES, provenance=prov,
        )
        for from_id, from_label, rel_type, to_id, to_label in RELATIONS[:5]:
            writer.ensure_entity(
                entity_type=from_label, entity_key=from_id,
                properties=None, provenance=prov,
            )
            writer.ensure_entity(
                entity_type=to_label, entity_key=to_id,
                properties=None, provenance=prov,
            )
            writer.write_relationship(
                relationship_type=rel_type, source_type=from_label,
                source_key=from_id, target_type=to_label, target_key=to_id,
                properties=None, provenance=prov,
            )
        db.commit()
        b1_before = _repo(db).get_entity("Building", "building-B1")
        assert b1_before is not None
        assert b1_before.properties is None  # bare, as the old seed left it
    finally:
        db.close()

    seed_ontology_demo()

    db = SessionLocal()
    try:
        versions_after_upgrade = db.query(OntologyVersion).count()
        b1 = _repo(db).get_entity("Building", "building-B1")
        assert b1.properties == ENDPOINT_PROPERTIES[("Building", "building-B1")]
        motor = _repo(db).get_entity("Component", "motor-M2")
        assert motor.properties["manufacturer"] == "Otis"
    finally:
        db.close()

    seed_ontology_demo()

    db = SessionLocal()
    try:
        assert db.query(OntologyVersion).count() == versions_after_upgrade
    finally:
        db.close()
