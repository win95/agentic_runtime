"""Seed canonical ontology + Neo4j projection for the demo assets.

O21 (building B1) is the original demo asset; O22/O23/O24 live in the
new building B2; O20 is an archived elevator in B1; three Document
nodes are linked to O21 by DOCUMENTED_BY (their keys are the Qdrant
doc ids, which is what `linked_documents` matches on).
"""

from __future__ import annotations

import logging

from app.core.db import SessionLocal
from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.write_service import OntologyWriteService
from ontology.metadata.store import (
    create_class,
    create_data_property,
    create_object_property,
)
from ontology.projection.engine import drain_pending_projections
from ontology.projection.provider import get_graph_store
from seeds.demo_data import ASSET_ID, RELATIONS

logger = logging.getLogger(__name__)

O21_PROPERTIES = {
    "name": "Passenger elevator O21",
    "building_id": "building-B1",
    "asset_class": "elevator",
}

# Real properties for the endpoints that were first seeded bare
# (properties=None via ensure_entity). They are written with explicit
# dicts via write_entity — not ensure_entity — so re-running the seed
# on an already-seeded DB FILLS them in, and a further run is a noop
# (state hash matches, see tests/test_seed_ontology_demo.py).
ENDPOINT_PROPERTIES = {
    ("Building", "building-B1"): {
        "name": "Building B1 — Daeha Business Center",
        "address": "360 Kim Ma, Ba Dinh, Hanoi",
        "floors": 20,
        "asset_class": "building",
    },
    ("Component", "motor-M2"): {
        "name": "Traction motor M2",
        "component_class": "traction_motor",
        "manufacturer": "Otis",
        "installed": "2019-03-15",
    },
    ("Component", "door-sensor-DS1"): {
        "name": "Door sensor DS1",
        "component_class": "door_sensor",
        "location": "car door",
    },
    ("Component", "controller-C1"): {
        "name": "Controller C1",
        "component_class": "controller",
        "firmware": "v2.4.1",
    },
    ("Component", "brake-BK1"): {
        "name": "Brake BK1",
        "component_class": "brake",
        "last_pad_change": "2026-10-14",
    },
}

# (entity_type, entity_key, properties, status)
EXTRA_ENTITIES = [
    ("Building", "building-B2", {
        "name": "Building B2 — The Nine",
        "address": "Cau Giay, Hanoi",
        "floors": 18,
        "asset_class": "building",
    }, "active"),
    ("Asset", "O22", {
        "name": "Passenger elevator O22",
        "building_id": "building-B2",
        "asset_class": "elevator",
    }, "active"),
    ("Component", "motor-M3", {
        "name": "Traction motor M3",
        "component_class": "traction_motor",
        "manufacturer": "Otis",
        "installed": "2020-06-01",
    }, "active"),
    ("Component", "controller-C2", {
        "name": "Controller C2",
        "component_class": "controller",
        "firmware": "v2.4.1",
    }, "active"),
    ("Asset", "O23", {
        "name": "Passenger elevator O23",
        "building_id": "building-B2",
        "asset_class": "elevator",
    }, "active"),
    ("Asset", "O24", {
        "name": "Passenger elevator O24",
        "building_id": "building-B2",
        "asset_class": "elevator",
    }, "active"),
    ("Asset", "O20", {
        "name": "Freight elevator O20 (retired)",
        "building_id": "building-B1",
        "asset_class": "elevator",
    }, "archived"),
    ("Component", "motor-M0", {
        "name": "Traction motor M0",
        "component_class": "traction_motor",
        "status": "retired",
    }, "active"),
    ("Document", "doc-door-sensor", {
        "title": "Door sensor troubleshooting — passenger elevators",
        "category": "door_system",
    }, "active"),
    ("Document", "doc-motor-overheat", {
        "title": "Traction motor overheating — causes and remedies",
        "category": "traction",
    }, "active"),
    ("Document", "doc-brake-inspection", {
        "title": "Brake system inspection — lining wear limits",
        "category": "brake_system",
    }, "active"),
]

ENTITY_PROPERTIES = {
    ("Asset", ASSET_ID): O21_PROPERTIES,
    **ENDPOINT_PROPERTIES,
    **{(t, k): p for t, k, p, _ in EXTRA_ENTITIES},
}
ENTITY_STATUS = {(t, k): s for t, k, _, s in EXTRA_ENTITIES}


# --- Ontology metadata (design §4.1, B2) --------------------------------
# Domain/range taken from the actual demo data, not from the design
# doc's prose:
# - PART_OF / HAS_COMPONENT / DOCUMENTED_BY: direction and endpoints
#   straight from RELATIONS in seeds/demo_data.py
#   (Asset -PART_OF-> Building, Asset -HAS_COMPONENT-> Component,
#   Asset -DOCUMENTED_BY-> Document — the same direction ingest's
#   documented_by_relations writes: (Asset, asset_id) -> (Document, doc id)).
# - LOCATED_IN: no demo relation writes it (neither seed nor ingest);
#   Asset -> Building is the reading retrieval assumes ("{s} is
#   located in {o}", runtime/retrieval_context.py). No inverse.
# - RELATED_TO: unconstrained both ends (domain/range "any").
# Data properties: every key the demo entities actually carry,
# datatype inferred from the sample values (floors is the only int;
# installed / last_pad_change are date *strings*). The user "status"
# key on motor-M0 is NOT a data property — it is a system key, exempt
# from validation (ontology.metadata.validate.SYSTEM_PROPERTY_KEYS).
METADATA_CLASSES = [
    ("Asset", "Asset", "Physical asset, e.g. a passenger elevator"),
    ("Building", "Building", "A building containing assets"),
    ("Component", "Component", "A component of an asset"),
    ("Document", "Document", "A technical document"),
]

METADATA_OBJECT_PROPERTIES = [
    # (id, domain, range, inverse_of)
    ("PART_OF", "Asset", "Building", "HAS_COMPONENT"),
    ("HAS_COMPONENT", "Asset", "Component", "PART_OF"),
    ("LOCATED_IN", "Asset", "Building", None),
    ("RELATED_TO", "any", "any", None),
    ("DOCUMENTED_BY", "Asset", "Document", None),
]

METADATA_DATA_PROPERTIES = [
    # (id, domain, datatype, required)
    ("name", "Asset", "str", False),
    ("building_id", "Asset", "str", False),
    ("asset_class", "Asset", "str", False),
    ("name", "Building", "str", False),
    ("address", "Building", "str", False),
    ("floors", "Building", "int", False),
    ("asset_class", "Building", "str", False),
    ("name", "Component", "str", False),
    ("component_class", "Component", "str", False),
    ("manufacturer", "Component", "str", False),
    ("installed", "Component", "str", False),
    ("location", "Component", "str", False),
    ("firmware", "Component", "str", False),
    ("last_pad_change", "Component", "str", False),
    ("title", "Document", "str", False),
    ("category", "Document", "str", False),
]


def seed_ontology_metadata(store) -> dict:
    """Seed the metadata nodes (idempotent MERGE — re-running changes
    nothing). Returns counts per node kind."""
    for class_id, name, description in METADATA_CLASSES:
        create_class(store, class_id=class_id, name=name, description=description)
    for property_id, domain, range_, inverse_of in METADATA_OBJECT_PROPERTIES:
        create_object_property(
            store,
            property_id=property_id,
            domain=domain,
            range=range_,
            inverse_of=inverse_of,
        )
    for property_id, domain, datatype, required in METADATA_DATA_PROPERTIES:
        create_data_property(
            store,
            property_id=property_id,
            domain=domain,
            datatype=datatype,
            required=required,
        )
    return {
        "classes": len(METADATA_CLASSES),
        "object_properties": len(METADATA_OBJECT_PROPERTIES),
        "data_properties": len(METADATA_DATA_PROPERTIES),
    }


def seed_ontology_demo(*, trace_id: str = "seed-ontology-demo") -> dict:
    db = SessionLocal()
    try:
        repo = CanonicalOntologyRepository(db)
        writer = OntologyWriteService(repo)
        prov = WriteProvenance(trace_id=trace_id)

        # Metadata first (idempotent): on a real store the instance
        # writes below then validate against it; on a fake store the
        # metadata calls no-op and validation falls back to v1.
        metadata = seed_ontology_metadata(get_graph_store())

        written: set[tuple[str, str]] = set()

        def write_with_props(entity_type: str, entity_key: str) -> None:
            writer.write_entity(
                entity_type=entity_type,
                entity_key=entity_key,
                properties=ENTITY_PROPERTIES.get((entity_type, entity_key)),
                status=ENTITY_STATUS.get((entity_type, entity_key), "active"),
                provenance=prov,
            )
            written.add((entity_type, entity_key))

        write_with_props("Asset", ASSET_ID)
        for entity_type, entity_key, _, _ in EXTRA_ENTITIES:
            write_with_props(entity_type, entity_key)

        relations_written = 0
        for from_id, from_label, rel_type, to_id, to_label in RELATIONS:
            write_with_props(from_label, from_id)
            write_with_props(to_label, to_id)
            writer.write_relationship(
                relationship_type=rel_type,
                source_type=from_label,
                source_key=from_id,
                target_type=to_label,
                target_key=to_id,
                properties=None,
                provenance=prov,
            )
            relations_written += 1

        db.commit()
        projection = drain_pending_projections(db)
        db.commit()
        logger.info(
            "Ontology seed: entities=%s rels=%s projection=%s metadata=%s",
            len(written),
            relations_written,
            projection,
            metadata,
        )
        return {
            "entities": len(written),
            "relations": relations_written,
            "projection": projection,
            "metadata": metadata,
        }
    finally:
        db.close()


def seed_falkor_demo() -> dict:
    """Backward-compatible name used by seed_demo_data."""
    result = seed_ontology_demo()
    return {"relations": result["relations"]}
