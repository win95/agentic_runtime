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
from ontology.projection.engine import drain_pending_projections
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


def seed_ontology_demo(*, trace_id: str = "seed-ontology-demo") -> dict:
    db = SessionLocal()
    try:
        repo = CanonicalOntologyRepository(db)
        writer = OntologyWriteService(repo)
        prov = WriteProvenance(trace_id=trace_id)

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
            "Ontology seed: entities=%s rels=%s projection=%s",
            len(written),
            relations_written,
            projection,
        )
        return {
            "entities": len(written),
            "relations": relations_written,
            "projection": projection,
        }
    finally:
        db.close()


def seed_falkor_demo() -> dict:
    """Backward-compatible name used by seed_demo_data."""
    result = seed_ontology_demo()
    return {"relations": result["relations"]}
