"""Seed canonical ontology + Neo4j projection for demo asset O21."""

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


def seed_ontology_demo(*, trace_id: str = "seed-ontology-demo") -> dict:
    db = SessionLocal()
    try:
        repo = CanonicalOntologyRepository(db)
        writer = OntologyWriteService(repo)
        prov = WriteProvenance(trace_id=trace_id)

        writer.write_entity(
            entity_type="Asset",
            entity_key=ASSET_ID,
            properties=O21_PROPERTIES,
            provenance=prov,
        )

        entities_written = 1
        relations_written = 0
        for from_id, from_label, rel_type, to_id, to_label in RELATIONS:
            writer.ensure_entity(
                entity_type=from_label,
                entity_key=from_id,
                properties=None,
                provenance=prov,
            )
            writer.ensure_entity(
                entity_type=to_label,
                entity_key=to_id,
                properties=None,
                provenance=prov,
            )
            writer.write_relationship(
                relationship_type=rel_type,
                source_type=from_label,
                source_key=from_id,
                target_type=to_label,
                target_key=to_id,
                properties=None,
                provenance=prov,
            )
            entities_written += 2
            relations_written += 1

        db.commit()
        projection = drain_pending_projections(db)
        db.commit()
        logger.info(
            "Ontology seed: entities~%s rels=%s projection=%s",
            entities_written,
            relations_written,
            projection,
        )
        return {
            "entities": entities_written,
            "relations": relations_written,
            "projection": projection,
        }
    finally:
        db.close()


def seed_falkor_demo() -> dict:
    """Backward-compatible name used by seed_demo_data."""
    result = seed_ontology_demo()
    return {"relations": result["relations"]}
