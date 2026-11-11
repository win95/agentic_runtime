"""Deprecated FalkorDB shim — delegates to Neo4j graph_client.

New code should import ``infra.graph_client`` directly.
"""

from __future__ import annotations

from infra.graph_client import query_asset_relationships


def get_graph():
    raise RuntimeError("FalkorDB removed; use Neo4j via infra.graph_client")


def merge_asset_relations(relations: list[tuple]) -> None:
    """Legacy hook — canonical ingest should use OntologyWriteService."""
    from app.core.db import SessionLocal
    from ontology.canonical.provenance import WriteProvenance
    from ontology.canonical.repository import CanonicalOntologyRepository
    from ontology.canonical.write_service import OntologyWriteService

    db = SessionLocal()
    try:
        writer = OntologyWriteService(CanonicalOntologyRepository(db))
        trace = WriteProvenance(trace_id="legacy-merge-asset-relations")
        for from_id, from_label, rel_type, to_id, to_label in relations:
            writer.write_entity(
                entity_type=from_label,
                entity_key=from_id,
                properties=None,
                provenance=trace,
                force_update=True,
            )
            writer.write_entity(
                entity_type=to_label,
                entity_key=to_id,
                properties=None,
                provenance=trace,
                force_update=True,
            )
            writer.write_relationship(
                relationship_type=rel_type,
                source_type=from_label,
                source_key=from_id,
                target_type=to_label,
                target_key=to_id,
                properties=None,
                provenance=trace,
                force_update=True,
            )
        db.commit()
        from ontology.projection.engine import drain_pending_projections

        drain_pending_projections(db)
    finally:
        db.close()


__all__ = ["get_graph", "merge_asset_relations", "query_asset_relationships"]
