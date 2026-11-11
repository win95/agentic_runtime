"""Ontology REST API — canonical reads + knowledge graph."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import schemas
from app.config import settings
from app.core.db import get_db
from app.core.security import require_api_key
from ontology.canonical.entity_read import get_entity as read_entity
from ontology.canonical.schema import ENTITY_TYPES_V1
from ontology.knowledge import graph_query
from ontology.projection.engine import drain_pending_projections
from runtime.security import ASSET_ID_RE

router = APIRouter(
    prefix="/ontology",
    tags=["ontology"],
    dependencies=[Depends(require_api_key)],
)


def _validate_entity_type(entity_type: str) -> str:
    value = (entity_type or "").strip()
    if value not in ENTITY_TYPES_V1:
        raise HTTPException(status_code=400, detail=f"entity_type {value!r} not allowed")
    return value


def _validate_entity_key(entity_type: str, entity_key: str) -> str:
    key = (entity_key or "").strip()
    if not key or len(key) > 200:
        raise HTTPException(status_code=400, detail="invalid entity_key")
    if entity_type == "Asset" and not ASSET_ID_RE.match(key):
        raise HTTPException(status_code=400, detail="invalid Asset entity_key")
    return key


@router.get(
    "/entities/{entity_type}/{entity_key}",
    response_model=schemas.OntologyEntityRead,
)
def get_entity(
    entity_type: str,
    entity_key: str,
    db: Session = Depends(get_db),
):
    entity_type = _validate_entity_type(entity_type)
    entity_key = _validate_entity_key(entity_type, entity_key)
    try:
        row = read_entity(db, entity_type, entity_key)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"store unavailable: {exc}") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="entity not found")
    return row


@router.get(
    "/entities/{entity_type}/{entity_key}/relationships",
    response_model=list[schemas.EntityRelationshipRead],
)
def get_entity_relationships(
    entity_type: str,
    entity_key: str,
    limit: int = Query(50, ge=1, le=200),
):
    entity_type = _validate_entity_type(entity_type)
    entity_key = _validate_entity_key(entity_type, entity_key)
    try:
        rows = graph_query.neighbors(entity_type, entity_key, depth=1, limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"graph unavailable: {exc}") from exc
    return [
        schemas.EntityRelationshipRead(
            from_id=r.get("from") or entity_key,
            rel_type=r.get("type") or "",
            to_id=r.get("to") or "",
            to_label=r.get("to_label"),
        )
        for r in rows
    ]


@router.get(
    "/entities/{entity_type}/{entity_key}/knowledge",
    response_model=schemas.KnowledgeLinksRead,
)
def get_knowledge_links(
    entity_type: str,
    entity_key: str,
    limit: int = Query(50, ge=1, le=200),
):
    entity_type = _validate_entity_type(entity_type)
    entity_key = _validate_entity_key(entity_type, entity_key)
    try:
        links = graph_query.linked_documents(entity_type, entity_key, limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"graph unavailable: {exc}") from exc
    return schemas.KnowledgeLinksRead(
        entity_type=entity_type,
        entity_key=entity_key,
        documents=links,
    )


@router.get("/graph", response_model=schemas.OntologySubgraphRead)
def get_subgraph(
    root_type: str = Query(...),
    root_key: str = Query(...),
    depth: int = Query(2, ge=1, le=3),
    limit: int = Query(100, ge=1, le=500),
):
    root_type = _validate_entity_type(root_type)
    root_key = _validate_entity_key(root_type, root_key)
    try:
        payload = graph_query.subgraph(root_type, root_key, depth=depth, limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"graph unavailable: {exc}") from exc
    return schemas.OntologySubgraphRead(**payload)


@router.post("/internal/project-drain", response_model=schemas.ProjectionDrainResult)
def trigger_projection_drain(db: Session = Depends(get_db)):
    # Same batch size as the beat task (settings.projection_drain_batch_size).
    result = drain_pending_projections(db, limit=settings.projection_drain_batch_size)
    return schemas.ProjectionDrainResult(**result)
