"""Ontology REST API — canonical reads + knowledge graph + B2 metadata
and instance CRUD.

Auth is ``require_api_key`` at router level, as before. Restricting
metadata writes to admin/service-key is an open design question
(§12) — NOT decided here: any API-key holder can currently edit
metadata. See the B2 report.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import schemas
from app.config import settings
from app.core.db import get_db
from app.core.security import require_api_key
from ontology.canonical.entity_read import get_entity as read_entity
from ontology.canonical.provenance import WriteProvenance
from ontology.canonical.repository import CanonicalOntologyRepository
from ontology.canonical.schema import ENTITY_TYPES_V1, RELATIONSHIP_TYPES_V1
from ontology.canonical.write_service import OntologyWriteService
from ontology.knowledge import graph_query
from ontology.metadata import store as metadata_store
from ontology.metadata.validate import default_entity_validator
from ontology.projection.engine import drain_pending_projections
from ontology.projection.provider import get_graph_store
from ontology.reasoning import RULES, recompute_all, recompute_for_entity
from ontology.reasoning.rules import INVERSE_RELATIONS
from runtime.security import ASSET_ID_RE

router = APIRouter(
    prefix="/ontology",
    tags=["ontology"],
    dependencies=[Depends(require_api_key)],
)


class _NotFound(ValueError):
    """ValueError subclass the metadata runner maps to 404."""


def _writer(db: Session) -> OntologyWriteService:
    return OntologyWriteService(CanonicalOntologyRepository(db))


def _provenance() -> WriteProvenance:
    return WriteProvenance(trace_id=f"api-{uuid.uuid4().hex[:12]}")


def _write_result(result) -> schemas.OntologyWriteResult:
    return schemas.OntologyWriteResult(
        ref=result.subject_ref,
        version=result.version,
        change_type=result.change_type,
        applied=result.applied,
        changed=result.applied,
        replayed=result.replayed,
    )


def _commit_write(db: Session, fn):
    """Run a writer call and commit L1; ValueError -> 400, any other
    failure (including the store being down) -> 503."""
    try:
        result = fn()
        db.commit()
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        raise HTTPException(status_code=503, detail=f"store unavailable: {exc}") from exc
    return result


def _run_metadata(fn):
    """Run a metadata-store call; ValueError -> 400, _NotFound ->
    404, any other failure (store down) -> 503."""
    try:
        return fn()
    except _NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"store unavailable: {exc}") from exc


def _metadata_class_ids() -> set[str]:
    """Class ids from Neo4j metadata. Empty when the store is down
    or metadata was never seeded — callers fall back to v1."""
    try:
        return {
            row["id"]
            for row in metadata_store.get_metadata(get_graph_store())["classes"]
        }
    except Exception:  # noqa: BLE001
        return set()


def _metadata_object_property_ids() -> set[str]:
    try:
        return {
            row["id"]
            for row in metadata_store.get_metadata(get_graph_store())[
                "object_properties"
            ]
        }
    except Exception:  # noqa: BLE001
        return set()


def _require_class(store, class_id: str) -> dict:
    row = metadata_store.get_class(store, class_id)
    if row is None:
        raise _NotFound(f"class {class_id!r} not found")
    return row


def _require_object_property(store, property_id: str) -> dict:
    row = metadata_store.get_object_property(store, property_id)
    if row is None:
        raise _NotFound(f"object property {property_id!r} not found")
    return row


def _require_data_property(store, property_id: str) -> dict:
    row = metadata_store.get_data_property(store, property_id)
    if row is None:
        raise _NotFound(f"data property {property_id!r} not found")
    return row


def _validate_entity_type(entity_type: str) -> str:
    """Accept a v1 class or a class defined in Neo4j metadata."""
    value = (entity_type or "").strip()
    if value in ENTITY_TYPES_V1 or value in _metadata_class_ids():
        return value
    raise HTTPException(status_code=400, detail=f"entity_type {value!r} not allowed")


def _validate_relationship_type(relationship_type: str) -> str:
    """Accept a v1 relation or an object property from metadata."""
    value = (relationship_type or "").strip().upper()
    if value in RELATIONSHIP_TYPES_V1 or value in _metadata_object_property_ids():
        return value
    raise HTTPException(
        status_code=400, detail=f"relationship_type {value!r} not allowed"
    )


def _validate_entity_key(entity_type: str, entity_key: str) -> str:
    key = (entity_key or "").strip()
    if not key or len(key) > 200:
        raise HTTPException(status_code=400, detail="invalid entity_key")
    if entity_type == "Asset" and not ASSET_ID_RE.match(key):
        raise HTTPException(status_code=400, detail="invalid Asset entity_key")
    return key


def _run_graph(fn):
    """Run a graph-query call; ValueError -> 400, any other failure
    (store down) -> 503. Same shape as ``_run_metadata``."""
    try:
        return fn()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"graph unavailable: {exc}") from exc


def _to_entity_read(row: dict) -> schemas.OntologyEntityRead:
    """Shape a graph_query entity row (properties may still carry
    the store's own ``status`` / ``version`` keys) into the API's
    entity shape."""
    props = dict(row.get("properties") or {})
    status = str(props.pop("status", None) or "active")
    try:
        version = int(props.pop("version", None) or 1)
    except (TypeError, ValueError):
        version = 1
    return schemas.OntologyEntityRead(
        entity_type=row.get("entity_type") or "",
        entity_key=row.get("entity_key") or "",
        properties=props,
        status=status,
        version=version,
    )


def _schema_label_and_relation_ids(store) -> tuple[list[str], list[str]]:
    """Class ids + object-property ids from Neo4j metadata, with the
    v1 fallback (same rule as B2 validation: no class node at all
    means metadata was never seeded)."""
    try:
        meta = metadata_store.get_metadata(store)
        classes = [row["id"] for row in meta["classes"]]
        if classes:
            return classes, [row["id"] for row in meta["object_properties"]]
    except Exception:  # noqa: BLE001
        pass
    return sorted(ENTITY_TYPES_V1), sorted(RELATIONSHIP_TYPES_V1)


def _default_constraints() -> list[dict]:
    """The UNIQUE (label, id) constraints ``ensure_constraints``
    creates — the fallback when SHOW CONSTRAINTS fails or returns
    nothing (e.g. a fake store)."""
    return [
        {
            "name": f"unique_{label}_id",
            "labels": [label],
            "properties": ["id"],
        }
        for label in sorted(ENTITY_TYPES_V1)
    ]


def _parse_constraints(rows: list[dict]) -> list[dict]:
    """Shape ``SHOW CONSTRAINTS`` rows (Neo4j 5 columns: ``name``,
    ``labelsOrTypes``, ``properties``) into the schema shape."""
    parsed: list[dict] = []
    for row in rows:
        labels = row.get("labelsOrTypes") or row.get("labels") or []
        properties = row.get("properties") or []
        parsed.append(
            {
                "name": row.get("name"),
                "labels": [str(item) for item in labels],
                "properties": [str(item) for item in properties],
            }
        )
    return parsed


def _count(rows: list[dict]) -> int:
    if not rows:
        return 0
    try:
        return int(rows[0].get("n") or 0)
    except (TypeError, ValueError):
        return 0


def _run_cypher_query(store, cypher: str, params: dict, timeout: float):
    """Run a user Cypher query with a transaction timeout.

    ``Neo4jGraphStore.run_query`` accepts ``timeout``; plain fake
    stores (whose ``run_query`` takes only ``(cypher, params)``)
    raise TypeError on the keyword, so fall back to the two-arg call
    — the timeout is a guard, never a reason to fail the query.
    """
    try:
        return store.run_query(cypher, params, timeout=timeout)
    except TypeError:
        return store.run_query(cypher, params)


def _validate_entity_key_loose(entity_key: str) -> str:
    """Key check for query endpoints that take no entity type (e.g.
    ``/reasoning/derived``): non-empty, max 200 chars. The Asset
    regex does not apply without a type."""
    key = (entity_key or "").strip()
    if not key or len(key) > 200:
        raise HTTPException(status_code=400, detail="invalid entity_key")
    return key


@router.get(
    "/entities/{entity_type}/{entity_key}",
    response_model=schemas.OntologyEntityRead,
)
def get_entity(
    entity_type: str,
    entity_key: str,
):
    entity_type = _validate_entity_type(entity_type)
    entity_key = _validate_entity_key(entity_type, entity_key)
    try:
        # Design §6: objects are read from Neo4j, not from L1.
        row = read_entity(get_graph_store(), entity_type, entity_key)
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


# ----------------------------------------------------------------------
# Metadata — classes (design §7, Ontology Management)
# ----------------------------------------------------------------------


@router.get("/classes", response_model=list[schemas.OntologyClassRead])
def list_classes():
    return _run_metadata(lambda: metadata_store.list_classes(get_graph_store()))


@router.post(
    "/classes", response_model=schemas.OntologyClassRead, status_code=201
)
def create_class(body: schemas.OntologyClassCreate):
    return _run_metadata(
        lambda: metadata_store.create_class(
            get_graph_store(),
            class_id=body.id,
            name=body.name,
            description=body.description,
        )
    )


@router.get("/classes/{class_id}", response_model=schemas.OntologyClassRead)
def get_class(class_id: str):
    def _get():
        return _require_class(get_graph_store(), class_id)

    return _run_metadata(_get)


@router.patch("/classes/{class_id}", response_model=schemas.OntologyClassRead)
def update_class(class_id: str, body: schemas.OntologyClassUpdate):
    def _update():
        store = get_graph_store()
        _require_class(store, class_id)
        return metadata_store.update_class(
            store, class_id=class_id, name=body.name, description=body.description
        )

    return _run_metadata(_update)


@router.delete("/classes/{class_id}", response_model=schemas.OntologyClassRead)
def delete_class(class_id: str):
    def _delete():
        store = get_graph_store()
        _require_class(store, class_id)
        return metadata_store.delete_class(store, class_id)

    return _run_metadata(_delete)


@router.post(
    "/classes/{class_id}/subclasses",
    response_model=schemas.OntologySubclassRead,
    status_code=201,
)
def add_subclass(class_id: str, body: schemas.OntologySubclassCreate):
    def _add():
        store = get_graph_store()
        _require_class(store, class_id)
        _require_class(store, body.parent_id)
        return metadata_store.add_subclass(
            store, child_id=class_id, parent_id=body.parent_id
        )

    return _run_metadata(_add)


@router.delete(
    "/classes/{class_id}/subclasses/{parent_id}",
    response_model=schemas.OntologySubclassRead,
)
def remove_subclass(class_id: str, parent_id: str):
    def _remove():
        store = get_graph_store()
        _require_class(store, class_id)
        _require_class(store, parent_id)
        return metadata_store.remove_subclass(
            store, child_id=class_id, parent_id=parent_id
        )

    return _run_metadata(_remove)


# ----------------------------------------------------------------------
# Metadata — object properties
# ----------------------------------------------------------------------


@router.get(
    "/object-properties", response_model=list[schemas.OntologyObjectPropertyRead]
)
def list_object_properties():
    return _run_metadata(
        lambda: metadata_store.list_object_properties(get_graph_store())
    )


@router.post(
    "/object-properties",
    response_model=schemas.OntologyObjectPropertyRead,
    status_code=201,
)
def create_object_property(body: schemas.OntologyObjectPropertyCreate):
    return _run_metadata(
        lambda: metadata_store.create_object_property(
            get_graph_store(),
            property_id=body.id,
            domain=body.domain,
            range=body.range,
            inverse_of=body.inverse_of,
        )
    )


@router.get(
    "/object-properties/{property_id}",
    response_model=schemas.OntologyObjectPropertyRead,
)
def get_object_property(property_id: str):
    def _get():
        return _require_object_property(get_graph_store(), property_id)

    return _run_metadata(_get)


@router.patch(
    "/object-properties/{property_id}",
    response_model=schemas.OntologyObjectPropertyRead,
)
def update_object_property(
    property_id: str, body: schemas.OntologyObjectPropertyUpdate
):
    def _update():
        store = get_graph_store()
        _require_object_property(store, property_id)
        return metadata_store.update_object_property(
            store,
            property_id=property_id,
            domain=body.domain,
            range=body.range,
            inverse_of=body.inverse_of,
        )

    return _run_metadata(_update)


@router.delete(
    "/object-properties/{property_id}",
    response_model=schemas.OntologyObjectPropertyRead,
)
def delete_object_property(property_id: str):
    def _delete():
        store = get_graph_store()
        _require_object_property(store, property_id)
        return metadata_store.delete_object_property(store, property_id)

    return _run_metadata(_delete)


# ----------------------------------------------------------------------
# Metadata — data properties
# ----------------------------------------------------------------------


@router.get(
    "/data-properties", response_model=list[schemas.OntologyDataPropertyRead]
)
def list_data_properties():
    return _run_metadata(
        lambda: metadata_store.list_data_properties(get_graph_store())
    )


@router.post(
    "/data-properties",
    response_model=schemas.OntologyDataPropertyRead,
    status_code=201,
)
def create_data_property(body: schemas.OntologyDataPropertyCreate):
    return _run_metadata(
        lambda: metadata_store.create_data_property(
            get_graph_store(),
            property_id=body.id,
            domain=body.domain,
            datatype=body.datatype,
            required=body.required,
        )
    )


@router.get(
    "/data-properties/{property_id}",
    response_model=schemas.OntologyDataPropertyRead,
)
def get_data_property(property_id: str):
    def _get():
        return _require_data_property(get_graph_store(), property_id)

    return _run_metadata(_get)


@router.patch(
    "/data-properties/{property_id}",
    response_model=schemas.OntologyDataPropertyRead,
)
def update_data_property(
    property_id: str, body: schemas.OntologyDataPropertyUpdate
):
    def _update():
        store = get_graph_store()
        _require_data_property(store, property_id)
        return metadata_store.update_data_property(
            store,
            property_id=property_id,
            domain=body.domain,
            datatype=body.datatype,
            required=body.required,
        )

    return _run_metadata(_update)


@router.delete(
    "/data-properties/{property_id}",
    response_model=schemas.OntologyDataPropertyRead,
)
def delete_data_property(property_id: str):
    def _delete():
        store = get_graph_store()
        _require_data_property(store, property_id)
        return metadata_store.delete_data_property(store, property_id)

    return _run_metadata(_delete)


# ----------------------------------------------------------------------
# Instance CRUD (design §7, Instance Management)
# ----------------------------------------------------------------------


@router.post("/entities", response_model=schemas.OntologyWriteResult)
def create_entity(
    body: schemas.OntologyEntityCreate, db: Session = Depends(get_db)
):
    entity_type = _validate_entity_type(body.entity_type)
    entity_key = _validate_entity_key(entity_type, body.entity_key)
    return _write_result(
        _commit_write(
            db,
            lambda: _writer(db).write_entity(
                entity_type=entity_type,
                entity_key=entity_key,
                properties=body.properties,
                provenance=_provenance(),
            ),
        )
    )


@router.patch(
    "/entities/{entity_type}/{entity_key}",
    response_model=schemas.OntologyWriteResult,
)
def update_entity(
    entity_type: str,
    entity_key: str,
    body: schemas.OntologyEntityUpdate,
    db: Session = Depends(get_db),
):
    """PATCH replaces the entity's properties wholesale: the body is
    passed to ``write_entity`` as an explicit dict, so keys absent
    from the body are removed from the node (an empty dict clears
    all user properties). 404 when the entity does not exist."""
    entity_type = _validate_entity_type(entity_type)
    entity_key = _validate_entity_key(entity_type, entity_key)
    try:
        current = read_entity(get_graph_store(), entity_type, entity_key)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail=f"store unavailable: {exc}"
        ) from exc
    if current is None:
        raise HTTPException(status_code=404, detail="entity not found")
    return _write_result(
        _commit_write(
            db,
            lambda: _writer(db).write_entity(
                entity_type=entity_type,
                entity_key=entity_key,
                properties=body.properties,
                provenance=_provenance(),
            ),
        )
    )


@router.delete(
    "/entities/{entity_type}/{entity_key}",
    response_model=schemas.OntologyWriteResult,
)
def delete_entity(
    entity_type: str, entity_key: str, db: Session = Depends(get_db)
):
    """Soft-delete: L1 status -> deleted, Neo4j node DETACH DELETEd
    (the write service's delete path). 404 when absent."""
    entity_type = _validate_entity_type(entity_type)
    entity_key = _validate_entity_key(entity_type, entity_key)
    try:
        current = read_entity(get_graph_store(), entity_type, entity_key)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail=f"store unavailable: {exc}"
        ) from exc
    if current is None:
        raise HTTPException(status_code=404, detail="entity not found")
    return _write_result(
        _commit_write(
            db,
            lambda: _writer(db).write_entity(
                entity_type=entity_type,
                entity_key=entity_key,
                properties=None,
                provenance=_provenance(),
                status="deleted",
            ),
        )
    )


@router.post("/relationships", response_model=schemas.OntologyWriteResult)
def create_relationship(
    body: schemas.OntologyRelationshipCreate, db: Session = Depends(get_db)
):
    relationship_type = _validate_relationship_type(body.relationship_type)
    source_type = _validate_entity_type(body.source_type)
    target_type = _validate_entity_type(body.target_type)
    source_key = _validate_entity_key(source_type, body.source_key)
    target_key = _validate_entity_key(target_type, body.target_key)
    return _write_result(
        _commit_write(
            db,
            lambda: _writer(db).write_relationship(
                relationship_type=relationship_type,
                source_type=source_type,
                source_key=source_key,
                target_type=target_type,
                target_key=target_key,
                properties=body.properties,
                provenance=_provenance(),
            ),
        )
    )


@router.delete("/relationships", response_model=schemas.OntologyWriteResult)
def delete_relationship(
    rel_type: str = Query(..., alias="type"),
    source_type: str = Query(...),
    source_key: str = Query(...),
    target_type: str = Query(...),
    target_key: str = Query(...),
    db: Session = Depends(get_db),
):
    """Soft-delete a relationship (query params identify the edge).
    404 when the edge does not exist on Neo4j."""
    relationship_type = _validate_relationship_type(rel_type)
    source_type = _validate_entity_type(source_type)
    target_type = _validate_entity_type(target_type)
    source_key = _validate_entity_key(source_type, source_key)
    target_key = _validate_entity_key(target_type, target_key)
    try:
        edge = get_graph_store().get_relationship(
            source_type, source_key, relationship_type, target_type, target_key
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail=f"store unavailable: {exc}"
        ) from exc
    if edge is None:
        raise HTTPException(status_code=404, detail="relationship not found")
    return _write_result(
        _commit_write(
            db,
            lambda: _writer(db).write_relationship(
                relationship_type=relationship_type,
                source_type=source_type,
                source_key=source_key,
                target_type=target_type,
                target_key=target_key,
                properties=None,
                provenance=_provenance(),
                status="deleted",
            ),
        )
    )


@router.post(
    "/entities/{entity_type}/{entity_key}/validate",
    response_model=schemas.OntologyValidateResult,
)
def validate_entity_endpoint(
    entity_type: str,
    entity_key: str,
    body: schemas.OntologyEntityValidate,
):
    """Dry-run validation: runs the metadata validator for the
    would-be write and reports the outcome. Writes nothing."""
    entity_type = _validate_entity_type(entity_type)
    entity_key = _validate_entity_key(entity_type, entity_key)
    try:
        default_entity_validator(
            get_graph_store(), entity_type=entity_type, properties=body.properties
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail=f"store unavailable: {exc}"
        ) from exc
    return schemas.OntologyValidateResult(
        valid=True, entity_type=entity_type, entity_key=entity_key
    )


# ----------------------------------------------------------------------
# B5 — Query API (design §7, Query and Retrieval) + Reasoning
# ----------------------------------------------------------------------


@router.post("/query", response_model=schemas.CypherQueryResponse)
def run_cypher_query(body: schemas.CypherQueryRequest):
    """Run an arbitrary READ-ONLY Cypher query against Neo4j.

    The read-only guard (``graph_query._assert_read_only_cypher``)
    rejects any write keyword before the query reaches the store:
    400 on violation. ``limit`` is clamped to 1..500 and applied to
    the rows returned (it is NOT appended to the Cypher — appending
    would break queries that already end with LIMIT or ``;``). The
    transaction timeout (default 30 s, clamp 1..120 s) bounds
    runaway queries; a store failure is 503. ``columns`` are the
    keys of the first row, or [] when there are no rows.
    """
    try:
        graph_query._assert_read_only_cypher(body.cypher)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    limit = max(1, min(body.limit, 500))
    timeout = body.timeout_seconds if body.timeout_seconds is not None else 30.0
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid timeout_seconds") from None
    timeout = max(1.0, min(timeout, 120.0))
    try:
        rows = _run_cypher_query(
            get_graph_store(), body.cypher, body.params, timeout
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"graph unavailable: {exc}") from exc
    rows = list(rows or [])[:limit]
    columns = list(rows[0].keys()) if rows else []
    return schemas.CypherQueryResponse(columns=columns, rows=rows)


@router.get("/search", response_model=list[schemas.OntologyEntityRead])
def search_entities_endpoint(
    q: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=500),
):
    """Find entities by id or ``name`` (case-insensitive CONTAINS).

    No full-text index is used — see ``graph_query.search_entities``.
    """
    rows = _run_graph(
        lambda: graph_query.search_entities(get_graph_store(), q, limit=limit)
    )
    return [_to_entity_read(row) for row in rows]


@router.get("/entities", response_model=list[schemas.OntologyEntityRead])
def list_entities(
    class_: str = Query(..., alias="class"),
    include_subclasses: bool = Query(True),
    limit: int = Query(100, ge=1, le=500),
):
    """List instances of a class — including its subclasses (R1) by
    default. No route conflict with ``GET /entities/{type}/{key}``:
    this path has no extra segments, so FastAPI never matches one
    for the other."""
    class_id = _validate_entity_type(class_)
    rows = _run_graph(
        lambda: graph_query.entities_by_class(
            get_graph_store(),
            class_id,
            include_subclasses=include_subclasses,
            limit=limit,
        )
    )
    return [_to_entity_read(row) for row in rows]


@router.get("/traverse", response_model=schemas.OntologySubgraphRead)
def traverse_graph(
    root_type: str = Query(...),
    root_key: str = Query(...),
    depth: int = Query(2, ge=1, le=3),
    limit: int = Query(100, ge=1, le=500),
    rel_types: list[str] | None = Query(default=None),
):
    """Multi-hop traversal from a root node, optionally filtered by
    relationship type. ``rel_types`` may repeat (``?rel_types=A&rel_types=B``)
    or be comma-separated (``?rel_types=A,B``)."""
    root_type = _validate_entity_type(root_type)
    root_key = _validate_entity_key(root_type, root_key)
    parsed: list[str] | None = None
    if rel_types:
        parsed = []
        for item in rel_types:
            for part in item.split(","):
                part = part.strip()
                if part:
                    parsed.append(_validate_relationship_type(part))
    payload = _run_graph(
        lambda: graph_query.traverse(
            get_graph_store(),
            root_type,
            root_key,
            depth=depth,
            limit=limit,
            rel_types=parsed,
        )
    )
    return schemas.OntologySubgraphRead(**payload)


@router.get("/schema", response_model=schemas.OntologySchemaRead)
def get_schema():
    """Ontology schema: classes + object/data properties from Neo4j
    metadata, uniqueness constraints from ``SHOW CONSTRAINTS``.

    When no metadata was seeded yet, classes/relations fall back to
    the v1 allowlists (same rule as B2 validation); when the
    constraints query fails or returns nothing, the fallback is the
    UNIQUE (label, id) constraint per v1 label that
    ``ensure_constraints`` creates.
    """
    try:
        store = get_graph_store()
        meta = metadata_store.get_metadata(store)
        classes = meta["classes"]
        object_properties = meta["object_properties"]
        data_properties = meta["data_properties"]
        if not classes:
            classes = [
                {"id": label, "name": label, "description": None}
                for label in sorted(ENTITY_TYPES_V1)
            ]
            object_properties = [
                {
                    "id": rel,
                    "domain": None,
                    "range": None,
                    "inverse_of": INVERSE_RELATIONS.get(rel),
                }
                for rel in sorted(RELATIONSHIP_TYPES_V1)
            ]
            data_properties = []
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail=f"store unavailable: {exc}"
        ) from exc
    try:
        constraints = _parse_constraints(store.run_query("SHOW CONSTRAINTS"))
    except Exception:  # noqa: BLE001
        constraints = []
    if not constraints:
        constraints = _default_constraints()
    return schemas.OntologySchemaRead(
        classes=[
            schemas.OntologyClassRead(
                id=row.get("id") or "",
                name=row.get("name") or row.get("id") or "",
                description=row.get("description"),
            )
            for row in classes
        ],
        object_properties=[
            schemas.OntologyObjectPropertyRead(
                id=row.get("id") or "",
                domain=row.get("domain"),
                range=row.get("range"),
                inverse_of=row.get("inverse_of"),
            )
            for row in object_properties
        ],
        data_properties=[
            schemas.OntologyDataPropertyRead(
                id=row.get("id") or "",
                domain=row.get("domain") or "",
                datatype=row.get("datatype") or "str",
                required=bool(row.get("required")),
            )
            for row in data_properties
        ],
        constraints=[
            schemas.OntologySchemaConstraintRead(**row) for row in constraints
        ],
    )


@router.get("/graph/stats", response_model=schemas.OntologyGraphStatsRead)
def get_graph_stats():
    """Node counts per label, edge counts per type, derived-edge
    count. Labels/types come from metadata with the v1 fallback."""
    try:
        store = get_graph_store()
        labels, rel_types = _schema_label_and_relation_ids(store)
        nodes_by_label: dict[str, int] = {}
        for label in labels:
            rows = store.run_query(f"MATCH (n:{label}) RETURN count(n) AS n")
            nodes_by_label[label] = _count(rows)
        edges_by_type: dict[str, int] = {}
        for rel_type in rel_types:
            rows = store.run_query(
                f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS n"
            )
            edges_by_type[rel_type] = _count(rows)
        derived_rows = store.run_query(
            "MATCH ()-[r {derived: true}]->() RETURN count(r) AS n"
        )
        derived = _count(derived_rows)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail=f"graph unavailable: {exc}"
        ) from exc
    return schemas.OntologyGraphStatsRead(
        nodes_by_label=nodes_by_label,
        edges_by_type=edges_by_type,
        derived_edges=derived,
        total_nodes=sum(nodes_by_label.values()),
        total_edges=sum(edges_by_type.values()),
    )


@router.get("/reasoning/rules", response_model=list[schemas.ReasoningRuleRead])
def list_reasoning_rules():
    """The v1 reasoning rule catalog (R1 subclass, R2 inverse,
    R3 transitive)."""
    return [schemas.ReasoningRuleRead(**rule) for rule in RULES]


@router.get("/reasoning/derived", response_model=list[schemas.DerivedEdgeRead])
def list_derived_edges_endpoint(
    entity_type: str | None = Query(default=None),
    entity_key: str | None = Query(default=None),
    limit: int = Query(100, ge=1, le=500),
):
    """Derived edges (``derived=true``). Whole graph when no entity
    is given; when ``entity_key`` is given, only edges touching that
    entity (``entity_type``, when also given, must be its label)."""
    if entity_type is not None:
        entity_type = _validate_entity_type(entity_type)
    if entity_key is not None:
        entity_key = _validate_entity_key_loose(entity_key)
    rows = _run_graph(
        lambda: graph_query.list_derived_edges(
            get_graph_store(),
            entity_type=entity_type,
            entity_key=entity_key,
            limit=limit,
        )
    )
    return [schemas.DerivedEdgeRead(**row) for row in rows]


@router.post(
    "/reasoning/recompute", response_model=schemas.ReasoningRecomputeResponse
)
def recompute_reasoning(body: schemas.ReasoningRecomputeRequest):
    """Re-run derived-edge materialization (design §4.3): for one
    entity (``entity_type`` + ``entity_key``) or the whole graph
    (``all=true``). Use after adding a rule or editing ontology
    metadata. Exactly one scope must be given, else 400."""
    if body.all and (body.entity_type or body.entity_key):
        raise HTTPException(
            status_code=400,
            detail="pass either all=true or an entity, not both",
        )
    if body.all:
        result = _run_graph(lambda: recompute_all(get_graph_store()))
        return schemas.ReasoningRecomputeResponse(
            removed=result.get("removed", 0),
            created=result.get("created", []),
            scope="all",
        )
    if body.entity_type and body.entity_key:
        entity_type = _validate_entity_type(body.entity_type)
        entity_key = _validate_entity_key(entity_type, body.entity_key)
        result = _run_graph(
            lambda: recompute_for_entity(
                get_graph_store(), entity_type=entity_type, entity_key=entity_key
            )
        )
        return schemas.ReasoningRecomputeResponse(
            removed=result.get("removed", 0),
            created=result.get("created", []),
            scope=f"{entity_type}:{entity_key}",
        )
    raise HTTPException(
        status_code=400,
        detail="provide entity_type + entity_key, or all=true",
    )
