"""Document Management — upload, read, list, grant (docs/documents.md).

Auth: ``require_api_key_or_user`` (JWT user or service API key).
Access follows ``document_service.check_access`` and is leak-free:
a document the caller may not read is 404, exactly as if it did
not exist — never 403.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app import schemas
from app.core.db import get_db
from app.core.security import get_current_user, require_api_key_or_user
from app.services import document_service

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(require_api_key_or_user)],
)


def _to_read(row) -> schemas.DocumentRead:
    return schemas.DocumentRead(
        doc_id=row.doc_id,
        version=row.version,
        title=row.title,
        filename=row.filename,
        content_type=row.content_type,
        content_hash=row.content_hash,
        size_bytes=row.size_bytes,
        owner_user_id=row.owner_user_id,
        asset_id=row.asset_id,
        created_at=row.created_at,
    )


@router.post("", response_model=schemas.DocumentRead, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    asset_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    try:
        row = document_service.upload_document(
            db,
            owner_user_id=user.get("sub") if user is not None else None,
            title=title or file.filename,
            filename=file.filename,
            content=await file.read(),
            content_type=file.content_type or "application/octet-stream",
            asset_id=asset_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_read(row)


@router.get("", response_model=list[schemas.DocumentRead])
def list_documents(
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    return [_to_read(row) for row in document_service.list_documents(db, user)]


@router.get("/{doc_id}", response_model=schemas.DocumentRead)
def get_document(
    doc_id: str,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    row = document_service.get_document(db, doc_id, user)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _to_read(row)


@router.get("/{doc_id}/download")
async def download_document(
    doc_id: str,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    row, data = document_service.download(db, doc_id, user)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(
        content=data,
        media_type=row.content_type,
        headers={"Content-Disposition": f'attachment; filename="{row.filename}"'},
    )


@router.post("/{doc_id}/versions", response_model=schemas.DocumentRead, status_code=201)
async def upload_new_version(
    doc_id: str,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    asset_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    try:
        row = document_service.upload_new_version(
            db,
            doc_id=doc_id,
            user=user,
            title=title,
            filename=file.filename,
            content=await file.read(),
            content_type=file.content_type or "application/octet-stream",
            asset_id=asset_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _to_read(row)


@router.post("/{doc_id}/grants", status_code=201)
def grant_access(
    doc_id: str,
    body: schemas.DocumentGrantCreate,
    db: Session = Depends(get_db),
    user: dict | None = Depends(get_current_user),
):
    ok = document_service.grant_access(
        db,
        doc_id=doc_id,
        user=user,
        principal_user_id=body.principal_user_id,
        permission=body.permission,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "doc_id": doc_id,
        "principal_user_id": body.principal_user_id,
        "permission": body.permission,
    }
