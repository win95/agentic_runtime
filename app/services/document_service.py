"""Document Management service — MinIO objects + Postgres metadata + ACL.

Storage: one MinIO object per version (immutable — a new upload is a
new object, never an overwrite), one ``documents`` row per version,
ACL grants in ``document_acls``. The latest version of a document is
its highest ``version``.

Access is fail-closed and leak-free, following the platform's
ownership pattern (``require_own_execution`` returns 404, never
403):

- A JWT user reads/writes a document when they own it
  (``owner_user_id`` == JWT ``sub``) or hold an explicit grant
  (a ``write`` grant implies ``read``).
- A service-key call (no JWT user) reads/writes only service-owned
  documents (``owner_user_id`` IS NULL). There is no wildcard
  principal and no service bypass for user-owned documents.
- Missing and denied are indistinguishable: every getter returns
  ``None`` for both, so the API never confirms that another
  principal's document exists.

``store`` is the MinIO access object (default: ``infra.minio_client``);
tests pass a fake. MinIO is written BEFORE the DB row, so a MinIO
failure leaves no row pointing at a missing object; a DB failure
after the put leaves an orphan object that is unreachable without
its row (no cleanup in the base).

NOT in the base: no ingest wiring (inbox JSON documents still go
straight to Qdrant), no retrieval ACL enforcement (the retrieval
hook does not check document ACLs yet), no presigned URLs, no
grant revocation endpoint (grants are additive only).
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from pathlib import PurePosixPath

from app import models

logger = logging.getLogger(__name__)

# Upload guard: the base refuses absurd uploads instead of filling MinIO.
MAX_CONTENT_BYTES = 50 * 1024 * 1024

READ = "read"
WRITE = "write"
_PERMISSIONS = (READ, WRITE)


def new_doc_id() -> str:
    return f"doc_{secrets.token_hex(6)}"


def object_key(doc_id: str, version: int, filename: str) -> str:
    return f"docs/{doc_id}/v{version}/{filename}"


def _sub(user: dict | None) -> str | None:
    return user.get("sub") if user else None


def _sanitize_filename(filename: str) -> str:
    name = PurePosixPath(filename or "").name
    if not name or name in (".", "..") or len(name) > 256:
        raise ValueError(f"invalid filename: {filename!r}")
    return name


def _latest_row(db, doc_id: str):
    return (
        db.query(models.Document)
        .filter(models.Document.doc_id == doc_id)
        .order_by(models.Document.version.desc())
        .first()
    )


def _has_grant(db, doc_id: str, principal: str, permission: str) -> bool:
    query = (
        db.query(models.DocumentACL)
        .filter(models.DocumentACL.doc_id == doc_id)
        .filter(models.DocumentACL.principal_user_id == principal)
    )
    if permission == READ:
        query = query.filter(models.DocumentACL.permission.in_([READ, WRITE]))
    else:
        query = query.filter(models.DocumentACL.permission == WRITE)
    return query.first() is not None


def check_access(db, doc_id: str, user: dict | None, permission: str = READ) -> bool:
    """Fail-closed access check: False when missing AND when denied."""
    if permission not in _PERMISSIONS:
        return False
    row = _latest_row(db, doc_id)
    if row is None:
        return False
    sub = _sub(user)
    if sub is not None:
        if row.owner_user_id == sub:
            return True
        return _has_grant(db, doc_id, sub, permission)
    # Service-key call: only service-owned documents (owner NULL).
    return row.owner_user_id is None


def upload_document(
    db,
    *,
    owner_user_id: str | None,
    title: str,
    filename: str,
    content: bytes,
    content_type: str = "application/octet-stream",
    asset_id: str | None = None,
    store=None,
):
    """Store a new document (version 1). Returns the version row."""
    if len(content) > MAX_CONTENT_BYTES:
        raise ValueError(f"content exceeds {MAX_CONTENT_BYTES} bytes")
    if store is None:
        from infra import minio_client as store
    name = _sanitize_filename(filename)
    doc_id = new_doc_id()
    key = object_key(doc_id, 1, name)
    store.upload_document_bytes(key, content, content_type)
    row = models.Document(
        doc_id=doc_id,
        version=1,
        title=title or name,
        filename=name,
        object_key=key,
        content_type=content_type,
        content_hash=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        owner_user_id=owner_user_id,
        asset_id=asset_id,
    )
    db.add(row)
    db.commit()
    return row


def upload_new_version(
    db,
    *,
    doc_id: str,
    user: dict | None,
    filename: str,
    content: bytes,
    title: str | None = None,
    content_type: str = "application/octet-stream",
    asset_id: str | None = None,
    store=None,
):
    """Store a new version of an existing document. None when missing/denied."""
    if len(content) > MAX_CONTENT_BYTES:
        raise ValueError(f"content exceeds {MAX_CONTENT_BYTES} bytes")
    if not check_access(db, doc_id, user, WRITE):
        return None
    if store is None:
        from infra import minio_client as store
    latest = _latest_row(db, doc_id)
    name = _sanitize_filename(filename)
    version = latest.version + 1
    key = object_key(doc_id, version, name)
    store.upload_document_bytes(key, content, content_type)
    row = models.Document(
        doc_id=doc_id,
        version=version,
        title=title if title is not None else latest.title,
        filename=name,
        object_key=key,
        content_type=content_type,
        content_hash=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        owner_user_id=latest.owner_user_id,
        asset_id=asset_id if asset_id is not None else latest.asset_id,
    )
    db.add(row)
    db.commit()
    return row


def get_document(db, doc_id: str, user: dict | None):
    """Latest version row, or None when missing or denied."""
    if not check_access(db, doc_id, user, READ):
        return None
    return _latest_row(db, doc_id)


def get_version(db, doc_id: str, version: int, user: dict | None):
    """One version row, or None when missing or denied."""
    if not check_access(db, doc_id, user, READ):
        return None
    return (
        db.query(models.Document)
        .filter(models.Document.doc_id == doc_id)
        .filter(models.Document.version == version)
        .first()
    )


def download(db, doc_id: str, user: dict | None, version: int | None = None, store=None):
    """(row, bytes) for the latest (or pinned) version; (None, None) when missing/denied."""
    row = (
        get_version(db, doc_id, version, user)
        if version is not None
        else get_document(db, doc_id, user)
    )
    if row is None:
        return None, None
    if store is None:
        from infra import minio_client as store
    return row, store.download_document_bytes(row.object_key)


def list_documents(db, user: dict | None):
    """Latest version of every document the caller can read."""
    rows = db.query(models.Document).order_by(models.Document.doc_id).all()
    latest: dict[str, models.Document] = {}
    for row in rows:
        if row.doc_id not in latest or row.version > latest[row.doc_id].version:
            latest[row.doc_id] = row
    return [row for row in latest.values() if check_access(db, row.doc_id, user, READ)]


def grant_access(
    db,
    *,
    doc_id: str,
    user: dict | None,
    principal_user_id: str,
    permission: str = READ,
) -> bool:
    """Grant ``permission`` on a document to a principal.

    Only the owner may grant (service-key calls only for
    service-owned documents). Returns False when missing, denied,
    or the permission is unknown. Grants are additive only — the
    base has no revocation.
    """
    if permission not in _PERMISSIONS or not principal_user_id:
        return False
    row = _latest_row(db, doc_id)
    if row is None:
        return False
    sub = _sub(user)
    if sub is not None:
        if row.owner_user_id != sub:
            return False
    elif row.owner_user_id is not None:
        return False
    db.merge(
        models.DocumentACL(
            doc_id=doc_id,
            principal_user_id=principal_user_id,
            permission=permission,
        )
    )
    db.commit()
    return True
