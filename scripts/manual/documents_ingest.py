"""Manual test: Document Management API + ingest DOCUMENTED_BY (live stack).

Covers the two features added in ``0da4e6a``:

1. **Document Management** (``docs/documents.md``) through the real API:
   multipart upload (v1), metadata, list, download, immutable new version
   (v2/v3), version pinning in MinIO, filename sanitisation, and the
   fail-closed ACL rules — owner vs grant, service-key vs JWT user, and
   the leak-free 404 (never 403, never confirm another principal's doc).
2. **Ingest + DOCUMENTED_BY** (``docs/ingest.md``): run one batch from a
   temp inbox with one document linked to ``O21`` and one without an
   ``asset_id``; assert the rows land in the four stores, the canonical
   ``(Asset, O21) —DOCUMENTED_BY→ (Document, id)`` edge exists, then
   drain the projection queue and assert ``neighbors(O21)`` (Neo4j) and
   ``linked_documents(O21)`` see it. A second forced batch must replay,
   not duplicate.

Requires the live stack: API (``localhost:8000``), Postgres, MinIO,
Qdrant, Neo4j, and the embedding endpoint (ingest embeds content).
``.env`` is loaded through ``scripts/manual/_api``; ``API_KEY`` and
``JWT_SECRET`` must be set to test both auth modes, otherwise the JWT
half is SKIPped (service-key checks still run).

Usage:
    .venv/bin/python scripts/manual/documents_ingest.py
    .venv/bin/python scripts/manual/documents_ingest.py --keep      # leave rows/objects
    .venv/bin/python scripts/manual/documents_ingest.py --documents-only
    .venv/bin/python scripts/manual/documents_ingest.py --ingest-only

Exit code: 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
import requests
from sqlalchemy import inspect

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _api import api_base, api_key  # noqa: E402  (loads .env at import)
from agent_config_check import Report  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.db import SessionLocal  # noqa: E402
from app.models import Document, DocumentACL, IngestQuarantine, IngestRun  # noqa: E402
from app.services.document_service import object_key  # noqa: E402
from infra import minio_client, qdrant_client  # noqa: E402
from ingest.run import run_batch  # noqa: E402
from ingest.sources import LocalDirSource  # noqa: E402
from ontology.knowledge import graph_query  # noqa: E402
from ontology.models import (  # noqa: E402
    OntologyEntity,
    OntologyProjectionState,
    OntologyRelationship,
    OntologyVersion,
)
from ontology.projection.engine import drain_pending_projections  # noqa: E402
from ontology.projection.provider import get_graph_store  # noqa: E402

ASSET_ID = "O21"  # seeded asset: neighbours are safe to read, we never delete it
TIMEOUT = 60
BUCKET = settings.minio_documents_bucket


# --- HTTP helpers ------------------------------------------------------------


def _base() -> str:
    return api_base().rstrip("/")


def _service_headers() -> dict:
    return {"X-API-Key": api_key()}


def _mint(sub: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": sub,
            "email": f"{sub}@manual.test",
            "role": "user",
            "iat": now,
            "exp": now + timedelta(hours=1),
        },
        settings.jwt_secret,
        algorithm="HS256",
    )


def _user_headers(sub: str) -> dict:
    return {"Authorization": f"Bearer {_mint(sub)}"}


def _req(method: str, path: str, *, headers: dict | None = None, **kwargs):
    return requests.request(
        method, f"{_base()}{path}", headers=headers, timeout=TIMEOUT, **kwargs
    )


def _upload(path: str, *, headers: dict, filename: str, content: bytes,
            title: str | None = None, asset_id: str | None = None):
    data: dict = {}
    if title is not None:
        data["title"] = title
    if asset_id is not None:
        data["asset_id"] = asset_id
    return _req(
        "POST",
        path,
        headers=headers,
        files={"file": (filename, content, "text/plain")},
        data=data,
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- Preflight ---------------------------------------------------------------


def check_environment(report: Report, db) -> bool:
    """Stack + stores reachable, migration applied, router mounted."""
    print("\n=== preflight ===")
    try:
        health = _req("GET", "/health")
        report.check(
            health.status_code == 200,
            "API health is 200",
            f"{health.status_code} {health.text[:120]}",
        )
        ready = health.status_code == 200
    except requests.RequestException as exc:
        report.check(False, "API health is 200", f"unreachable: {exc}")
        ready = False

    tables = set(inspect(db.get_bind()).get_table_names())
    report.check(
        {"documents", "document_acls"} <= tables,
        "migration 0012 created documents + document_acls",
        f"{len(tables)} tables",
    )

    try:
        client = minio_client.get_client()
        exists = bool(client and client.bucket_exists(BUCKET))
    except Exception as exc:  # noqa: BLE001
        exists = False
        detail = str(exc)
    else:
        detail = f"bucket={BUCKET}"
    report.check(exists, "MinIO documents bucket exists", detail)

    try:
        q_ok = qdrant_client.collection_exists()
        q_detail = f"collection={settings.qdrant_collection}"
    except Exception as exc:  # noqa: BLE001
        q_ok, q_detail = False, str(exc)
    report.check(q_ok, "Qdrant documents collection exists", q_detail)

    try:
        assets = graph_query.list_entity_keys("Asset", limit=5)
        g_ok = bool(assets)
        g_detail = f"assets={','.join(assets[:3])}"
    except Exception as exc:  # noqa: BLE001
        g_ok, g_detail = False, str(exc)
    report.check(g_ok, "Neo4j graph reachable (Asset keys)", g_detail)

    try:
        paths = _req("GET", "/openapi.json").json().get("paths", {})
        mounted = any(p.startswith("/documents") for p in paths)
    except Exception:  # noqa: BLE001
        mounted = False
    report.check(mounted, "documents router is mounted", "/documents")

    report.check(
        bool(api_key()),
        "service API key is configured (tests both auth modes)",
        "API_KEY set" if api_key() else "API_KEY empty — service-key checks cannot run",
    )
    return ready


def check_auth_config(report: Report) -> bool:
    ok = bool(settings.jwt_secret)
    report.check(
        ok,
        "JWT secret is configured (ACL user checks)",
        "JWT_SECRET set" if ok else "JWT_SECRET empty — skip ACL user checks",
    )
    return ok


# --- Document Management -----------------------------------------------------


def scenario_documents(report: Report, cleanup: dict) -> None:
    print("\n=== Document Management (API) ===")
    tag = secrets.token_hex(4)
    service = _service_headers()
    user_a = f"manual-doc-a-{tag}"
    user_b = f"manual-doc-b-{tag}"
    jwt_ok = bool(settings.jwt_secret)

    content_v1 = f"manual document v1 {tag}\n".encode()
    content_v2 = f"manual document v2 {tag}\n".encode()
    filename = f"manual-{tag}.txt"

    # -- service-key upload / read / list -----------------------------------
    r = _upload("/documents", headers=service, filename=filename,
                content=content_v1, title="Manual doc", asset_id=ASSET_ID)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    doc_id = body.get("doc_id", "")
    report.check(r.status_code == 201 and body.get("version") == 1,
                 "service upload returns 201 / version 1",
                 f"{r.status_code} doc_id={doc_id} v={body.get('version')}")
    if not doc_id:
        report.check(False, "upload returned a doc_id", r.text[:200])
        return
    cleanup["doc_ids"].add(doc_id)
    for version, name, data in ((1, filename, content_v1),):
        cleanup["objects"].add(object_key(doc_id, version, name))

    report.check(
        body.get("content_hash") == _sha(content_v1)
        and body.get("size_bytes") == len(content_v1)
        and body.get("filename") == filename
        and body.get("asset_id") == ASSET_ID,
        "upload metadata: hash, size, filename, asset_id",
        json.dumps({k: body.get(k) for k in
                    ("content_hash", "size_bytes", "filename", "asset_id")}),
    )
    report.check(
        body.get("owner_user_id") is None,
        "service upload is service-owned (owner_user_id NULL)",
        str(body.get("owner_user_id")),
    )

    key_v1 = object_key(doc_id, 1, filename)
    try:
        stored = minio_client.download_document_bytes(key_v1)
    except Exception as exc:  # noqa: BLE001
        stored, store_detail = b"", str(exc)
    else:
        store_detail = key_v1
    report.check(stored == content_v1, "MinIO object v1 holds the bytes", store_detail)

    r = _req("GET", f"/documents/{doc_id}", headers=service)
    report.check(r.status_code == 200 and r.json().get("version") == 1,
                 "GET metadata returns latest (v1)", f"{r.status_code}")
    r = _req("GET", f"/documents/{doc_id}/download", headers=service)
    report.check(
        r.status_code == 200 and r.content == content_v1,
        "download returns v1 bytes",
        f"{r.status_code} bytes={len(r.content)}",
    )
    r = _req("GET", "/documents", headers=service)
    listed = {row.get("doc_id"): row.get("version") for row in r.json()} \
        if r.status_code == 200 else {}
    report.check(r.status_code == 200 and listed.get(doc_id) == 1,
                 "list includes the document (latest v1)", f"{r.status_code}")

    # -- immutable version 2 ------------------------------------------------
    r = _upload(f"/documents/{doc_id}/versions", headers=service,
                filename=filename, content=content_v2)
    report.check(r.status_code == 201 and r.json().get("version") == 2,
                 "service adds version 2", f"{r.status_code} {r.text[:120]}")
    if r.status_code == 201:
        cleanup["objects"].add(object_key(doc_id, 2, filename))
    report.check(
        r.status_code == 201 and r.json().get("content_hash") == _sha(content_v2),
        "version 2 has its own hash",
        r.json().get("content_hash", "")[:16] if r.status_code == 201 else "",
    )
    try:
        v1_still = minio_client.download_document_bytes(key_v1)
    except Exception as exc:  # noqa: BLE001
        v1_still, v1_detail = b"", str(exc)
    else:
        v1_detail = "v1 object untouched"
    report.check(v1_still == content_v1, "v1 object is immutable after v2", v1_detail)
    r = _req("GET", f"/documents/{doc_id}", headers=service)
    latest_ok = r.status_code == 200 and r.json().get("version") == 2
    report.check(latest_ok, "GET metadata follows the latest version (v2)",
                 f"v={r.json().get('version') if r.status_code == 200 else '?'}")
    r = _req("GET", f"/documents/{doc_id}/download", headers=service)
    report.check(r.status_code == 200 and r.content == content_v2,
                 "download returns the latest bytes (v2)",
                 f"{r.status_code} bytes={len(r.content)}")

    # -- filename sanitisation ---------------------------------------------
    r = _upload("/documents", headers=service, filename="../../evil.txt",
                content=b"sanitised")
    evil = r.json() if r.status_code == 201 else {}
    if evil.get("doc_id"):
        cleanup["doc_ids"].add(evil["doc_id"])
        cleanup["objects"].add(object_key(evil["doc_id"], 1, evil["filename"]))
    report.check(
        r.status_code == 201
        and evil.get("filename") == "evil.txt"
        and ".." not in object_key(evil.get("doc_id", "x"), 1, evil.get("filename", "x")),
        "upload filename is sanitised (no path traversal)",
        f"{r.status_code} filename={evil.get('filename')}",
    )

    if not jwt_ok:
        report.skip("ACL: JWT user checks", "JWT_SECRET is empty")
        return

    # -- ACL: service-owned doc is invisible to users until granted --------
    a = _user_headers(user_a)
    b = _user_headers(user_b)
    r = _req("GET", f"/documents/{doc_id}", headers=a)
    report.check(r.status_code == 404,
                 "ACL: un-granted user gets 404 (service-owned doc)",
                 f"{r.status_code}")
    r = _req("POST", f"/documents/{doc_id}/grants", headers=service,
             json={"principal_user_id": user_a, "permission": "read"})
    report.check(r.status_code == 201, "ACL: service grants read to user A",
                 f"{r.status_code}")
    report.check(_req("GET", f"/documents/{doc_id}", headers=a).status_code == 200,
                 "ACL: user A reads after the read grant", "200")
    report.check(_req("GET", f"/documents/{doc_id}", headers=b).status_code == 404,
                 "ACL: user B still gets 404", "404")
    r = _req("GET", "/documents", headers=b)
    listed_b = {row.get("doc_id") for row in r.json()} if r.status_code == 200 else set()
    report.check(doc_id not in listed_b, "ACL: user B's list omits the doc",
                 f"{len(listed_b)} docs")
    r = _req("GET", "/documents", headers=a)
    listed_a = {row.get("doc_id") for row in r.json()} if r.status_code == 200 else set()
    report.check(doc_id in listed_a, "ACL: user A's list includes the doc",
                 f"{len(listed_a)} docs")

    r = _upload(f"/documents/{doc_id}/versions", headers=b,
                filename=filename, content=b"nope")
    report.check(r.status_code == 404,
                 "ACL: read-only user cannot add a version (404, not 403)",
                 f"{r.status_code}")
    r = _req("POST", f"/documents/{doc_id}/grants", headers=b,
             json={"principal_user_id": user_b, "permission": "read"})
    report.check(r.status_code == 404,
                 "ACL: non-owner cannot grant (404, not 403)", f"{r.status_code}")

    r = _req("POST", f"/documents/{doc_id}/grants", headers=service,
             json={"principal_user_id": user_a, "permission": "write"})
    report.check(r.status_code == 201, "ACL: service grants write to user A",
                 f"{r.status_code}")
    v3 = f"manual document v3 by A {tag}\n".encode()
    r = _upload(f"/documents/{doc_id}/versions", headers=a,
                filename=filename, content=v3)
    report.check(r.status_code == 201 and r.json().get("version") == 3,
                 "ACL: write grant allows user A to add v3",
                 f"{r.status_code} v={r.json().get('version') if r.status_code == 201 else '?'}")
    if r.status_code == 201:
        cleanup["objects"].add(object_key(doc_id, 3, filename))

    # -- JWT-owned document: service key and other users are denied ---------
    user_content = f"owned by A {tag}\n".encode()
    r = _upload("/documents", headers=a, filename=f"owned-{tag}.txt",
                content=user_content, title="Owned by A")
    owned = r.json() if r.status_code == 201 else {}
    owned_id = owned.get("doc_id", "")
    if owned_id:
        cleanup["doc_ids"].add(owned_id)
        cleanup["objects"].add(object_key(owned_id, 1, f"owned-{tag}.txt"))
    report.check(
        r.status_code == 201 and owned.get("owner_user_id") == user_a,
        "JWT upload is owned by the JWT sub",
        f"{r.status_code} owner={owned.get('owner_user_id')}",
    )
    if owned_id:
        report.check(_req("GET", f"/documents/{owned_id}", headers=a).status_code == 200,
                     "ACL: owner reads their document", "200")
        report.check(_req("GET", f"/documents/{owned_id}", headers=b).status_code == 404,
                     "ACL: other user gets 404 on the owned doc", "404")
        report.check(_req("GET", f"/documents/{owned_id}", headers=service).status_code == 404,
                     "ACL: service key gets 404 on a user-owned doc", "404")
        report.check(
            _req("GET", f"/documents/{owned_id}/download", headers=b).status_code == 404,
            "ACL: denied download is 404 (no existence leak)", "404",
        )
        r = _req("GET", "/documents", headers=service)
        service_list = {row.get("doc_id") for row in r.json()} if r.status_code == 200 else set()
        report.check(owned_id not in service_list,
                     "ACL: service list omits user-owned docs", f"{len(service_list)} docs")
        r = _req("POST", f"/documents/{owned_id}/grants", headers=b,
                 json={"principal_user_id": user_b, "permission": "read"})
        report.check(r.status_code == 404,
                     "ACL: non-owner service/user cannot grant on owned doc",
                     f"{r.status_code}")
        r = _req("POST", f"/documents/{owned_id}/grants", headers=a,
                 json={"principal_user_id": user_b, "permission": "read"})
        report.check(r.status_code == 201, "ACL: owner grants read to B",
                     f"{r.status_code}")
        report.check(_req("GET", f"/documents/{owned_id}", headers=b).status_code == 200,
                     "ACL: B reads after the owner grant", "200")

    # -- missing vs denied are the same answer ------------------------------
    missing = f"doc_missing_{tag}"
    for method, path in (
        ("GET", f"/documents/{missing}"),
        ("GET", f"/documents/{missing}/download"),
        ("POST", f"/documents/{missing}/versions"),
        ("POST", f"/documents/{missing}/grants"),
    ):
        kwargs = {"json": {"principal_user_id": user_b, "permission": "read"}} \
            if method == "POST" else {}
        if path.endswith("/versions"):
            r = _upload(path, headers=service, filename="x.txt", content=b"x")
        else:
            r = _req(method, path, headers=service, **kwargs)
        report.check(r.status_code == 404,
                     f"missing doc: {method} {path.split('/documents/')[-1]} → 404",
                     f"{r.status_code}")


# --- Ingest + DOCUMENTED_BY --------------------------------------------------


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{doc_id}"))


def scenario_ingest(report: Report, db, cleanup: dict) -> None:
    print("\n=== ingest + DOCUMENTED_BY + projection drain ===")
    tag = secrets.token_hex(4)
    linked_doc = f"manual-doc-linked-{tag}"
    plain_doc = f"manual-doc-plain-{tag}"
    source = f"documents-manual-{tag}.json"
    cleanup["ingest_docs"].update({linked_doc, plain_doc})
    cleanup["ingest_sources"].add(source)

    rows = [
        {
            "id": linked_doc,
            "title": f"Manual linked document {tag}",
            "content": f"Manual ingest document about {ASSET_ID}: {uuid.uuid4()}",
            "category": "manual",
            "asset_id": ASSET_ID,
        },
        {
            "id": plain_doc,
            "title": f"Manual unlinked document {tag}",
            "content": f"Manual ingest document with no asset link: {uuid.uuid4()}",
            "category": "manual",
        },
    ]
    inbox = tempfile.mkdtemp(prefix="manual-ingest-")
    cleanup["tmpdirs"].add(inbox)
    (Path(inbox) / source).write_text(json.dumps(rows), encoding="utf-8")

    summary = run_batch(source=LocalDirSource(inbox), db=db)
    report.check(
        summary["files"] == 1 and summary["processed"] == 1 and not summary["failed"],
        "ingest batch processed the documents file",
        json.dumps({k: summary[k] for k in ("files", "processed", "failed")}),
    )
    report.check(
        summary["rows_ok"] == 2 and summary["rows_failed"] == 0,
        "both document rows loaded (2 ok / 0 failed)",
        f"rows_ok={summary['rows_ok']} rows_failed={summary['rows_failed']}",
    )
    run = db.query(IngestRun).filter_by(source=source).order_by(
        IngestRun.started_at.desc()).first()
    report.check(run is not None and run.status == "SUCCEEDED",
                 "ingest_runs records SUCCEEDED",
                 f"status={getattr(run, 'status', None)}")
    report.check(
        db.query(IngestQuarantine).filter_by(source=source).count() == 0,
        "no quarantine rows for the file", "0",
    )

    # -- Qdrant: both points, payload carries the link ---------------------
    client = qdrant_client.get_client()
    points = client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[_point_id(linked_doc), _point_id(plain_doc)],
        with_payload=True,
    )
    by_id = {str(p.id): (p.payload or {}) for p in points}
    linked_payload = by_id.get(_point_id(linked_doc), {})
    plain_payload = by_id.get(_point_id(plain_doc), {})
    report.check(
        len(by_id) == 2 and linked_payload.get("doc_id") == linked_doc,
        "Qdrant holds both document points",
        f"points={len(by_id)}",
    )
    report.check(
        linked_payload.get("asset_id") == ASSET_ID,
        "Qdrant payload carries asset_id for the linked doc",
        str(linked_payload.get("asset_id")),
    )
    report.check(
        not plain_payload.get("asset_id"),
        "Qdrant payload stays unlinked for the plain doc",
        str(plain_payload.get("asset_id")),
    )

    # -- canonical relation (L1) -------------------------------------------
    edge = (
        db.query(OntologyRelationship)
        .filter_by(
            relationship_type="DOCUMENTED_BY",
            source_type="Asset",
            source_key=ASSET_ID,
            target_type="Document",
            target_key=linked_doc,
        )
        .first()
    )
    report.check(edge is not None, "canonical DOCUMENTED_BY edge exists (L1)",
                 f"{ASSET_ID} → {linked_doc}")
    plain_edges = (
        db.query(OntologyRelationship)
        .filter_by(relationship_type="DOCUMENTED_BY", target_key=plain_doc)
        .count()
    )
    report.check(plain_edges == 0, "no invented edge for the plain doc", str(plain_edges))
    report.check(
        db.query(OntologyEntity).filter_by(entity_type="Document",
                                           entity_key=linked_doc).count() == 1,
        "canonical Document endpoint exists", linked_doc,
    )

    # -- projection queue → drain → Neo4j ----------------------------------
    subject_ref = (
        f"rel:DOCUMENTED_BY:Asset:{ASSET_ID}:Document:{linked_doc}"
    )
    state = (
        db.query(OntologyProjectionState).filter_by(subject_ref=subject_ref).first()
    )
    report.check(state is not None, "projection state queued for the edge",
                 f"status={getattr(state, 'status', None)}")
    try:
        drained = drain_pending_projections(db, limit=200)
        drain_ok = drained.get("errors", 0) == 0
        drain_detail = json.dumps(drained)
    except Exception as exc:  # noqa: BLE001
        drain_ok, drain_detail = False, str(exc)
    report.check(drain_ok, "projection drain completes without errors", drain_detail)

    state = (
        db.query(OntologyProjectionState).filter_by(subject_ref=subject_ref).first()
    )
    report.check(
        state is not None
        and state.projected_version == state.canonical_version
        and state.status == "synced",
        "projection state marks the edge projected",
        f"status={getattr(state, 'status', None)} "
        f"v={getattr(state, 'projected_version', None)}/"
        f"{getattr(state, 'canonical_version', None)}",
    )

    try:
        neighbours = graph_query.neighbors("Asset", ASSET_ID, depth=1, limit=200)
        linked = graph_query.linked_documents("Asset", ASSET_ID, limit=200)
    except Exception as exc:  # noqa: BLE001
        neighbours, linked = [], []
        graph_detail = str(exc)
    else:
        graph_detail = f"{len(neighbours)} neighbours"
    report.check(
        any(n.get("type") == "DOCUMENTED_BY" and n.get("to") == linked_doc
            for n in neighbours),
        "neighbors(O21) includes DOCUMENTED_BY → the new doc",
        graph_detail,
    )
    report.check(
        any(d.get("document_id") == linked_doc for d in linked),
        "linked_documents(O21) includes the new doc",
        f"{len(linked)} docs",
    )
    report.check(
        not any(n.get("type") == "DOCUMENTED_BY" and n.get("to") == plain_doc
                for n in neighbours),
        "neighbors(O21) has no edge for the plain doc", "absent",
    )

    # -- replay is idempotent ----------------------------------------------
    replay = run_batch(source=LocalDirSource(inbox), db=db, force=True)
    report.check(
        replay["rows_ok"] == 2 and replay["rows_failed"] == 0,
        "forced re-run replays both rows", f"rows_ok={replay['rows_ok']}")
    edge_count = (
        db.query(OntologyRelationship)
        .filter_by(
            relationship_type="DOCUMENTED_BY",
            source_type="Asset",
            source_key=ASSET_ID,
            target_type="Document",
            target_key=linked_doc,
        )
        .count()
    )
    report.check(edge_count == 1, "re-run does not duplicate the canonical edge",
                 f"edges={edge_count}")
    replay_points = client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[_point_id(linked_doc), _point_id(plain_doc)],
        with_payload=False,
    )
    report.check(len(replay_points) == 2,
                 "re-run does not duplicate the Qdrant points",
                 f"points={len(replay_points)}")
    try:
        after = drain_pending_projections(db, limit=200)
        again_ok = after.get("errors", 0) == 0
    except Exception as exc:  # noqa: BLE001
        again_ok, after = False, str(exc)
    report.check(again_ok, "drain after replay stays error-free", str(after))


# --- Cleanup -----------------------------------------------------------------


def cleanup_all(report: Report, db, cleanup: dict) -> None:
    """Best-effort removal of everything this run created (never O21)."""
    print("\n=== cleanup ===")
    errors: list[str] = []

    for key in cleanup["objects"]:
        try:
            minio_client.get_client().remove_object(BUCKET, key)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"minio {key}: {exc}")

    doc_ids = list(cleanup["doc_ids"])
    if doc_ids:
        try:
            db.query(DocumentACL).filter(DocumentACL.doc_id.in_(doc_ids)).delete(
                synchronize_session=False)
            db.query(Document).filter(Document.doc_id.in_(doc_ids)).delete(
                synchronize_session=False)
            db.commit()
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            errors.append(f"documents rows: {exc}")

    ingest_docs = list(cleanup["ingest_docs"])
    if ingest_docs:
        try:
            client = qdrant_client.get_client()
            from qdrant_client.http import models as qm

            client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=qm.PointIdsList(
                    points=[_point_id(d) for d in ingest_docs]
                ),
                wait=True,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"qdrant points: {exc}")

        refs = [
            f"rel:DOCUMENTED_BY:Asset:{ASSET_ID}:Document:{d}" for d in ingest_docs
        ]
        try:
            db.query(OntologyProjectionState).filter(
                OntologyProjectionState.subject_ref.in_(refs)
            ).delete(synchronize_session=False)
            db.query(OntologyVersion).filter(
                OntologyVersion.subject_ref.in_(refs)
            ).delete(synchronize_session=False)
            db.query(OntologyRelationship).filter(
                OntologyRelationship.relationship_type == "DOCUMENTED_BY",
                OntologyRelationship.target_type == "Document",
                OntologyRelationship.target_key.in_(ingest_docs),
            ).delete(synchronize_session=False)
            db.query(OntologyEntity).filter(
                OntologyEntity.entity_type == "Document",
                OntologyEntity.entity_key.in_(ingest_docs),
            ).delete(synchronize_session=False)
            db.commit()
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            errors.append(f"canonical rows: {exc}")

        try:
            store = get_graph_store()
            for d in ingest_docs:
                store.delete_relationship(
                    "Asset", ASSET_ID, "DOCUMENTED_BY", "Document", d)
                store.delete_node("Document", d)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"neo4j: {exc}")

    sources = list(cleanup["ingest_sources"])
    if sources:
        try:
            db.query(IngestQuarantine).filter(
                IngestQuarantine.source.in_(sources)
            ).delete(synchronize_session=False)
            db.query(IngestRun).filter(IngestRun.source.in_(sources)).delete(
                synchronize_session=False)
            db.commit()
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            errors.append(f"ingest rows: {exc}")

    for tmpdir in cleanup["tmpdirs"]:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if errors:
        report.warn("cleanup left residue", "; ".join(errors)[:900])
    else:
        print("  cleanup: removed documents, objects, points, canonical rows, ingest runs")


# --- Main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true",
                        help="keep the rows/objects/points this run creates")
    parser.add_argument("--documents-only", action="store_true")
    parser.add_argument("--ingest-only", action="store_true")
    args = parser.parse_args(argv)

    report = Report()
    cleanup: dict = {
        "doc_ids": set(),
        "objects": set(),
        "ingest_docs": set(),
        "ingest_sources": set(),
        "tmpdirs": set(),
    }
    db = SessionLocal()
    try:
        ready = check_environment(report, db)
        check_auth_config(report)
        if not ready:
            report.warn("live checks skipped", "API is not healthy")
        else:
            if not args.ingest_only:
                scenario_documents(report, cleanup)
            if not args.documents_only:
                scenario_ingest(report, db, cleanup)
        if args.keep:
            print("\n--keep: leaving created rows/objects/points in place")
        else:
            cleanup_all(report, db, cleanup)
    finally:
        db.close()

    total = len(report.failures)
    print(
        f"\nchecks: {'PASS' if not total else 'FAIL'} — "
        f"{report.passed} passed, {report.skipped} skipped, "
        f"{len(report.warnings)} warning(s), {total} failure(s)"
    )
    print("FAIL: documents + ingest" if total else "PASS: documents + ingest")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
