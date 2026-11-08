"""MinIO client — files, artifacts, reports (Layer 6)."""

from __future__ import annotations

import io

from app.config import settings


def get_client():
    """Build the MinIO client, or None when credentials are not configured."""
    if not settings.minio_access_key or not settings.minio_secret_key:
        return None
    from minio import Minio

    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def ensure_bucket(bucket: str | None = None) -> None:
    client = get_client()
    if client is None:
        raise RuntimeError("MinIO not configured (MINIO_ACCESS_KEY/SECRET_KEY empty)")
    bucket = bucket or settings.minio_bucket
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def ensure_reports_bucket() -> None:
    """Create the `reports` bucket (used for execution reports) if missing."""
    ensure_bucket(settings.minio_reports_bucket)


def upload_report(key: str, markdown: str) -> str:
    """Upload an execution report to the reports bucket.

    Returns the object key. Raises RuntimeError when MinIO is not
    configured or unreachable — the caller stores the report inline
    instead.
    """
    client = get_client()
    if client is None:
        raise RuntimeError("MinIO not configured (MINIO_ACCESS_KEY/SECRET_KEY empty)")
    ensure_reports_bucket()
    data = markdown.encode("utf-8")
    client.put_object(
        settings.minio_reports_bucket,
        key,
        io.BytesIO(data),
        len(data),
        content_type="text/markdown",
    )
    return key


def report_location(key: str) -> str:
    """MinIO-style location string for a stored report object."""
    return f"minio://{settings.minio_reports_bucket}/{key}"


def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    client = get_client()
    if client is None:
        raise RuntimeError("MinIO not configured (MINIO_ACCESS_KEY/SECRET_KEY empty)")
    ensure_bucket()
    client.put_object(
        settings.minio_bucket, key, io.BytesIO(data), len(data), content_type=content_type
    )
    return key


def download_bytes(key: str) -> bytes:
    client = get_client()
    if client is None:
        raise RuntimeError("MinIO not configured (MINIO_ACCESS_KEY/SECRET_KEY empty)")
    response = client.get_object(settings.minio_bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()
