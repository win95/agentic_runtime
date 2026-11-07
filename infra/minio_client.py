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


def ensure_bucket() -> None:
    client = get_client()
    if client is None:
        raise RuntimeError("MinIO not configured (MINIO_ACCESS_KEY/SECRET_KEY empty)")
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)


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
