"""Probe OpenSandbox from the same settings as the Celery worker.

Usage (repo root, server đang chạy, .env đã set SANDBOX_*):

    .venv/bin/python scripts/manual/probe_sandbox.py

Exit 0 = create + ls + execute OK; 1 = SDK/server error (in ra chi tiết).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from _api import load_dotenv  # noqa: E402


def main() -> int:
    load_dotenv()
    from app.config import settings
    from runtime.sandbox.opensandbox_backend import OpenSandboxBackend, format_command_output
    from runtime.sandbox.provider import default_client

    print(f"SANDBOX_ENABLED={settings.sandbox_enabled}")
    print(f"SANDBOX_DOMAIN={settings.sandbox_domain!r}")
    print(f"SANDBOX_USE_SERVER_PROXY={settings.sandbox_use_server_proxy}")
    print(f"SANDBOX_IMAGE={settings.sandbox_image!r}")
    key = (settings.sandbox_api_key or "").strip()
    print(f"SANDBOX_API_KEY={'(set)' if key else '(EMPTY — worker sẽ fallback in-memory)'}")

    if not settings.sandbox_enabled:
        print("probe_sandbox: bật SANDBOX_ENABLED=true trong .env và restart worker")
        return 2
    if not key:
        print("probe_sandbox: điền SANDBOX_API_KEY trùng key trong ~/.sandbox.toml")
        return 2

    client = default_client(settings)
    try:
        handle = client.create(settings.sandbox_image)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL create: {type(exc).__name__}: {exc}")
        return 1
    sid = getattr(handle, "id", None) or getattr(handle, "sandbox_id", "?")
    print(f"OK create sandbox_id={sid}")

    backend = OpenSandboxBackend(handle, str(sid))
    try:
        resp = backend.execute("python3 -c \"print('probe-ok')\"")
        print(f"OK execute exit={resp.exit_code} output={resp.output!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL execute: {type(exc).__name__}: {exc}")
        if hasattr(exc, "response_body") and exc.response_body:
            body = exc.response_body
            if isinstance(body, bytes):
                body = body[:500].decode("utf-8", errors="replace")
            print(f"  response_body={body!r}")
        return 1
    try:
        client.destroy(handle)
        print("OK destroy")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN destroy: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())