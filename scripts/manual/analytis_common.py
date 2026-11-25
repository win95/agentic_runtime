"""Shared helpers for agent_analytis manual scripts."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path


def save_files(result: dict, out_dir: Path, execution_id: str) -> list[Path]:
    """Write ``output.files[]`` under ``out_dir/<execution_id>/``.

    Uses inline ``content`` when present; otherwise HTTP GETs ``url``.
    """
    output = result.get("output") or {}
    files = output.get("files") or []
    saved: list[Path] = []
    run_dir = out_dir / execution_id
    for entry in files:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        target = run_dir / Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = entry.get("content")
        if content is not None:
            text = content if isinstance(content, str) else str(content)
            target.write_text(text, encoding="utf-8")
            saved.append(target)
            print(f"  đã lưu: {target} ({len(text.encode('utf-8'))} bytes)")
            continue
        url = str(entry.get("url") or "").strip()
        if not url:
            continue
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                body = response.read()
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            print(f"  không tải được {name} từ url ({exc})", flush=True)
            continue
        target.write_bytes(body)
        saved.append(target)
        print(f"  đã lưu: {target} ({len(body)} bytes, từ presigned url)")
    return saved