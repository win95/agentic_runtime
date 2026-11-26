"""Shared helpers for agent-analytis manual scripts (chat CLI + scenario suite)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _api import api_key, get_result, submit, wait_for

DEFAULT_API_URL = "http://localhost:8000"
MISSION_ID = "agent_analytis_mission"
DEFAULT_OUT_DIR = "analytis-output"
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_TIMEOUT = 600.0
ANSWER_LOG_MAX = 400

SELF_TEST_QUESTION = (
    "Nghiên cứu 3 nguồn mới nhất về giá cà phê robusta Việt Nam, "
    "tóm tắt xu hướng giá và xuất báo cáo HTML (bảng + biểu đồ), "
    "dữ liệu CSV và tóm tắt Markdown."
)
REQUIRED_EXTENSIONS = (".html", ".csv", ".md")


def die(message: str, code: int = 2, *, prefix: str = "analytis") -> SystemExit:
    print(f"{prefix}: {message}", file=sys.stderr)
    return SystemExit(code)


def resolve_api_key(args: argparse.Namespace | None = None, api_key_flag: str | None = None) -> None:
    """Ensure API_KEY is set (from args or env). Raises SystemExit(2) if missing."""
    if args is not None and getattr(args, "api_key", None):
        os.environ["API_KEY"] = args.api_key
    if api_key_flag:
        os.environ["API_KEY"] = api_key_flag
    if not api_key():
        raise die(
            "chưa có API_KEY. Đặt API_KEY trong repo .env hoặc truyền --api-key <key>.\n"
            "Mission này chỉ chạy bằng API key (JWT thường bị 404) — xem require_mission_for_role."
        )


def html_has_chart(html: str) -> bool:
    """True when an HTML report embeds a Chart.js chart (canvas + constructor call)."""
    return "<canvas" in html and "new Chart(" in html


def save_files(result: dict, out_dir: Path, execution_id: str) -> list[Path]:
    """Write output.files[] under out_dir/<execution_id>/; return the paths."""
    output = result.get("output") or {}
    files = output.get("files") or []
    saved: list[Path] = []
    run_dir = out_dir / execution_id
    for entry in files:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        content = entry.get("content")
        if not name or content is None:
            continue
        target = run_dir / Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        text = content if isinstance(content, str) else str(content)
        target.write_text(text, encoding="utf-8")
        saved.append(target)
        print(f"  đã lưu: {target} ({len(text.encode('utf-8'))} bytes)")
    return saved


def _short_answer(text: str, limit: int = ANSWER_LOG_MAX) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def print_turn_summary(
    result: dict,
    *,
    scenario: str | None = None,
    session_id: str | None = None,
    out_dir: Path | None = None,
) -> None:
    """Minimal stdout after a successful run (full answer also logged via _api === A)."""
    if scenario or session_id:
        bits = [f"scenario={scenario or '-'}", f"session={session_id or '-'}"]
        if out_dir is not None:
            bits.append(f"out_dir={out_dir}")
        print("  " + ", ".join(bits))
    output = result.get("output") or {}
    answer = str(output.get("answer") or "").strip()
    if answer:
        print(f"analytis> {_short_answer(answer)}")
    else:
        print("analytis> (không có câu trả lời)")
    actions = output.get("actions_taken")
    if isinstance(actions, list) and actions:
        print(f"  (đã làm: {', '.join(str(a) for a in actions)})")
    files = output.get("files") or []
    if isinstance(files, list):
        where = str(out_dir) if out_dir is not None else "--out-dir"
        print(f"  (files: {len(files)} — xem {where})")


def export_checks(result: dict, *, succeeded: bool = True) -> list[tuple[str, bool]]:
    """Pure PASS/FAIL labels for a completed research+report run."""
    checks: list[tuple[str, bool]] = []
    checks.append(("run SUCCEEDED", succeeded))
    output = result.get("output") or {}
    answer = str(output.get("answer") or "").strip()
    checks.append(("answer không rỗng", bool(answer)))
    files = output.get("files") or []
    names = [str(f.get("name") or f.get("path") or "") for f in files if isinstance(f, dict)]
    for ext in REQUIRED_EXTENSIONS:
        checks.append((f"files có {ext}", any(n.endswith(ext) for n in names)))
    checks.append(("files không rỗng (sandbox đã chạy)", bool(files)))
    html_contents = [
        str(f.get("content") or "")
        for f in files
        if isinstance(f, dict) and str(f.get("name") or "").endswith(".html")
    ]
    checks.append(
        ("báo cáo HTML có biểu đồ Chart.js", any(html_has_chart(h) for h in html_contents))
    )
    return checks


def run_once(
    api_url: str,
    message: str,
    session_id: str,
    out_dir: Path,
    interval: float,
    timeout: float,
    *,
    scenario: str | None = None,
    prefix: str = "analytis",
) -> tuple[dict, dict, list[Path]]:
    """One message -> (execution, result, saved files). Raises SystemExit on failure."""
    try:
        sent = submit(MISSION_ID, {"message": message}, session_id=session_id)
    except Exception as exc:  # noqa: BLE001
        hint = " — kiểm tra API_KEY trong .env" if "401" in str(exc) else ""
        raise die(
            f"POST /executions thất bại ({exc}){hint} — kiểm tra API ở {api_url} đã chạy chưa.",
            prefix=prefix,
        )
    execution_id = str(sent.get("execution_id") or "")
    if not execution_id:
        raise die(f"POST /executions không trả execution_id: {sent}", prefix=prefix)
    session_id = str(sent.get("session_id") or session_id or "")
    print(f"... đang chờ (execution {execution_id}, session {session_id})", file=sys.stderr)
    try:
        execution, _elapsed = wait_for(execution_id, timeout=timeout, interval=interval)
    except TimeoutError as exc:
        raise die(str(exc), code=1, prefix=prefix)
    except Exception as exc:  # noqa: BLE001
        raise die(f"poll kết quả thất bại ({exc}).", code=1, prefix=prefix)
    status = str(execution.get("status") or "")
    if status != "SUCCEEDED":
        print(
            f"execution {execution_id}: status={status}"
            f" error={execution.get('error') or '-'}",
            file=sys.stderr,
        )
        raise die(f"run không thành công (status={status}).", code=1, prefix=prefix)
    try:
        result = get_result(execution_id)
    except Exception as exc:  # noqa: BLE001
        raise die(f"GET result thất bại ({exc}).", code=1, prefix=prefix)
    print_turn_summary(result, scenario=scenario, session_id=session_id, out_dir=out_dir)
    saved = save_files(result, out_dir, execution_id)
    output = result.get("output") or {}
    warnings = output.get("export_warnings") or []
    if warnings:
        for line in warnings:
            print(f"  export warning: {line}", file=sys.stderr)
    if not saved:
        print("  (không có file nào được xuất — xem lại output/ trong sandbox)", file=sys.stderr)
        raise die(
            "output schema yêu cầu files nhưng không có file nào được export.",
            code=1,
            prefix=prefix,
        )
    return execution, result, saved