"""Agent-analytis test CLI — research + report files over the platform API.

A real test interface for ``agent_analytis_mission``: send a message,
poll until the run finishes, print the answer and save the exported
report files (``output.files[]``) to ``--out-dir``. Stdlib + the sibling
``_api`` helper only (no extra dependency).

Flow per message: ``POST /executions`` (mission
``agent_analytis_mission``, ``input={"message": ...}``) -> 202
``{execution_id, session_id}`` -> poll ``GET /executions/{id}`` until
terminal (``wait_for``) -> ``GET /executions/{id}/result`` -> print
``output["answer"]`` and write each ``output["files"]`` entry
(``{name, path, content}``) under ``--out-dir/<execution_id>/``.

Auth is API-key only: the mission is admin/API-key-only (a plain JWT
user gets 404 for it — see ``require_mission_for_role``), so the CLI
uses the ``X-API-Key`` the ``_api`` helper sends from ``API_KEY``
(repo ``.env`` or ``--api-key``). A ``session_id`` is always sent:
the sandbox backend only exists for sessions, and the REPL reuses one
``session_id`` across turns so the sandbox survives and the agent can
edit earlier files in place.

Usage::

    .venv/bin/python -m scripts.manual.analytis_chat                  # interactive REPL
    .venv/bin/python -m scripts.manual.analytis_chat "nghiên cứu giá cà phê robusta"  # one shot
    .venv/bin/python -m scripts.manual.analytis_chat --self-test     # canned checks, exit 0/1
    .venv/bin/python -m scripts.manual.analytis_chat --session sess_x "..."  # resume session

Env: ``API_BASE_URL`` (default ``http://localhost:8000``), ``API_KEY``.
``/quit`` (or Ctrl-D) leaves the REPL. Exit codes: 0 ok / self-test
all passed, 1 the execution failed or a self-test check failed, 2
usage, auth or connection problem.

Self-test checks (one canned research question): the run SUCCEEDED,
the answer is non-empty, and ``files`` contains ``.html`` + ``.csv`` +
``.md``. The files check also proves the sandbox backend ran: the
in-memory fallback exports nothing (``files: []``), so non-empty files
mean the agent really wrote and ran code in a sandbox.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _api import (  # noqa: E402
    api_key,
    get_result,
    load_dotenv,
    submit,
    wait_for,
)

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_MISSION = "agent_analytis_mission"
DEFAULT_OUT_DIR = "analytis-output"
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_TIMEOUT = 600.0

SELF_TEST_QUESTION = (
    "Nghiên cứu 3 nguồn mới nhất về giá cà phê robusta Việt Nam, "
    "tóm tắt xu hướng giá và xuất báo cáo HTML (bảng + biểu đồ), "
    "dữ liệu CSV và tóm tắt Markdown."
)
SELF_TEST_EXTENSIONS = (".html", ".csv", ".md")


def _die(message: str, code: int = 2) -> "SystemExit":
    print(f"analytis_chat: {message}", file=sys.stderr)
    return SystemExit(code)


def _resolve_api_key(args: argparse.Namespace) -> None:
    if args.api_key:
        os.environ["API_KEY"] = args.api_key
    if not api_key():
        raise _die(
            "chưa có API_KEY. Đặt API_KEY trong repo .env hoặc truyền --api-key <key>.\n"
            "Mission này chỉ chạy bằng API key (JWT thường bị 404) — xem require_mission_for_role."
        )


def _save_files(result: dict, out_dir: Path, execution_id: str) -> list[Path]:
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
        # name is a basename (export.py); keep it flat under the run dir
        # and refuse any path escape just in case the payload is odd.
        target = run_dir / Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        text = content if isinstance(content, str) else str(content)
        target.write_text(text, encoding="utf-8")
        saved.append(target)
        print(f"  đã lưu: {target} ({len(text.encode('utf-8'))} bytes)")
    return saved


def _print_answer(result: dict) -> None:
    output = result.get("output") or {}
    answer = str(output.get("answer") or "").strip()
    print(f"analytis> {answer}" if answer else "analytis> (không có câu trả lời)")
    actions = output.get("actions_taken")
    if isinstance(actions, list) and actions:
        print(f"  (đã làm: {', '.join(str(a) for a in actions)})")
    files = output.get("files") or []
    if isinstance(files, list):
        print(f"  (files: {len(files)} — xem --out-dir)")


def _run_once(api_url: str, message: str, session_id: str,
              out_dir: Path, interval: float, timeout: float) -> tuple[dict, dict, list[Path]]:
    """One message -> (execution, result, saved files). Raises SystemExit on failure."""
    try:
        sent = submit(DEFAULT_MISSION, {"message": message}, session_id=session_id)
    except Exception as exc:  # noqa: BLE001 — connection + HTTP errors surface here
        hint = " — kiểm tra API_KEY trong .env" if "401" in str(exc) else ""
        raise _die(
            f"POST /executions thất bại ({exc}){hint} — kiểm tra API ở {api_url} đã chạy chưa."
        )
    execution_id = str(sent.get("execution_id") or "")
    if not execution_id:
        raise _die(f"POST /executions không trả execution_id: {sent}")
    session_id = str(sent.get("session_id") or session_id or "")
    print(f"... đang chờ (execution {execution_id}, session {session_id})", file=sys.stderr)
    try:
        execution, _elapsed = wait_for(execution_id, timeout=timeout, interval=interval)
    except TimeoutError as exc:
        raise _die(str(exc), code=1)
    except Exception as exc:  # noqa: BLE001
        raise _die(f"poll kết quả thất bại ({exc}).", code=1)
    status = str(execution.get("status") or "")
    if status != "SUCCEEDED":
        print(
            f"execution {execution_id}: status={status}"
            f" error={execution.get('error') or '-'}",
            file=sys.stderr,
        )
        raise _die(f"run không thành công (status={status}).", code=1)
    try:
        result = get_result(execution_id)
    except Exception as exc:  # noqa: BLE001
        raise _die(f"GET result thất bại ({exc}).", code=1)
    _print_answer(result)
    saved = _save_files(result, out_dir, execution_id)
    if not saved:
        print("  (không có file nào được xuất — xem lại output/ trong sandbox)", file=sys.stderr)
    return execution, result, saved


def _self_test(api_url: str, out_dir: Path, interval: float, timeout: float) -> int:
    """One canned research question + PASS/FAIL checks. Returns the exit code."""
    session_id = f"sess-analytis-test-{uuid.uuid4().hex[:8]}"
    print(f"self-test: session {session_id}")
    print(f"self-test: hỏi: {SELF_TEST_QUESTION}")
    checks: list[tuple[str, bool]] = []
    try:
        _execution, result, _saved = _run_once(
            api_url, SELF_TEST_QUESTION, session_id, out_dir, interval, timeout
        )
        checks.append(("run SUCCEEDED", True))
        output = result.get("output") or {}
        answer = str(output.get("answer") or "").strip()
        checks.append(("answer không rỗng", bool(answer)))
        files = output.get("files") or []
        names = [str(f.get("name") or f.get("path") or "") for f in files if isinstance(f, dict)]
        for ext in SELF_TEST_EXTENSIONS:
            checks.append((f"files có {ext}", any(n.endswith(ext) for n in names)))
        checks.append(("files không rỗng (sandbox đã chạy)", bool(files)))
    except SystemExit as exc:
        checks.append((f"run hoàn thành (exit {exc.code})", False))
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if failed:
        print(f"self-test: {len(failed)}/{len(checks)} check FAILED.")
        return 1
    print(f"self-test: {len(checks)}/{len(checks)} check PASSED.")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Test agent-analytis qua platform API (POST /executions, API key).",
    )
    parser.add_argument("message", nargs="?", help="một câu hỏi gửi rồi thoát (không có = REPL)")
    parser.add_argument("--api-url", default=os.environ.get("API_BASE_URL", DEFAULT_API_URL),
                        help="platform API base URL (mặc định %(default)s)")
    parser.add_argument("--api-key", default=None, help="API key (ưu tiên hơn API_KEY trong .env)")
    parser.add_argument("--session", default=None, help="tiếp tục session_id có sẵn")
    parser.add_argument("--new", action="store_true", help="bỏ session cũ, bắt đầu session mới")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="thư mục lưu file báo cáo (mặc định %(default)s)")
    parser.add_argument("--interval", type=float, default=DEFAULT_POLL_INTERVAL,
                        help="giây giữa 2 lần poll (mặc định %(default)s)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="chờ tối đa bao nhiêu giây (mặc định %(default)s)")
    parser.add_argument("--self-test", action="store_true",
                        help="chạy 1 câu hỏi mẫu + kiểm tra PASS/FAIL rồi thoát")
    parser.add_argument("--verbose", action="store_true", help="bật log HTTP của _api")
    args = parser.parse_args(argv)
    if not args.verbose:
        os.environ["MANUAL_LOG"] = "0"
    if args.api_url:
        os.environ["API_BASE_URL"] = args.api_url

    try:
        _resolve_api_key(args)
    except SystemExit as exc:
        return int(exc.code or 2)

    out_dir = Path(args.out_dir)
    try:
        if args.self_test:
            return _self_test(args.api_url, out_dir, args.interval, args.timeout)
        session_id = None if args.new else args.session
        if session_id is None:
            session_id = f"sess-analytis-{uuid.uuid4().hex[:12]}"
        print(f"session: {session_id} (file lưu ở {out_dir}/<execution_id>/)")
        if args.message is not None:
            _run_once(args.api_url, args.message, session_id, out_dir,
                      args.interval, args.timeout)
            return 0
        print("Chat với agent-analytis (gõ /quit để thoát).")
        while True:
            try:
                message = input("bạn> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                break
            if not message:
                continue
            if message in ("/quit", "/exit"):
                break
            try:
                _run_once(args.api_url, message, session_id, out_dir,
                          args.interval, args.timeout)
            except SystemExit as exc:
                print(f"(lỗi: exit {exc.code})", file=sys.stderr)
                if exc.code == 2:
                    return int(exc.code)
                continue
        return 0
    except SystemExit as exc:
        return int(exc.code or 2)


if __name__ == "__main__":
    raise SystemExit(main())
