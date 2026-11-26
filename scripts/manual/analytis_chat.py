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
mean the agent really wrote and ran code in a sandbox. The HTML
report must also embed a Chart.js chart (``<canvas`` + ``new Chart(``)
— see ``html_has_chart`` in ``analytis_common`` — or the check fails.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _api import load_dotenv  # noqa: E402
from analytis_common_1 import (  # noqa: E402
    DEFAULT_API_URL,
    DEFAULT_OUT_DIR,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_TIMEOUT,
    SELF_TEST_QUESTION,
    export_checks,
    resolve_api_key,
    run_once,
)

PREFIX = "analytis_chat"


def _self_test(api_url: str, out_dir: Path, interval: float, timeout: float) -> int:
    """One canned research question + PASS/FAIL checks. Returns the exit code."""
    session_id = f"sess-analytis-test-{uuid.uuid4().hex[:8]}"
    print(f"self-test: session {session_id}")
    print(f"self-test: hỏi: {SELF_TEST_QUESTION}")
    checks: list[tuple[str, bool]] = []
    try:
        _execution, result, _saved = run_once(
            api_url,
            SELF_TEST_QUESTION,
            session_id,
            out_dir,
            interval,
            timeout,
            prefix=PREFIX,
        )
        checks = export_checks(result, succeeded=True)
    except SystemExit as exc:
        checks = [(f"run hoàn thành (exit {exc.code})", False)]
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
        resolve_api_key(args)
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
            run_once(
                args.api_url,
                args.message,
                session_id,
                out_dir,
                args.interval,
                args.timeout,
                prefix=PREFIX,
            )
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
                run_once(
                    args.api_url,
                    message,
                    session_id,
                    out_dir,
                    args.interval,
                    args.timeout,
                    prefix=PREFIX,
                )
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