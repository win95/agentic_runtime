"""Guest chat CLI — talk to the hotel receptionist over the public guest API.

A real chat interface for ``POST /guest/chat``: send a message, poll
``GET /executions/{id}/result`` until the answer is ready, print it.
Stdlib + the sibling ``_api`` helper only (no extra dependency).

Flow per message: ``POST /guest/chat`` -> 202 ``{execution_id, session_id}``
-> poll ``GET /executions/{id}/result`` every ``--interval`` seconds
(404 = "Result not ready yet") until 200 -> print ``output["answer"]``
(the result router already projects the guest-safe reply).

Auth is JWT-only. Resolve the token in this order:
``--token`` > ``GUEST_JWT`` env > ``--login`` (prompts email + password,
``POST {user-url}/auth/login`` on the user service) — otherwise the CLI
explains how to get a token and exits 2.

Usage::

    .venv/bin/python -m scripts.manual.guest_chat                  # interactive REPL
    .venv/bin/python -m scripts.manual.guest_chat "Tôi muốn đặt phòng"  # one message, exit
    .venv/bin/python -m scripts.manual.guest_chat --login          # login first
    .venv/bin/python -m scripts.manual.guest_chat --token "$JWT" --session sess_abc123 "..."

Env: ``GUEST_API_URL`` (default ``http://localhost:8000``),
``GUEST_JWT``, ``USER_SERVICE_URL`` (default ``http://localhost:8001``).
``/quit`` (or Ctrl-D) leaves the REPL. Exit codes: 0 ok, 1 the execution
failed / no result, 2 usage, auth or connection problem.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _api import call, load_dotenv  # noqa: E402

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_USER_URL = "http://localhost:8001"
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_TIMEOUT = 300.0
MAX_MESSAGE_CHARS = 4000


def _die(message: str, code: int = 2) -> "SystemExit":
    print(f"guest_chat: {message}", file=sys.stderr)
    return SystemExit(code)


def _login(user_url: str) -> str:
    """POST /auth/login on the user service; return the access token."""
    email = input("email: ").strip()
    password = getpass.getpass("password: ")
    if not email or not password:
        raise _die("email/password trống — không login được.")
    try:
        status, payload = call(
            "POST", "/auth/login", {"email": email, "password": password},
            base_url=user_url, log=False,
        )
    except Exception as exc:  # noqa: BLE001 — connection errors surface here
        raise _die(f"không tới được user service ở {user_url} ({exc}).")
    if status != 200 or not payload or not payload.get("access_token"):
        detail = (payload or {}).get("detail", payload) if payload else "no response"
        raise _die(f"login thất bại (HTTP {status}): {detail}")
    return str(payload["access_token"])


def _resolve_token(args: argparse.Namespace) -> str:
    if args.token:
        return args.token
    token = os.environ.get("GUEST_JWT", "").strip()
    if token:
        return token
    if args.login:
        return _login(args.user_url)
    raise _die(
        "chưa có JWT. Lấy token bằng 1 trong 3 cách:\n"
        "  --token <jwt> | GUEST_JWT=<jwt> | --login (hỏi email/password rồi gọi user service)"
    )


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _send_message(api_url: str, token: str, message: str, session_id: str | None) -> dict:
    body: dict[str, str] = {"message": message}
    if session_id:
        body["session_id"] = session_id
    try:
        status, payload = call(
            "POST", "/guest/chat", body,
            headers=_auth_headers(token), base_url=api_url, log=False,
        )
    except Exception as exc:  # noqa: BLE001 — connection errors surface here
        raise _die(f"không tới được API ở {api_url} ({exc}) — kiểm tra server đã chạy chưa.")
    if status == 401 or status == 403:
        raise _die(f"token không hợp lệ / hết hạn (HTTP {status}) — login lại rồi chạy tiếp.")
    if status == 429:
        raise _die("bị rate limit (20 tin/phút) — đợi một lát rồi gửi lại.")
    if status != 202 or not payload or not payload.get("execution_id"):
        detail = (payload or {}).get("detail", payload) if payload else "no response"
        raise _die(f"POST /guest/chat thất bại (HTTP {status}): {detail}")
    return payload


def _poll_result(api_url: str, token: str, execution_id: str,
                 interval: float, timeout: float) -> dict:
    """Poll GET /executions/{id}/result until 200. 404 = not ready yet."""
    started = time.monotonic()
    while True:
        try:
            status, payload = call(
                "GET", f"/executions/{execution_id}/result",
                headers=_auth_headers(token), base_url=api_url, log=False,
            )
        except Exception as exc:  # noqa: BLE001 — connection errors surface here
            raise _die(f"mất kết nối khi poll kết quả ({exc}).")
        if status == 200 and payload:
            return payload
        if status in (401, 403):
            raise _die(f"token không hợp lệ / hết hạn khi poll (HTTP {status}) — login lại.")
        if status not in (200, 404):
            detail = (payload or {}).get("detail", payload) if payload else "no response"
            raise _die(
                f"GET /executions/{execution_id}/result lỗi (HTTP {status}): {detail}",
                code=1,
            )
        if time.monotonic() - started >= timeout:
            _report_execution_status(api_url, token, execution_id)
            raise _die(
                f"hết {timeout:.0f}s vẫn chưa có kết quả cho {execution_id}.",
                code=1,
            )
        time.sleep(interval)


def _report_execution_status(api_url: str, token: str, execution_id: str) -> None:
    try:
        status, payload = call(
            "GET", f"/executions/{execution_id}",
            headers=_auth_headers(token), base_url=api_url, log=False,
        )
    except Exception:  # noqa: BLE001 — best effort only
        return
    if status == 200 and payload:
        print(
            f"execution {execution_id}: status={payload.get('status')}"
            f" error={payload.get('error') or '-'}",
            file=sys.stderr,
        )


def _print_answer(result: dict) -> None:
    output = result.get("output") or {}
    answer = str(output.get("answer") or "").strip()
    print(f"lễ tân> {answer}" if answer else "lễ tân> (không có câu trả lời)")
    actions = output.get("actions_taken")
    if isinstance(actions, list) and actions:
        print(f"  (đã làm: {', '.join(str(a) for a in actions)})")


def _chat_once(api_url: str, token: str, message: str, session_id: str | None,
               interval: float, timeout: float) -> str | None:
    if len(message) > MAX_MESSAGE_CHARS:
        raise _die(f"tin nhắn dài {len(message)} ký tự, tối đa {MAX_MESSAGE_CHARS}.")
    sent = _send_message(api_url, token, message, session_id)
    execution_id = str(sent["execution_id"])
    session_id = str(sent.get("session_id") or session_id or "")
    print(f"... đang chờ (execution {execution_id})", file=sys.stderr)
    result = _poll_result(api_url, token, execution_id, interval, timeout)
    _print_answer(result)
    return session_id or None


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Chat với lễ tân khách sạn qua guest API (/guest/chat).",
    )
    parser.add_argument("message", nargs="?", help="một tin nhắn gửi rồi thoát (không có = REPL)")
    parser.add_argument("--api-url", default=os.environ.get("GUEST_API_URL", DEFAULT_API_URL),
                        help="platform API base URL (mặc định %(default)s)")
    parser.add_argument("--user-url", default=os.environ.get("USER_SERVICE_URL", DEFAULT_USER_URL),
                        help="user service base URL cho --login (mặc định %(default)s)")
    parser.add_argument("--token", default=None, help="JWT guest (ưu tiên hơn GUEST_JWT)")
    parser.add_argument("--login", action="store_true", help="login user service để lấy JWT")
    parser.add_argument("--session", default=None, help="tiếp tục session_id có sẵn")
    parser.add_argument("--new", action="store_true", help="bỏ session cũ, bắt đầu hội thoại mới")
    parser.add_argument("--interval", type=float, default=DEFAULT_POLL_INTERVAL,
                        help="giây giữa 2 lần poll (mặc định %(default)s)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="chờ tối đa bao nhiêu giây (mặc định %(default)s)")
    parser.add_argument("--verbose", action="store_true", help="bật log HTTP của _api")
    args = parser.parse_args(argv)
    if not args.verbose:
        os.environ["MANUAL_LOG"] = "0"

    try:
        token = _resolve_token(args)
    except SystemExit as exc:
        return int(exc.code or 2)

    session_id = None if args.new else args.session
    try:
        if args.message is not None:
            _chat_once(args.api_url, token, args.message, session_id,
                       args.interval, args.timeout)
            return 0
        print("Chat với lễ tân (gõ /quit để thoát).")
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
                session_id = _chat_once(args.api_url, token, message, session_id,
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
