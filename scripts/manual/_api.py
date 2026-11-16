"""Shared helper for the manual API test scripts in this directory.

Stdlib only (JSON GET/POST). Loads the repo-root ``.env`` (existing
environment variables win) so ``API_KEY`` / ``API_BASE_URL`` work
without exporting them by hand, and wraps the two things every script
needs: a JSON POST/GET against the API and a poll-until-terminal loop
for executions.

HTTP logging (INPUT/OUTPUT) is on by default; set ``MANUAL_LOG=0`` or
pass ``--quiet`` in scripts that support it to disable.

Imported by the sibling scripts, not run directly:

    from _api import call, submit, wait_for, get_result, request
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def load_dotenv(path: Path | None = None) -> Path | None:
    """Load KEY=VALUE lines from the nearest ``.env`` into os.environ."""
    start = Path(path) if path is not None else Path(__file__).resolve().parent
    for directory in [start, *start.parents]:
        dotenv = directory / ".env"
        if not dotenv.is_file():
            continue
        for raw in dotenv.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
        return dotenv
    return None


_SENSITIVE_HEADER_NAMES = frozenset(
    {
        "authorization",
        "x-api-key",
    }
)
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "password",
        "token",
        "access_token",
        "api_key",
        "secret",
    }
)


def manual_log_enabled() -> bool:
    val = (os.environ.get("MANUAL_LOG") or "1").strip().lower()
    return val not in ("0", "false", "no", "off", "")


def log_max_chars() -> int:
    try:
        return max(0, int(os.environ.get("MANUAL_LOG_MAX_CHARS", "4000")))
    except (TypeError, ValueError):
        return 4000


def set_quiet(flag: bool) -> None:
    os.environ["MANUAL_LOG"] = "0" if flag else "1"


def _is_sensitive_field(key: str) -> bool:
    n = (key or "").strip().lower().replace("_", "-")
    return n in _SENSITIVE_HEADER_NAMES or key.lower() in _SENSITIVE_FIELD_NAMES


def redact_for_log(value: Any) -> Any:
    """Deep-copy structures and mask secrets for stdout logging."""
    if isinstance(value, dict):
        out: dict = {}
        for key, val in value.items():
            if _is_sensitive_field(str(key)):
                out[key] = "***"
            else:
                out[key] = redact_for_log(val)
        return out
    if isinstance(value, list):
        return [redact_for_log(item) for item in value]
    return value


def truncate_log_text(text: str, *, limit: int | None = None) -> str:
    limit = log_max_chars() if limit is None else limit
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"... ({len(text) - limit} more chars)"


def _format_json_for_log(obj: Any) -> str:
    if obj is None:
        return ""
    try:
        text = json.dumps(redact_for_log(obj), ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        text = str(obj)
    return truncate_log_text(text)


def api_base() -> str:
    return os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


def api_key() -> str:
    return os.environ.get("API_KEY", "")


def _default_headers(extra: dict | None) -> dict:
    hdrs = dict(extra or {})
    if "X-API-Key" not in hdrs and "Authorization" not in hdrs and api_key():
        hdrs["X-API-Key"] = api_key()
    return hdrs


def _resolve_url(path_or_url: str, base_url: str | None) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    base = (base_url or api_base()).rstrip("/")
    if not path_or_url.startswith("/"):
        return f"{base}/{path_or_url}"
    return base + path_or_url


def _log_request(method: str, url: str, headers: dict | None, body: Any) -> None:
    if not manual_log_enabled():
        return
    print(f">>> {method.upper()} {url}")
    if headers:
        print(_format_json_for_log({"headers": dict(headers)}))
    if body is not None:
        print(_format_json_for_log(body))


def _log_response(status: int, elapsed: float, payload: Any) -> None:
    if not manual_log_enabled():
        return
    print(f"<<< HTTP {status} ({elapsed:.3f}s)")
    if payload is not None:
        print(_format_json_for_log(payload))


def log_http_exchange(
    method: str,
    url: str,
    *,
    headers: dict | None,
    body: Any,
    status: int,
    elapsed: float,
    payload: Any,
) -> None:
    _log_request(method, url, headers, body)
    _log_response(status, elapsed, payload)


@dataclass
class ManualHttpResponse:
    """Minimal requests-like response for manual scripts."""

    status_code: int
    _text: str = ""
    _content: bytes = b""
    headers: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self._text

    @property
    def content(self) -> bytes:
        return self._content

    def json(self) -> Any:
        if not self._text.strip():
            return None
        return json.loads(self._text)


def request(
    method: str,
    path_or_url: str,
    *,
    headers: dict | None = None,
    base_url: str | None = None,
    json_body: dict | None = None,
    data: dict | None = None,
    files: dict | None = None,
    timeout: float = 60,
    log: bool | None = None,
) -> ManualHttpResponse:
    """HTTP call with logging. Uses ``requests`` only when ``files`` is set."""
    url = _resolve_url(path_or_url, base_url)
    hdrs = _default_headers(headers)
    do_log = manual_log_enabled() if log is None else log
    body_for_log: Any = json_body if json_body is not None else data
    if do_log:
        _log_request(method, url, hdrs, body_for_log)

    started = time.monotonic()
    if files is not None:
        import requests

        try:
            resp = requests.request(
                method,
                url,
                headers=hdrs,
                json=json_body,
                data=data,
                files=files,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            elapsed = time.monotonic() - started
            if do_log:
                print(f"<<< HTTP error ({elapsed:.3f}s) {exc}")
            raise
        elapsed = time.monotonic() - started
        try:
            payload = resp.json() if resp.content else None
        except (TypeError, ValueError):
            payload = resp.text if resp.text else None
        if do_log:
            _log_response(resp.status_code, elapsed, payload)
        return ManualHttpResponse(
            status_code=resp.status_code,
            _text=resp.text,
            _content=resp.content,
            headers=dict(resp.headers),
        )

    req_data: bytes | None = None
    req_headers = {"Content-Type": "application/json", **hdrs}
    if json_body is not None:
        req_data = json.dumps(json_body).encode()
    req = urllib.request.Request(
        url, data=req_data, method=method.upper(), headers=req_headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            elapsed = time.monotonic() - started
            text = raw.decode("utf-8", errors="replace") if raw else ""
            try:
                payload = json.loads(raw) if raw else None
            except (TypeError, ValueError):
                payload = text if text else None
            if do_log:
                _log_response(resp.status, elapsed, payload)
            return ManualHttpResponse(
                status_code=resp.status,
                _text=text,
                _content=raw,
                headers=dict(resp.headers),
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        elapsed = time.monotonic() - started
        text = raw.decode("utf-8", errors="replace") if raw else ""
        try:
            payload = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            payload = text if text else None
        if do_log:
            _log_response(exc.code, elapsed, payload)
        return ManualHttpResponse(
            status_code=exc.code,
            _text=text,
            _content=raw,
            headers=dict(exc.headers or {}),
        )


def call(
    method: str,
    path: str,
    body: dict | None = None,
    timeout: float = 60,
    *,
    headers: dict | None = None,
    base_url: str | None = None,
    log: bool | None = None,
) -> tuple[int, dict | None]:
    """JSON request against the API. Returns (status_code, payload)."""
    hdrs = _default_headers(headers)
    resp = request(
        method,
        path,
        headers=hdrs,
        base_url=base_url,
        json_body=body,
        timeout=timeout,
        log=log,
    )
    try:
        payload = resp.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        return resp.status_code, payload
    if resp.text.strip().startswith("{"):
        try:
            return resp.status_code, json.loads(resp.text)
        except (TypeError, ValueError):
            return resp.status_code, None
    return resp.status_code, payload if isinstance(payload, dict) else None


def submit(mission_id: str, input_data: dict,
           session_id: str | None = None) -> dict:
    body: dict = {"mission_id": mission_id, "input": input_data}
    if session_id is not None:
        body["session_id"] = session_id
    status, payload = call("POST", "/executions", body)
    if status != 202 or not payload:
        raise RuntimeError(f"POST /executions -> HTTP {status}: {payload}")
    return payload


def wait_for(execution_id: str, timeout: float = 300,
             interval: float = 2) -> tuple[dict, float]:
    """Poll GET /executions/{id} until terminal. Returns (execution, seconds)."""
    start = time.monotonic()
    last_status: str | None = None
    poll = 0
    terminal = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
    path = f"/executions/{execution_id}"
    url = _resolve_url(path, None)

    while time.monotonic() - start < timeout:
        t0 = time.monotonic()
        status, execution = call("GET", path, log=False)
        elapsed = time.monotonic() - t0
        current = execution.get("status") if isinstance(execution, dict) else None
        status_changed = current != last_status
        is_terminal = current in terminal
        if manual_log_enabled() and (poll == 0 or status_changed or is_terminal):
            log_http_exchange(
                "GET",
                url,
                headers=_default_headers(None),
                body=None,
                status=status,
                elapsed=elapsed,
                payload=execution,
            )
        if execution and is_terminal:
            return execution, time.monotonic() - start
        last_status = current
        poll += 1
        time.sleep(interval)

    t0 = time.monotonic()
    status, execution = call("GET", path, log=False)
    elapsed = time.monotonic() - t0
    if manual_log_enabled():
        log_http_exchange(
            "GET", url, headers=_default_headers(None), body=None,
            status=status, elapsed=elapsed, payload=execution,
        )
    raise TimeoutError(
        f"execution {execution_id} still {execution and execution.get('status')} "
        f"after {timeout}s"
    )


def get_result(execution_id: str) -> dict:
    status, payload = call("GET", f"/executions/{execution_id}/result")
    if status != 200 or payload is None:
        raise RuntimeError(f"GET result -> HTTP {status}: {payload}")
    return payload


def evidence_kinds(evidence: list) -> dict[str, int]:
    kinds: dict[str, int] = {}
    for item in evidence:
        kind = item.get("kind", "?") if isinstance(item, dict) else "?"
        kinds[kind] = kinds.get(kind, 0) + 1
    return kinds


def print_result(execution: dict, result: dict, answer_chars: int = 1200) -> None:
    output = result.get("output") or {}
    evidence = result.get("evidence") or output.get("evidence") or []
    print(f"  status={execution['status']} agent={execution.get('agent_id')} "
          f"v={execution.get('agent_version')} error={execution.get('error')}")
    print(f"  evidence={len(evidence)} kinds={evidence_kinds(evidence)}")
    for key in ("actions_taken", "root_causes", "recommendations"):
        if output.get(key) is not None:
            print(f"  {key}={json.dumps(output[key], ensure_ascii=False)[:800]}")
    answer = output.get("answer") or output.get("summary")
    if answer:
        print(f"  answer={json.dumps(answer, ensure_ascii=False)[:answer_chars]}")


def require_worker() -> None:
    try:
        status, _ = call("GET", "/health", timeout=10)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"API not reachable at {api_base()}: {exc}")
    if status != 200:
        sys.exit(f"API /health -> HTTP {status} at {api_base()}")


load_dotenv()