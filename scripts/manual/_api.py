"""Shared helper for the manual API test scripts in this directory.

Stdlib only. Loads the repo-root ``.env`` (existing environment
variables win) so ``API_KEY`` / ``API_BASE_URL`` work without exporting
them by hand, and wraps the two things every script needs: a JSON
POST/GET against the API and a poll-until-terminal loop for executions.

Imported by the sibling scripts, not run directly:

    from _api import call, submit, wait_for, get_result
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_dotenv(path: Path | None = None) -> Path | None:
    """Load KEY=VALUE lines from the nearest ``.env`` into os.environ.

    Same stdlib parser as ``scripts/smoke_e2e.py``: searches upward from
    this script, skips comments/blank lines, strips ``export `` and
    surrounding quotes, and never overwrites a variable already set.
    """
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
                line = line[len("export "):].strip()
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
        return dotenv
    return None


def api_base() -> str:
    return os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


def api_key() -> str:
    return os.environ.get("API_KEY", "")


def call(method: str, path: str, body: dict | None = None,
         timeout: float = 60) -> tuple[int, dict | None]:
    """JSON request against the API. Returns (status_code, payload).

    HTTP errors are returned, not raised, so a test can assert on 4xx
    validation responses. Connection errors still raise.
    """
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key():
        headers["X-API-Key"] = api_key()
    req = urllib.request.Request(api_base() + path, data=data, method=method,
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, (json.loads(raw) if raw else None)


def submit(mission_id: str, input_data: dict,
           session_id: str | None = None) -> dict:
    """POST /executions and return the ExecutionRead payload.

    ``session_id`` is optional (Option B, docs/sessions.md): executions
    that share it form one conversation and later runs see the prior
    turns. Omitted -> the API creates a fresh one-turn session.
    """
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
    while time.monotonic() - start < timeout:
        _, execution = call("GET", f"/executions/{execution_id}")
        if execution and execution["status"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return execution, time.monotonic() - start
        time.sleep(interval)
    _, execution = call("GET", f"/executions/{execution_id}")
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
    """Fail fast with a clear message when the API is not reachable."""
    try:
        status, _ = call("GET", "/health", timeout=10)
    except Exception as exc:  # noqa: BLE001 — connection refused, DNS, ...
        sys.exit(f"API not reachable at {api_base()}: {exc}")
    if status != 200:
        sys.exit(f"API /health -> HTTP {status} at {api_base()}")


# Load the repo-root .env once at import time so API_KEY is available to
# every call() without the caller exporting it by hand.
load_dotenv()
