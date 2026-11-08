"""Smoke test end-to-end (stdlib + httpx only).

Checks the 4 required LiteLLM env vars, POSTs a fault_analysis execution
for asset O21 (Oct 2026) to the API, polls until a terminal status, then
prints status, root causes, evidence count and report location.

Usage:
    python scripts/smoke_e2e.py [--api-base http://localhost:8000]
    make demo

Exits non-zero on missing env, API error, failed execution or timeout.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

REQUIRED_ENV = ["LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL", "EMBEDDING_MODEL"]
POLL_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 5


def check_env() -> list[str]:
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the agentic runtime end-to-end.")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("API_BASE", "http://localhost:8000"),
        help="API base URL (default: $API_BASE or http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=POLL_TIMEOUT_SECONDS,
        help=f"Max seconds to wait for completion (default {POLL_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args()

    missing = check_env()
    if missing:
        print(f"ERROR: missing required env vars: {', '.join(missing)}")
        print("Copy .env.example to .env and fill LLM_API_BASE, LLM_API_KEY, "
              "LLM_MODEL, EMBEDDING_MODEL.")
        return 2

    payload = {
        "mission_id": "fault_analysis",
        "input": {
            "asset_id": "O21",
            "start_date": "2026-10-01",
            "end_date": "2026-10-31",
        },
    }
    headers = {}
    if os.environ.get("API_KEY"):
        headers["X-API-Key"] = os.environ["API_KEY"]
    try:
        with httpx.Client(base_url=args.api_base, timeout=30, headers=headers) as client:
            response = client.post("/executions", json=payload)
            response.raise_for_status()
            created = response.json()
            execution_id = created["execution_id"]
            print(f"Execution created: {execution_id} (status {created.get('status')})")

            deadline = time.monotonic() + args.timeout
            final = None
            while time.monotonic() < deadline:
                time.sleep(POLL_INTERVAL_SECONDS)
                response = client.get(f"/executions/{execution_id}")
                response.raise_for_status()
                final = response.json()
                status = final.get("status")
                print(f"  ... {status}")
                if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
                    break
            else:
                print(f"ERROR: execution did not finish within {args.timeout}s")
                return 3

            result = {}
            if final.get("status") == "SUCCEEDED":
                response = client.get(f"/executions/{execution_id}/result")
                if response.status_code == 200:
                    result = response.json()
    except httpx.HTTPError as exc:
        print(f"ERROR: API request failed: {exc}")
        print("Is the API running? `docker compose up -d` then retry.")
        return 4

    status = final.get("status")
    output = result.get("output") or {}
    evidence = result.get("evidence") or []
    print(f"\nStatus: {status}")
    print(f"Root causes: {output.get('root_causes') or []}")
    print(f"Evidence items: {len(evidence)}")
    print(f"Report location: {output.get('report_location') or output.get('report_storage')}")
    if status != "SUCCEEDED":
        print(f"ERROR: execution ended with status {status}: {final.get('error')}")
        return 5
    print("Smoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
