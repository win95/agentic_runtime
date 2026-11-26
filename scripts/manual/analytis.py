"""Manual scenario suite for ``agent_analytis_mission`` (agent-analytis).

Runs canned scenarios through the live API stack and prints PASS/FAIL per
check. Default logging uses ``_api`` (``=== Q`` / ``=== A``, ``>>>`` /
``<<<``); use ``--quiet`` to disable HTTP logs.

Usage (repo root, API + worker + Tavily + sandbox up)::

    .venv/bin/python -m scripts.manual.analytis
    .venv/bin/python -m scripts.manual.analytis --scenario research_report
    .venv/bin/python -m scripts.manual.analytis --quiet
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _api import api_base, load_dotenv, require_worker, set_quiet  # noqa: E402
from agent_config_check import Report  # noqa: E402
from analytis_common_1 import (  # noqa: E402
    DEFAULT_API_URL,
    DEFAULT_OUT_DIR,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_TIMEOUT,
    SELF_TEST_QUESTION,
    die,
    export_checks,
    resolve_api_key,
    run_once,
)

SCENARIOS = ("research_report",)
PREFIX = "analytis"


def _env_flag(name: str) -> bool:
    val = (os.environ.get(name) or "").strip().lower()
    return val in ("1", "true", "yes", "on")


def preflight(report: Report, api_url: str) -> int | None:
    """Hard failures return exit code 2; soft issues are WARN only."""
    require_worker()
    try:
        resolve_api_key()
    except SystemExit as exc:
        return int(exc.code or 2)

    print(f"API at {api_url or api_base()}")
    if not os.environ.get("TAVILY_API_KEY", "").strip():
        report.warn("TAVILY_API_KEY thiếu", "tavily_search/extract có thể không chạy")
    if not _env_flag("SANDBOX_ENABLED"):
        report.warn("SANDBOX_ENABLED không bật", "in-memory backend — files thường rỗng")
    return None


def scenario_research_report(
    report: Report,
    api_url: str,
    out_dir: Path,
    interval: float,
    timeout: float,
    session_id: str | None,
) -> None:
    sid = session_id or f"sess-analytis-{uuid.uuid4().hex[:12]}"
    print(f"\n=== scenario research_report ===")
    print(f"session: {sid}")
    print(f"hỏi: {SELF_TEST_QUESTION}")
    try:
        _execution, result, _saved = run_once(
            api_url,
            SELF_TEST_QUESTION,
            sid,
            out_dir,
            interval,
            timeout,
            scenario="research_report",
            prefix=PREFIX,
        )
        for name, ok in export_checks(result, succeeded=True):
            report.check(ok, name)
    except SystemExit:
        report.check(False, "run hoàn thành")
        for name, ok in export_checks({"output": {}}, succeeded=False):
            if name == "run SUCCEEDED":
                continue
            report.check(ok, name)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Kịch bản manual agent-analytis (PASS/FAIL qua platform API).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIOS,
        metavar="NAME",
        help=f"kịch bản chạy (mặc định: {', '.join(SCENARIOS)})",
    )
    parser.add_argument("--api-url", default=os.environ.get("API_BASE_URL", DEFAULT_API_URL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--session", default=None, help="session_id cố định (mặc định: random)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--quiet", action="store_true", help="tắt log HTTP (_api)")
    args = parser.parse_args(argv)

    if args.quiet:
        set_quiet(True)
    if args.api_url:
        os.environ["API_BASE_URL"] = args.api_url

    report = Report()
    try:
        code = preflight(report, args.api_url)
        if code is not None:
            return code
    except SystemExit as exc:
        return int(exc.code or 2)

    selected = args.scenario or list(SCENARIOS)
    for name in selected:
        if name == "research_report":
            scenario_research_report(
                report, args.api_url, Path(args.out_dir), args.interval, args.timeout, args.session
            )

    failed = len(report.failures)
    total = report.passed + failed
    print(
        f"\n{'PASS' if failed == 0 else 'FAIL'} — "
        f"{report.passed} passed, {failed} failed"
        + (f", {len(report.warnings)} warnings" if report.warnings else "")
    )
    for w in report.warnings:
        print(f"  [WARN] {w}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())