"""On-demand live-agent check — runs the live missions through the REAL
mission graph with the REAL configured LLM, then logs to MLflow.

WHEN: only when tran asks for a check — after an agent changes (mission
YAML / skill / agent code). There is deliberately NO automatic trigger:
no cron, no hook, no CI wiring, no schedule anywhere. This script runs
only when a human invokes it (``make agent-check`` or the module path).

HOW: seeds missions + agents into an in-memory SQLite DB (the same
pattern as the local live-run script and ``tests/conftest.py``), builds
each mission dict with ``mission_service.mission_to_dict`` (the DB-row
shape the worker uses — never the raw seed YAML), and calls
``runtime.mission_graph.run_mission``. The LLM provider / model come
from the environment through the repo's normal factory (LiteLLM proxy
or Cloudflare, per ``LLM_PROVIDER`` / ``LLM_MODEL`` in ``.env``) —
nothing about the provider, model, account or token is hardcoded here.
The check never touches the live Postgres: ``DATABASE_URL`` is forced
to in-memory SQLite below, before any repo import.

WHAT: one fixed probe case per live mission (archived agents —
fault_investigator / hotel_receptionist — are NOT covered; mock
coverage is ``make eval``'s job, so there is no ``--mock`` mode here).

Verdict per case (see :func:`decide_verdict`):
- PASS — status SUCCEEDED and no known environmental limitation showed.
- LIMIT — SUCCEEDED or FAILED only because of a KNOWN environmental
  limitation (see :data:`KNOWN_LIMITATIONS`): missing Lark app
  credentials on this VM, or a missing Tavily key.
- FAIL — anything else.

Results: a Markdown table is printed first, then logged to MLflow
(experiment ``agent-check``: one parent run per check session, one
child run per case, ``report.md`` artifact on both). A logging failure
never masks the case results — the table is printed before logging,
and the exit code then reflects the verdicts, not the logging error.

Usage (repo root, project venv):
    .venv/bin/python scripts/manual/agent_check.py
    .venv/bin/python scripts/manual/agent_check.py --missions system_admin
    make agent-check
    make agent-check MISSIONS=system_admin TRACKING_URI=sqlite:////tmp/mlruns-agent-check/mlflow.db
    # (MLflow 3.x here rejects the file:// backend — use sqlite, see
    # docs/agent-check-pipeline.md.)

See docs/agent-check-pipeline.md for the full quy trình (WHEN / HOW /
WHAT / WHERE results live).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Env for the in-memory run — set BEFORE any repo import, mirroring
# tests/conftest.py. DATABASE_URL is forced (not setdefault): the check
# must never seed into, or read executions from, the live Postgres.
# LLM settings are deliberately NOT set here: provider / model / keys
# come from the caller's environment / .env via the normal factory.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ.setdefault("API_KEY", "")
os.environ["CELERY_BROKER_URL"] = "memory://"
os.environ["CELERY_RESULT_BACKEND"] = "cache+memory://"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["CHECKPOINT_ENABLED"] = "false"
os.environ["HARNESS_SOURCE"] = "seed"

EXPERIMENT = "agent-check"

# Fixed probe cases — one per live mission. The agent_analytis probe
# is a REAL short research question, not arithmetic: the skill map in
# skills/agent-analytis/SKILL.md routes to the research flow only when
# the turn "cần nghiên cứu: tìm nguồn mới, trích xuất, phân tích"; an
# arithmetic question matches neither branch and the run ends with
# "Turn ended without executing any flow".
CASES: dict[str, dict] = {
    "agent_lark_me_mission": {
        "name": "lark_ping",
        "input": {
            "message": "ping",
            "chat_id": "live-test-chat",
            "chat_type": "p2p",
            "sender_id": {"open_id": "ou_live_test"},
            "message_id": "om_live_test",
        },
    },
    "system_admin": {
        "name": "admin_list_missions",
        "input": {"request": "Liệt kê các mission hiện có trong hệ thống."},
    },
    "agent_analytis_mission": {
        "name": "analytis_gold_price",
        "input": {
            "message": "Giá vàng SJC hôm nay là bao nhiêu? "
            "Tìm nguồn mới nhất trên web và tóm tắt ngắn gọn."
        },
    },
}

# Known environmental limitations — LIMIT, not FAIL. Each substring is
# the exact error text the tool raises (not a paraphrase):
# - runtime/integrations/lark_client.py: LARK_APP_ID / LARK_APP_SECRET
#   are not set on this VM, so Lark tools report "not configured" and
#   the run still SUCCEEDS with that note.
# - runtime/tools/tavily_tools.py: TAVILY_API_KEY is not set, so the
#   research flow's search returns the missing-key error instead of
#   live results.
KNOWN_LIMITATIONS = (
    "LARK_APP_ID / LARK_APP_SECRET are not set",
    "TAVILY_API_KEY is not set",
)


def decide_verdict(status: str, text: str) -> str:
    """Pure verdict function: PASS / LIMIT / FAIL for one case.

    ``text`` is the run's combined errors + notes + answer text.
    A known-limitation substring anywhere in it means the outcome is
    explained by the environment, so the case is LIMIT whether the run
    SUCCEEDED (limitation noted) or FAILED (blocked by it). Otherwise
    SUCCEEDED is PASS and anything else is FAIL.
    """
    haystack = text or ""
    if any(marker in haystack for marker in KNOWN_LIMITATIONS):
        return "LIMIT"
    if (status or "").upper() == "SUCCEEDED":
        return "PASS"
    return "FAIL"


def _answer(state: dict) -> str:
    """Short answer text out of a run_mission final state."""
    result = state.get("result") or {}
    out = result.get("output") if isinstance(result, dict) else result
    answer = ""
    if isinstance(out, dict):
        answer = str(out.get("answer") or out.get("final_answer") or "")
    elif isinstance(out, str):
        answer = out
    if not answer and isinstance(result, dict):
        answer = str(result)
    return answer[:600]


def _haystack(state: dict) -> str:
    """Combined errors + notes + answer + result text, for LIMIT matching.

    The tool error dicts (e.g. the Lark / Tavily missing-key errors)
    may surface inside the result rather than in errors/notes, so the
    stringified result is included too (bounded).
    """
    parts = list(state.get("errors") or []) + list(state.get("notes") or [])
    parts.append(_answer(state))
    parts.append(str(state.get("result") or "")[:4000])
    return "\n".join(str(p) for p in parts)


def count_tool_calls(state: dict) -> int:
    """Tool calls counted from the run's plan_trace (per-case metric).

    Each skill trace lists ``tools_by_node`` (node -> tools called) plus
    ``deviations`` (tools outside the map); both count. When the run
    produced no plan_trace (mapless mission / no tools), fall back to
    the result's ``actions_taken`` list.
    """
    result = state.get("result") or {}
    plan_trace = result.get("plan_trace") or {}
    total = 0
    for trace in plan_trace.values():
        if not isinstance(trace, dict):
            continue
        tools_by_node = trace.get("tools_by_node") or {}
        total += sum(len(v or []) for v in tools_by_node.values())
        total += len(trace.get("deviations") or [])
    if total:
        return total
    actions = result.get("actions_taken") or []
    return len(actions)


def run_case(mission: dict, case: dict, execution_id: str) -> dict:
    """Run one probe case through the real mission graph; return its result."""
    from runtime.mission_graph import run_mission

    start = time.monotonic()
    try:
        state = run_mission(
            mission, case["input"], execution_id,
            session_id=f"agent-check-{mission['mission_id']}",
        )
        status = state.get("status", "FAILED")
        errors = list(state.get("errors") or [])[:5]
        notes = list(state.get("notes") or [])[:5]
        answer = _answer(state)
        tool_calls = count_tool_calls(state)
        haystack = _haystack(state)
    except Exception as exc:  # noqa: BLE001 — one case failing must not kill the rest
        status = "EXCEPTION"
        errors = [f"{type(exc).__name__}: {exc}"[:500]]
        notes, answer, tool_calls = [], "", 0
        haystack = errors[0]
    duration = round(time.monotonic() - start, 2)
    return {
        "mission": mission["mission_id"],
        "case": case["name"],
        "status": status,
        "verdict": decide_verdict(status, haystack),
        "success": 1 if status == "SUCCEEDED" else 0,
        "duration_seconds": duration,
        "tool_calls": tool_calls,
        "errors": errors,
        "notes": notes,
        "answer": answer,
    }


def build_report(results: list[dict], model: str) -> str:
    """Markdown report: one row per case (also the MLflow artifact)."""
    lines = [
        "# Agent check",
        "",
        f"Model: `{model}` — {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| mission | case | verdict | status | duration (s) | tool calls | errors / notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        detail = "; ".join(str(e) for e in (r["errors"] + r["notes"])[:3])
        detail = detail.replace("|", "/").replace("\n", " ")[:300]
        lines.append(
            f"| {r['mission']} | {r['case']} | {r['verdict']} | {r['status']} | "
            f"{r['duration_seconds']} | {r['tool_calls']} | {detail} |"
        )
    lines.append("")
    counts = {v: sum(1 for r in results if r["verdict"] == v) for v in ("PASS", "LIMIT", "FAIL")}
    lines.append(
        f"Totals: {counts['PASS']} PASS, {counts['LIMIT']} LIMIT, {counts['FAIL']} FAIL "
        f"of {len(results)} case(s)."
    )
    return "\n".join(lines) + "\n"


def log_to_mlflow(
    results: list[dict],
    tracking_uri: str,
    model: str,
    run_name: str | None = None,
    experiment: str = EXPERIMENT,
) -> str:
    """Log one check session to MLflow; returns the parent run id.

    Mirrors scripts/eval_mission.py's log_to_mlflow shape: experiment,
    params, metrics, report.md artifact — here as one parent run per
    session (run_name ``agent-check-<YYYYMMDD-HHMMSS>``) with one nested
    child run per case.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    run_name = run_name or datetime.now(timezone.utc).strftime("agent-check-%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.log_params(
            {
                "model": model,
                "missions": ",".join(r["mission"] for r in results),
                "cases": len(results),
            }
        )
        mlflow.log_metrics(
            {
                "passed": sum(1 for r in results if r["verdict"] == "PASS"),
                "limited": sum(1 for r in results if r["verdict"] == "LIMIT"),
                "failed": sum(1 for r in results if r["verdict"] == "FAIL"),
            }
        )
        mlflow.log_text(build_report(results, model), "report.md")
        for r in results:
            with mlflow.start_run(run_name=r["case"], nested=True):
                mlflow.log_params(
                    {"mission": r["mission"], "case": r["case"], "model": model}
                )
                mlflow.log_metrics(
                    {
                        "success": r["success"],
                        "duration_seconds": r["duration_seconds"],
                        "tool_calls": r["tool_calls"],
                    }
                )
                mlflow.log_text(
                    build_report([r], model),
                    "report.md",
                )
        return parent.info.run_id


def _print_table(results: list[dict]) -> None:
    print("mission | case | verdict | status | duration (s) | tool calls")
    for r in results:
        print(
            f"{r['mission']} | {r['case']} | {r['verdict']} | {r['status']} | "
            f"{r['duration_seconds']} | {r['tool_calls']}"
        )


def _model_name() -> str:
    try:
        from app.config import settings

        return settings.llm_model
    except Exception:  # noqa: BLE001 — config import must not block the check
        return os.environ.get("LLM_MODEL", "unknown")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="On-demand live-agent check (real LLM, MLflow experiment "
        f"'{EXPERIMENT}'). No cron/hook/CI — runs only when invoked."
    )
    parser.add_argument(
        "--missions",
        default=",".join(CASES),
        help="comma-separated missions to check (default: all live missions)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or http://localhost:5000)",
    )
    parser.add_argument(
        "--experiment",
        default=EXPERIMENT,
        help=f"MLflow experiment (default: {EXPERIMENT})",
    )
    args = parser.parse_args(argv)

    wanted = [m.strip() for m in args.missions.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in CASES]
    if unknown:
        print(f"ERROR: unknown mission(s): {unknown} (known: {sorted(CASES)})")
        return 2
    if not wanted:
        print("ERROR: no missions selected.")
        return 2

    print("Seeding missions + agents into in-memory SQLite ...", flush=True)
    from seeds.load_missions import main as seed_main

    failures = seed_main()
    if failures:
        print(f"Seed failures (continuing with what seeded): {failures}", flush=True)

    from app import models
    from app.core.db import SessionLocal
    from app.services import mission_service

    db = SessionLocal()
    model = _model_name()
    results: list[dict] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for mission_id in wanted:
        case = CASES[mission_id]
        row = db.get(models.Mission, mission_id)
        if row is None:
            results.append(
                {
                    "mission": mission_id,
                    "case": case["name"],
                    "status": "NOT_FOUND",
                    "verdict": "FAIL",
                    "success": 0,
                    "duration_seconds": 0.0,
                    "tool_calls": 0,
                    "errors": [f"mission {mission_id} not found after seed"],
                    "notes": [],
                    "answer": "",
                }
            )
            continue
        mission = mission_service.mission_to_dict(row)
        print(f"=== {mission_id} ({case['name']}) ===", flush=True)
        results.append(run_case(mission, case, f"agent-check-{mission_id}-{stamp}"))

    # Print the verdict table FIRST: an MLflow failure below must never
    # mask the case results.
    print()
    _print_table(results)
    print()
    print(build_report(results, model))

    if not results:
        print("ERROR: no case ran, nothing to log.")
        return 2
    try:
        parent_run_id = log_to_mlflow(results, args.tracking_uri, model,
                                      experiment=args.experiment)
    except Exception as exc:  # noqa: BLE001 — MLflow down, missing boto3, ...
        print(f"ERROR: could not log to MLflow at {args.tracking_uri}: {exc}")
        print("Case results above are unaffected (exit code still reflects verdicts).")
    else:
        print(f"MLflow parent run: {parent_run_id}")
        print(f"Tracking URI: {args.tracking_uri}  experiment: {args.experiment}")
    if any(r["verdict"] == "FAIL" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
