"""MLflow eval harness — fault_analysis on fixed test cases.

Mock mode (default): the graph runs with a fake chat model, so no LLM
key or proxy is needed. Real mode (--real) uses the LiteLLM factory.

Every run logs REAL measured values to the MLflow experiment
`agentic-runtime-eval`:
- params: mission, mode, model, llm (mock | litellm)
- metrics: success 0/1, evidence_count, error_evidence_count,
  validation_passed 0/1, duration_seconds
- artifact: the Markdown report (report.md)

View results: `mlflow ui` (or the compose mlflow service at
http://localhost:5000 when the observability profile is up).

Usage:
    python scripts/eval_mission.py [--mock-llm|--real] [--tracking-uri URI]
    make eval        # mock mode
    make eval-real   # real mode (needs the 4 LiteLLM env vars)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

EXPERIMENT = "agentic-runtime-eval"

TEST_CASES = [
    {
        "name": "o21_october_deterministic",
        "input": {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"},
        "mode": "deterministic",
    },
    {
        "name": "o21_october_autonomous",
        "input": {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"},
        "mode": "autonomous",
    },
    {
        "name": "o21_first_week_hybrid",
        "input": {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-07"},
        "mode": "hybrid",
    },
]

logger = logging.getLogger(__name__)


class _FakeAgent:
    """Stand-in for a compiled Deep Agent: returns a canned summary."""

    def invoke(self, state):  # noqa: ARG002 - prompt ignored on purpose
        return {
            "messages": [
                SimpleNamespace(
                    content=(
                        "MOCK SUMMARY: recurring door sensor faults (F-101) and "
                        "traction motor overheating (F-204) are the dominant "
                        "root causes for O21 in October 2026."
                    )
                )
            ]
        }


def install_mock_llm() -> None:
    """Patch the deep-agent factory so the graph runs without any LLM."""
    import runtime.deep_agent as deep_agent_module

    def _fake_build_deep_agent(**kwargs):  # noqa: ARG001
        return _FakeAgent()

    deep_agent_module.build_deep_agent = _fake_build_deep_agent


def load_fault_analysis_mission() -> dict:
    path = Path(__file__).resolve().parents[1] / "seeds" / "missions" / "fault_analysis.yaml"
    with path.open() as fh:
        return yaml.safe_load(fh)


def run_case(case: dict, mock_llm: bool, model_name: str) -> dict:
    """Run one test case through the Mission Graph; return measured metrics."""
    from runtime.mission_graph import run_mission
    from runtime.report import build_markdown_report

    mission = load_fault_analysis_mission()
    mission["execution_mode"] = case["mode"]
    execution_id = f"eval-{case['name']}"
    start = time.monotonic()
    final_state = run_mission(mission, case["input"], execution_id)
    duration = time.monotonic() - start

    evidence = final_state.get("evidence") or []
    error_evidence = [e for e in evidence if e.get("kind") == "error"]
    status = final_state.get("status", "FAILED")
    success = 1 if status == "SUCCEEDED" else 0
    # Validation only counts when the run produced real evidence (no error
    # items) — a run that only reports store failures is not a pass.
    validation_passed = 1 if (success and evidence and not error_evidence) else 0
    report = build_markdown_report(execution_id, mission, case["input"], final_state)
    return {
        "name": case["name"],
        "status": status,
        "success": success,
        "evidence_count": len(evidence),
        "error_evidence_count": len(error_evidence),
        "validation_passed": validation_passed,
        "duration_seconds": round(duration, 2),
        "report": report,
        "model": model_name,
        "llm": "mock" if mock_llm else "litellm",
        "mode": case["mode"],
    }


def log_to_mlflow(result: dict, tracking_uri: str) -> str:
    """Log one case result to MLflow. Returns the run id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=result["name"]) as run:
        mlflow.log_params(
            {
                "mission": "fault_analysis",
                "mode": result["mode"],
                "model": result["model"],
                "llm": result["llm"],
            }
        )
        mlflow.log_metrics(
            {
                "success": result["success"],
                "evidence_count": result["evidence_count"],
                "error_evidence_count": result["error_evidence_count"],
                "validation_passed": result["validation_passed"],
                "duration_seconds": result["duration_seconds"],
            }
        )
        mlflow.log_text(result["report"], "report.md")
        return run.info.run_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fault_analysis and log to MLflow.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mock-llm", action="store_true", default=True,
                       help="run with a fake model (default, no LLM needed)")
    group.add_argument("--real", action="store_true",
                       help="run with the LiteLLM factory (needs the 4 LiteLLM env vars)")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or http://localhost:5000)")
    parser.add_argument("--case", dest="cases", action="append", default=None,
                        help="run only this case name (repeatable)")
    args = parser.parse_args()

    mock_llm = not args.real
    if mock_llm:
        install_mock_llm()
        model_name = "mock"
    else:
        from app.config import settings

        model_name = settings.llm_model

    cases = TEST_CASES
    if args.cases:
        cases = [c for c in TEST_CASES if c["name"] in args.cases]
        if not cases:
            print(f"ERROR: unknown case(s): {args.cases}")
            return 2

    print(f"Eval: {len(cases)} case(s), llm={'mock' if mock_llm else 'litellm'}, "
          f"tracking={args.tracking_uri}")
    exit_code = 0
    for case in cases:
        result = run_case(case, mock_llm=mock_llm, model_name=model_name)
        try:
            run_id = log_to_mlflow(result, args.tracking_uri)
        except Exception as exc:  # noqa: BLE001 — MLflow server down, etc.
            print(f"ERROR: could not log to MLflow at {args.tracking_uri}: {exc}")
            print("Start it with `docker compose --profile observability up -d mlflow` "
                  "or `mlflow ui`, or pass --tracking-uri sqlite:///tmp/mlflow-test/mlflow.db")
            return 3
        print(
            f"  {result['name']}: status={result['status']} "
            f"success={result['success']} evidence={result['evidence_count']} "
            f"error_evidence={result['error_evidence_count']} "
            f"validation={result['validation_passed']} "
            f"duration={result['duration_seconds']}s run_id={run_id}"
        )
        if not result["success"]:
            exit_code = 1
    print("Eval done. View results in MLflow (experiment "
          f"'{EXPERIMENT}').")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
