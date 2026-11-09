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
    python scripts/eval_mission.py [--mock-llm|--real] [--genai] [--tracking-uri URI]
    make eval        # mock mode
    make eval-real   # real mode (needs the 4 LiteLLM env vars)
    make eval-genai  # + mlflow.genai.evaluate with a judge via the LiteLLM proxy
"""

from __future__ import annotations

import argparse
import copy
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


JUDGE_NAME = "evidence_grounding"


def _proxy_base_url() -> str:
    """OpenAI-compatible base URL of the LiteLLM proxy (with /v1)."""
    from app.config import settings

    base = settings.llm_api_base.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def run_genai_eval(cases: list[dict], mock_llm: bool, tracking_uri: str) -> int | None:
    """Run mlflow.genai.evaluate with a judge routed through the LiteLLM proxy.

    Returns 0 on success, 1 on failure, or None when skipped because the
    judge cannot be routed (LiteLLM proxy not configured) — the classic
    eval results are unaffected in that case.
    """
    from app.config import settings

    if not settings.llm_api_key or not settings.llm_api_base:
        print("SKIPPED: genai eval needs judge routing through the LiteLLM proxy "
              "(LLM_API_BASE/LLM_API_KEY not configured); classic eval results "
              "above are unaffected.")
        return None
    # The judge's OpenAI-compatible client needs the proxy key + base URL.
    # No cost estimation is done here (see docs: LiteLLM price-table lookup
    # is deliberately skipped — the user owns the LiteLLM/vLLM setup).
    proxy_base = _proxy_base_url()
    os.environ.setdefault("OPENAI_API_KEY", settings.llm_api_key)
    os.environ.setdefault("OPENAI_API_BASE", proxy_base)
    try:
        import mlflow
        from mlflow.genai import evaluate, make_judge

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT)
        judge = make_judge(
            name=JUDGE_NAME,
            instructions=(
                "You judge whether the agent's root-cause summary is grounded "
                "in the evidence it collected. The mission input is {{ inputs }} "
                "and the agent's output summary is {{ outputs }}. Answer 'yes' "
                "when every root cause stated in the summary is supported by "
                "the evidence described, otherwise answer 'no'."
            ),
            model=f"openai:/{settings.llm_model}",
            base_url=proxy_base,
        )

        def predict_fn(asset_id: str, start_date: str, end_date: str, mode: str = "hybrid") -> dict:
            from runtime.mission_graph import run_mission

            mission = load_fault_analysis_mission()
            mission = copy.deepcopy(mission)
            mission.setdefault("execution", {})["mode"] = mode
            final_state = run_mission(
                mission,
                {"asset_id": asset_id, "start_date": start_date, "end_date": end_date},
                f"eval-genai-{asset_id}-{mode}",
            )
            result = final_state.get("result") or {}
            causes = result.get("root_causes") or []
            return {"summary": "; ".join(str(c) for c in causes)}

        data = [{"inputs": {**case["input"], "mode": case["mode"]}} for case in cases]
        result = evaluate(data=data, scorers=[judge], predict_fn=predict_fn)
        print(f"GenAI eval done. metrics={getattr(result, 'metrics', None)}")
        return 0
    except Exception as exc:  # noqa: BLE001 — proxy down, judge model missing, ...
        print(f"ERROR: genai eval failed: {exc}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fault_analysis and log to MLflow.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mock-llm", action="store_true", default=True,
                       help="run with a fake model (default, no LLM needed)")
    group.add_argument("--real", action="store_true",
                       help="run with the LiteLLM factory (needs the 4 LiteLLM env vars)")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or http://localhost:5000)")
    parser.add_argument("--genai", action="store_true",
                        help="also run mlflow.genai.evaluate with a judge routed through "
                             "the LiteLLM proxy (needs LLM_API_BASE/LLM_API_KEY)")
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
            if isinstance(exc, ModuleNotFoundError) and exc.name == "boto3":
                # The server is reachable, but uploading the report artifact
                # to its S3 (MinIO) artifact root needs boto3 on this machine.
                print("The MLflow server is up, but this machine is missing boto3,")
                print("which the MLflow client needs to upload artifacts to the")
                print("server's S3 (MinIO) artifact root.")
                print("Fix: pip install boto3  (or: pip install -r requirements.txt)")
                print("Then set MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 and")
                print("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY to the MinIO")
                print("credentials (see .env.example).")
            else:
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
    if args.genai:
        genai_code = run_genai_eval(cases, mock_llm=mock_llm, tracking_uri=args.tracking_uri)
        if genai_code is not None and genai_code != 0:
            exit_code = genai_code
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
