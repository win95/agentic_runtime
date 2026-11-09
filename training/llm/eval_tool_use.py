"""Tool-use eval (track C).

Scores predicted tool calls against ToolUseExample cases:
- correct-tool rate: predicted tool == expected tool.
- correct-args rate: predicted args == expected args (only counted when
  the tool is also correct).
- out-of-scope rate: predicted tool is not in available_tools.
- trap pass rate: cases with expected_tool=None where nothing was called.

Predictions come from --predictions (JSONL: {"id", "predicted_tool",
"predicted_args"}) or from --mock-predictions, a deterministic mock
predictor (correct except every 5th case: wrong tool; every 7th case:
wrong args) that exists only to exercise the metric pipeline offline
on synthetic data. Mock results are labeled as such and mean nothing
about a real model.

Usage:
    python -m training.llm.eval_tool_use --data <cases.jsonl> --mock-predictions
    python -m training.llm.eval_tool_use --data <cases.jsonl> --predictions <preds.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _mock_predict(case, index: int) -> tuple[str | None, dict]:
    """Deterministic mock predictor (pipeline exercise only)."""
    if (index + 1) % 5 == 0:
        # Wrong tool: pick the first allowed tool that is not expected.
        wrong = next((t for t in case.available_tools if t != case.expected_tool), None)
        return wrong, dict(case.expected_args)
    if (index + 1) % 7 == 0:
        # Right tool, wrong args.
        return case.expected_tool, {"asset_id": "WRONG"}
    return case.expected_tool, dict(case.expected_args)


def _load_predictions(path: str) -> dict[str, dict]:
    preds: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        preds[row["id"]] = {
            "predicted_tool": row.get("predicted_tool"),
            "predicted_args": row.get("predicted_args") or {},
        }
    return preds


def evaluate(cases, predictions: dict[str, dict]) -> dict:
    """Compute tool-use metrics. ``predictions`` maps case id -> prediction."""
    n = len(cases)
    correct_tool = 0
    correct_args = 0
    out_of_scope = 0
    trap_total = 0
    trap_passed = 0
    for case in cases:
        pred = predictions.get(case.id, {"predicted_tool": None, "predicted_args": {}})
        predicted_tool = pred.get("predicted_tool")
        predicted_args = pred.get("predicted_args") or {}
        if predicted_tool is not None and predicted_tool not in case.available_tools:
            out_of_scope += 1
        if case.expected_tool is None:
            trap_total += 1
            if predicted_tool is None:
                trap_passed += 1
            continue
        if predicted_tool == case.expected_tool:
            correct_tool += 1
            if predicted_args == case.expected_args:
                correct_args += 1
    denom = max(n, 1)
    return {
        "n_cases": n,
        "correct_tool_rate": round(correct_tool / denom, 4),
        "correct_args_rate": round(correct_args / denom, 4),
        "out_of_scope_rate": round(out_of_scope / denom, 4),
        "trap_pass_rate": round(trap_passed / max(trap_total, 1), 4) if trap_total else None,
        "n_traps": trap_total,
        "mock": False,
    }


def _log_mlflow(metrics: dict, tracking_uri: str) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("agentic-runtime-finetune")
        with mlflow.start_run(run_name="tool-use-eval"):
            mlflow.log_params({"track": "tool-use"})
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )
    except Exception as exc:  # noqa: BLE001 — observability must never fail eval
        print(f"WARNING: could not log to MLflow ({exc}).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tool-use eval (track C).")
    parser.add_argument("--data", required=True, help="ToolUseExample JSONL.")
    parser.add_argument("--predictions", default=None, help="Predictions JSONL.")
    parser.add_argument("--mock-predictions", action="store_true",
                        help="Use the deterministic mock predictor.")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args(argv)

    if bool(args.predictions) == bool(args.mock_predictions):
        parser.error("pass exactly one of --predictions or --mock-predictions")

    from training.common.schemas import ToolUseExample, validate_jsonl

    cases, errors = validate_jsonl(args.data, ToolUseExample)
    if errors:
        print(f"ERROR: {len(errors)} invalid line(s) in {args.data}; fix the dataset first.")
        return 2
    if not cases:
        print(f"ERROR: no valid cases in {args.data}.")
        return 2

    if args.mock_predictions:
        predictions = {
            case.id: {
                "predicted_tool": tool,
                "predicted_args": pred_args,
            }
            for i, case in enumerate(cases)
            for tool, pred_args in [_mock_predict(case, i)]
        }
        print("NOTE: using mock predictions — metrics exercise the pipeline only.")
    else:
        predictions = _load_predictions(args.predictions)

    metrics = evaluate(cases, predictions)
    metrics["mock"] = bool(args.mock_predictions)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if metrics["out_of_scope_rate"] > 0:
        print("GATE: out-of-scope calls present — this model version must NOT deploy.")
    if not args.no_mlflow:
        _log_mlflow(metrics, args.tracking_uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
