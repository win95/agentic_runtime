"""Collection hook: execution results -> training examples.

Reads execution records in the shape produced by
``scripts/eval_mission.py::run_case`` (name/status/success/
evidence_count/error_evidence_count/validation_passed/duration_seconds/
report/model/llm/mode), optionally with ``input`` (asset_id/start_date/
end_date) and ``tool_calls`` (list of {name, arguments}) attached.

Rules (deliberately conservative — this hook must never invent data):
- Only runs with ``success == 1`` AND ``validation_passed == 1`` become
  SFT examples. Failed runs and runs with error evidence are skipped and
  counted, not converted.
- ToolUse examples are emitted ONLY when the record carries explicit
  ``tool_calls``. The current eval output does not record tool calls, so
  today this yields zero ToolUse examples; once real executions run
  through MLflow traces, the trace's tool calls feed this path.
- The assistant turn reuses the execution's Markdown report (grounded
  output), truncated to keep examples bounded.

Usage:
    python -m training.common.collect_from_traces --in results.jsonl --out-dir training/data
    python -m training.common.collect_from_traces --demo   # 3 synthetic records
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.common.schemas import (
    ChatMessage,
    SFTExample,
    ToolUseExample,
)

MAX_REPORT_CHARS = 4000

SYSTEM_PROMPT = (
    "Bạn là fault_investigator. Phân tích lỗi thang máy và chỉ kết luận "
    "dựa trên evidence thu thập được."
)

# Tools the fault_analysis mission is allowed to call (must match
# runtime/tools/registry.py registrations).
ALLOWED_TOOLS = [
    "search_fault_history",
    "get_maintenance_records",
    "search_technical_documents",
    "query_asset_relationships",
    "get_sensor_data",
    "calculate_failure_rate",
]


def _request_text(record: dict) -> str:
    """Reconstruct the mission request from the record's input/case name."""
    inp = record.get("input") or {}
    asset = inp.get("asset_id")
    start = inp.get("start_date")
    end = inp.get("end_date")
    if asset and start and end:
        return (
            f"Phân tích nguyên nhân các lỗi của thang máy {asset} "
            f"từ {start} đến {end}."
        )
    return f"Phân tích nguyên nhân các lỗi ({record.get('name', 'unknown')})."


def result_to_sft(record: dict, index: int) -> SFTExample | None:
    """Convert one successful, validated execution into an SFT example.

    Returns None (and the caller counts a skip) when the run did not
    succeed or its validation did not pass.
    """
    if record.get("success") != 1 or record.get("validation_passed") != 1:
        return None
    report = (record.get("report") or "").strip()
    if not report:
        return None
    return SFTExample(
        id=f"trace-sft-{index:03d}",
        mission_id="fault_analysis",
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=_request_text(record)),
            ChatMessage(
                role="assistant",
                content=report[:MAX_REPORT_CHARS],
            ),
        ],
        source="trace",
        split="train",
    )


def result_to_tool_use(record: dict, index: int) -> list[ToolUseExample]:
    """Convert a record's explicit tool calls into ToolUse examples.

    Only records that carry ``tool_calls`` produce examples. No tool
    calls are inferred from evidence or from the report text.
    """
    examples: list[ToolUseExample] = []
    for j, call in enumerate(record.get("tool_calls") or []):
        name = call.get("name")
        if name not in ALLOWED_TOOLS:
            continue
        examples.append(
            ToolUseExample(
                id=f"trace-tool-{index:03d}-{j + 1:02d}",
                mission_id="fault_analysis",
                context=_request_text(record),
                available_tools=list(ALLOWED_TOOLS),
                expected_tool=name,
                expected_args=dict(call.get("arguments") or {}),
                source="trace",
                split="train",
            )
        )
    return examples


def collect(records: list[dict]) -> tuple[list[SFTExample], list[ToolUseExample], dict]:
    """Convert records; returns (sft, tool_use, stats)."""
    sft: list[SFTExample] = []
    tool_use: list[ToolUseExample] = []
    stats = {"records": len(records), "sft": 0, "tool_use": 0, "skipped": 0}
    for i, record in enumerate(records, start=1):
        example = result_to_sft(record, i)
        if example is None:
            stats["skipped"] += 1
            continue
        sft.append(example)
        tool_use.extend(result_to_tool_use(record, i))
    stats["sft"] = len(sft)
    stats["tool_use"] = len(tool_use)
    return sft, tool_use, stats


def _demo_records() -> list[dict]:
    """3 clearly-synthetic records exercising every branch."""
    return [
        {
            "name": "o21_october_deterministic",
            "status": "SUCCEEDED",
            "success": 1,
            "evidence_count": 4,
            "error_evidence_count": 0,
            "validation_passed": 1,
            "duration_seconds": 12.5,
            "report": "# Báo cáo O21\n\nNguyên nhân chính: cảm biến cửa F-101.",
            "model": "mock",
            "llm": "mock",
            "mode": "deterministic",
            "input": {
                "asset_id": "O21",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
            },
            "tool_calls": [
                {
                    "name": "search_fault_history",
                    "arguments": {
                        "asset_id": "O21",
                        "start_date": "2026-10-01",
                        "end_date": "2026-10-31",
                    },
                }
            ],
        },
        {
            "name": "o22_failed_run",
            "status": "FAILED",
            "success": 0,
            "evidence_count": 0,
            "error_evidence_count": 1,
            "validation_passed": 0,
            "duration_seconds": 3.1,
            "report": "",
            "model": "mock",
            "llm": "mock",
            "mode": "deterministic",
            "input": {
                "asset_id": "O22",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
            },
        },
        {
            "name": "o23_no_tool_calls_recorded",
            "status": "SUCCEEDED",
            "success": 1,
            "evidence_count": 2,
            "error_evidence_count": 0,
            "validation_passed": 1,
            "duration_seconds": 8.0,
            "report": "# Báo cáo O23\n\nNguyên nhân chính: mòn cáp F-310.",
            "model": "mock",
            "llm": "mock",
            "mode": "hybrid",
            "input": {
                "asset_id": "O23",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
            },
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect training examples from execution records."
    )
    parser.add_argument("--in", dest="input", default=None, help="Input JSONL of result dicts.")
    parser.add_argument("--out-dir", default="training/data", help="Output directory.")
    parser.add_argument("--demo", action="store_true", help="Run on 3 synthetic records.")
    args = parser.parse_args()

    if args.demo:
        records = _demo_records()
    elif args.input:
        text = Path(args.input).read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parser.error("pass --demo or --in <results.jsonl>")

    sft, tool_use, stats = collect(records)
    out_dir = Path(args.out_dir)
    from training.common.schemas import write_jsonl

    n_sft = write_jsonl(out_dir / "collected_sft.jsonl", sft) if sft else 0
    n_tool = write_jsonl(out_dir / "collected_tool_use.jsonl", tool_use) if tool_use else 0
    print(f"records={stats['records']} sft={n_sft} tool_use={n_tool} skipped={stats['skipped']}")
    if not tool_use:
        print("NOTE: no ToolUse examples — records carry no explicit tool_calls yet.")
        print("Tool-call supervision needs MLflow trace data (see docs/fine-tuning-lifecycle.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
