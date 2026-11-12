"""Build a repeatable MLflow report for the manual hotel-reception runs.

This is the analysis pipeline behind ``hotel_reception_report.md``: it
reads what the manual runs already left in MLflow and re-derives every
number, so the report can be regenerated after any test run.

Pipeline stages (each is a function below):

1. **discover** — resolve the experiment, list runs and traces.
2. **join** — keep runs/traces whose ``session_id`` starts with the
   configured prefix (``sess-hotel-``), link a trace to its run via
   ``mlflow.sourceRun``, and de-duplicate.
3. **classify** — per trace: scenario, turn index in the session,
   status/version/drift, latency, tokens, tool calls (contract vs
   built-in), hook failures, derailed prose, prompt style.
4. **aggregate** — per-scenario table, latency, tokens, tool usage,
   file-tool noise, fan-out, booking quote/confirm, memory recall.
5. **render** — Markdown report + one-row-per-trace CSV.

The trace is the source of truth because the manual ``--cleanup`` only
deletes Postgres rows; MLflow runs and traces are kept.

Usage (from the repo root, stack and MLflow running):

    .venv/bin/python scripts/manual/mlflow_report.py
    .venv/bin/python scripts/manual/mlflow_report.py --since 2026-11-12
    .venv/bin/python scripts/manual/mlflow_report.py --out "" --csv ""
    .venv/bin/python scripts/manual/mlflow_report.py \
        --experiment agentic-runtime --session-prefix sess-hotel- \
        --out reports/hotel-mlflow-report.md \
        --csv reports/hotel-mlflow-traces.csv

Defaults: experiment ``agentic-runtime``, session prefix ``sess-hotel-``,
contract tools = the hotel agent's 14 MCP tools (keep in sync with
``seeds/agents/hotel_receptionist.yaml``), outputs under ``reports/``.

Only reads MLflow; it never starts executions, never touches the mock
MCP server, and does not need the API/worker.

Caveat on duplicated tool spans/audit rows: deepagents instruments a
tool twice (``ToolNode`` span → registered-tool wrapper → inner tool),
so one MCP call shows two nested spans with the same ``tool_call_id``
and two audit rows. The report counts these as *nested instrumentation*,
not as two executions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _api import load_dotenv  # noqa: E402

# The hotel agent's MCP contract. Kept literal so the report works even
# when a seed/registry is broken; override with --contract-tools.
HOTEL_TOOLS = (
    "check_room_availability",
    "get_room_rates",
    "find_guest",
    "register_guest",
    "create_booking",
    "get_booking",
    "cancel_booking",
    "change_booking_dates",
    "list_staff",
    "create_task",
    "assign_task",
    "get_task",
    "search_policy_documents",
    "read_policy_document",
)

# deepagents built-ins that run outside the MCP registry gate. They are
# the main source of noise with the local 3B model.
BUILTIN_TOOLS = (
    "read_file",
    "write_file",
    "edit_file",
    "ls",
    "glob",
    "grep",
    "task",
)

WRITE_TOOLS = {"create_booking", "cancel_booking", "change_booking_dates"}
QUOTE_TOOLS = {"check_room_availability", "get_room_rates"}
DERAIL_MARKERS = (
    "tool_error_feedback",
    "does not exist",
    "deviation:",
    "subagent",
    "plan nodes visited:",
)
TOOL_NAMES = re.compile(r"\"name\":\s*\"([a-z_]+)\"")


# --- small helpers -----------------------------------------------------------


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def _median(values) -> float | None:
    values = [v for v in values if v is not None and not pd.isna(v)]
    return round(statistics.median(values), 1) if values else None


def _quantile(values, q: float) -> float | None:
    values = [v for v in values if v is not None and not pd.isna(v)]
    return round(float(pd.Series(values).quantile(q)), 1) if values else None


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


# --- stage 1: discover -------------------------------------------------------


def resolve_experiment(experiment: str) -> tuple[str, str]:
    import mlflow

    if experiment.isdigit():
        exp = mlflow.get_experiment(experiment)
    else:
        exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"MLflow experiment not found: {experiment!r}")
    return exp.experiment_id, exp.name


def load_runs(experiment_id: str, prefix: str, agent: str | None,
              since: datetime | None, max_results: int) -> pd.DataFrame:
    import mlflow

    runs = mlflow.search_runs(experiment_ids=[experiment_id],
                              max_results=max_results)
    if runs.empty:
        return runs
    runs = runs.copy()
    runs["session"] = runs.get("tags.session_id", "").fillna("")
    runs = runs[runs["session"].str.startswith(prefix)]
    if agent:
        runs = runs[runs.get("tags.agent_id", "").fillna("") == agent]
    runs["start"] = pd.to_datetime(runs["start_time"], utc=True)
    if since is not None:
        runs = runs[runs["start"] >= since]
    return runs


def load_traces(experiment_id: str, max_results: int) -> pd.DataFrame:
    import mlflow

    return mlflow.search_traces(
        locations=[experiment_id],
        max_results=max_results,
        order_by=["timestamp ASC"],
    )


# --- stage 2 + 3: join and classify -----------------------------------------


def collect(runs: pd.DataFrame, traces: pd.DataFrame, *, prefix: str,
            contract: set[str], builtin: set[str],
            memory_regex: re.Pattern) -> pd.DataFrame:
    by_run: dict[str, list] = {}
    for _, trace in traces.iterrows():
        meta = _as_dict(trace.get("trace_metadata"))
        run_id = meta.get("mlflow.sourceRun")
        if run_id:
            by_run.setdefault(str(run_id), []).append(trace)

    runs_by_id = {str(row["run_id"]): row for _, row in runs.iterrows()}
    rows, seen = [], set()
    for run_id, run in runs_by_id.items():
        for trace in by_run.get(run_id, []):
            response = _as_dict(trace.get("response"))
            request = _as_dict(trace.get("request"))
            session = str(response.get("session_id") or run.get("session") or "")
            execution = str(response.get("execution_id") or "")
            if not session.startswith(prefix):
                continue
            key = (run_id, execution)
            if key in seen:
                continue
            seen.add(key)

            inp = _as_dict(request.get("input"))
            message = str(inp.get("message") or "")
            answer = str((_as_dict(response.get("result"))).get("answer") or "")
            spans = _as_list(trace.get("spans"))
            names = [s.get("name") for s in spans]
            parent_names = {}
            for span in spans:
                parent_names[span.get("span_id")] = span.get("name")
            nested = 0
            for span in spans:
                parent = parent_names.get(span.get("parent_span_id"))
                if span.get("name") in contract and parent == span.get("name"):
                    nested += 1

            tools = [n for n in names if n in contract or n in builtin]
            usage = _as_dict(trace.get("trace_metadata"))
            try:
                usage = json.loads(usage.get("mlflow.trace.tokenUsage") or "{}")
            except (TypeError, ValueError):
                usage = {}

            hook_events = _as_list(response.get("hook_events"))
            failures = [h for h in hook_events
                        if h.get("event") == "PostToolUseFailure"]
            history_turns = _as_list(
                _as_dict(response.get("session_history")).get("turns")
            )
            fanout = []
            for span in spans:
                if span.get("name") not in ("model", "ChatOpenAI"):
                    continue
                out = span.get("attributes", {}).get("mlflow.spanOutputs", "{}")
                if isinstance(out, str):
                    try:
                        out = json.loads(out)
                    except (TypeError, ValueError):
                        out = {}
                if not isinstance(out, dict):
                    continue
                msg = (_as_list(out.get("choices")) or [{}])[0].get("message", {})
                calls = _as_list(msg.get("tool_calls"))
                if calls:
                    fanout.append(len(calls))

            suffix = session[len(prefix):]
            suffix = re.sub(r"^[0-9a-f]{4,}-", "", suffix, count=1)
            scenario = re.sub(r"-q\d+$", "", suffix).split("-")[0]

            rows.append({
                "execution_id": execution,
                "run_id": str(run_id),
                "session_id": session,
                "scenario": scenario or "?",
                "adhoc": scenario == "adhoc",
                "start": run.get("start"),
                "duration_ms": trace.get("execution_duration"),
                "tokens_in": usage.get("input_tokens"),
                "tokens_out": usage.get("output_tokens"),
                "tokens_total": usage.get("total_tokens"),
                "model_spans": sum(1 for n in names if n in ("model", "ChatOpenAI")),
                "n_spans": len(spans),
                "nested_tool_spans": nested,
                "tools": tools,
                "contract_tools": [n for n in tools if n in contract],
                "noise_tools": [n for n in tools if n in builtin],
                "read_file": names.count("read_file"),
                "write_file": names.count("write_file"),
                "hook_failures": len(failures),
                "hook_feedback": any(
                    "feedback" in (h.get("decisions") or []) for h in failures
                ),
                "status": response.get("status"),
                "agent_version": response.get("agent_version"),
                "drift": response.get("agent_version_drift"),
                "answer": answer,
                "derailed": any(m in answer.lower() for m in DERAIL_MARKERS),
                "message": message,
                "history_turns": len(history_turns),
                "history_text": json.dumps(history_turns, ensure_ascii=False),
                "explicit": bool(re.search(r"gọi tool|hãy gọi|call tool",
                                           message, re.I)),
                "fanout": ",".join(str(n) for n in fanout),
                "memory_code": (memory_regex.search(message).group(0)
                                if memory_regex.search(message) else ""),
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df = df.sort_values(["session_id", "start"]).reset_index(drop=True)
    df["turn"] = df.groupby("session_id").cumcount()
    return df


# --- stage 4: aggregate ------------------------------------------------------


def scenario_rows(df: pd.DataFrame) -> list[list]:
    out = []
    for name, group in df.groupby("scenario"):
        noise = (group["read_file"] + group["write_file"]) > 0
        out.append([
            name,
            len(group),
            _pct((group["status"] == "SUCCEEDED").sum(), len(group)),
            f"{(_median(group['duration_ms']) or 0) / 1000:.1f}",
            f"{_median(group['tokens_in']) or 0:.0f}",
            f"{_median(group['tokens_out']) or 0:.0f}",
            _pct(group["derailed"].sum(), len(group)),
            _pct(noise.sum(), len(group)),
            _pct(group["hook_feedback"].sum(), len(group)),
        ])
    return sorted(out, key=lambda row: row[0])


def latency_rows(df: pd.DataFrame) -> list[list]:
    out = []
    for name, group in df.groupby("scenario"):
        values = sorted(v / 1000 for v in group["duration_ms"].dropna())
        out.append([name, len(values), f"{values[0]:.0f}",
                    f"{_median(values) or 0:.0f}",
                    f"{_quantile(values, 0.9) or 0:.0f}",
                    f"{values[-1]:.0f}"])
    return sorted(out, key=lambda row: row[0])


def tool_rows(df: pd.DataFrame, contract: set[str],
              builtin: set[str]) -> tuple[list[list], list[str]]:
    counts = Counter()
    for tools in df["tools"]:
        counts.update(set(tools))
    rows = []
    for name, count in counts.most_common():
        kind = "contract" if name in contract else (
            "builtin" if name in builtin else "other")
        rows.append([f"`{name}`", _pct(count, len(df)), kind])
    return rows, sorted(contract - set(counts))


def fanout_stats(df: pd.DataFrame) -> tuple[int, float, float, int, dict]:
    values = []
    for raw in df["fanout"]:
        values.extend(int(v) for v in str(raw).split(",") if v)
    if not values:
        return 0, 0, 0, 0, {}
    return (len(values), statistics.median(values),
            round(statistics.mean(values), 1), max(values),
            dict(sorted(Counter(values).items())))


def booking_stats(df: pd.DataFrame) -> dict:
    book = df[df["scenario"] == "booking"]
    stats = {"sessions": book["session_id"].nunique(), "quote": 0,
             "clean": 0, "early_write": 0, "derailed": 0,
             "confirm": 0, "confirm_booking": 0, "confirm_derailed": 0}
    for _, group in book.groupby("session_id"):
        group = group.sort_values("start")
        first = group.iloc[0]
        stats["quote"] += 1
        tools = set(first["tools"])
        if tools & WRITE_TOOLS:
            stats["early_write"] += 1
        if tools & QUOTE_TOOLS and not tools & WRITE_TOOLS:
            stats["clean"] += 1
        if first["derailed"]:
            stats["derailed"] += 1
        for _, turn in group.iloc[1:].iterrows():
            stats["confirm"] += 1
            if "create_booking" in set(turn["tools"]):
                stats["confirm_booking"] += 1
            if turn["derailed"]:
                stats["confirm_derailed"] += 1
    return stats


def memory_stats(df: pd.DataFrame) -> dict:
    """Three-turn memory sessions: turn 1 plants a code, later turns recall it.

    The code is only in the first turn's message, so it has to be located
    per session (not per row), then checked against the *later* turns.
    """
    mem = df[df["scenario"] == "memory"]
    stats = {"sessions": 0, "later": 0, "stated": 0, "sessions_hit": 0,
             "history": 0, "history_total": 0}
    for _, group in mem.groupby("session_id"):
        group = group.sort_values("start").reset_index(drop=True)
        planted = [i for i, code in enumerate(group["memory_code"]) if code]
        if not planted:
            continue
        code = str(group.loc[planted[0], "memory_code"])
        later = group.iloc[planted[0] + 1:]
        stats["sessions"] += 1
        hits = [code in str(answer) for answer in later["answer"]]
        stats["later"] += len(hits)
        stats["stated"] += sum(hits)
        stats["sessions_hit"] += int(any(hits))
        for _, turn in later.iterrows():
            if turn["history_turns"]:
                stats["history_total"] += 1
                if code in str(turn["history_text"]):
                    stats["history"] += 1
    return stats


# --- stage 5: render ---------------------------------------------------------


def render(df: pd.DataFrame, meta: dict, *, contract: set[str],
           builtin: set[str], top: int, excluded: set[str]) -> str:
    # Analytical scope: the manual suite. ``adhoc`` (exploratory probes)
    # and ``probe`` (one-off wiring checks) stay in the CSV but are kept
    # out of the statistics, which would otherwise mix prompt versions.
    off = df[~df["scenario"].isin(excluded)].copy()
    out = ["# Báo cáo MLflow — manual test hotel reception", ""]
    out.append(f"- Thời điểm tạo: {meta['generated_at']}")
    out.append(f"- Tracking URI: `{meta['tracking_uri']}`")
    out.append(f"- Experiment: `{meta['experiment']}` (id={meta['experiment_id']})")
    out.append(f"- Bộ lọc: session prefix `{meta['prefix']}`"
               + (f", agent `{meta['agent']}`" if meta["agent"] else "")
               + (f", từ {meta['since']}" if meta["since"] else ""))
    out.append(f"- Phạm vi thống kê: {len(off)} trace thuộc kịch bản của suite"
               + (f" (loại `{'`, `'.join(sorted(excluded))}`)" if excluded else ""))
    out.append("")

    out += ["## 1. Khối lượng dữ liệu", ""]
    status = df["status"].value_counts().to_dict()
    versions = df["agent_version"].value_counts().to_dict()
    drift_true = df[df["drift"] == True]  # noqa: E712
    drift_text = f"{len(df) - len(drift_true)}/{len(df)} False"
    if len(drift_true):
        first = drift_true["start"].min()
        drift_text += (f"; True đầu/cuối: {first:%Y-%m-%d %H:%M} UTC "
                       f"({len(drift_true)} lần)")
    noise_all = (df["read_file"] + df["write_file"]) > 0
    out += _table(["Mục", "Giá trị"], [
        ["Run thủ công (sau lọc)", meta["runs"]],
        ["Trace khớp (toàn bộ)", len(df)],
        ["Session", df["session_id"].nunique()],
        ["Execution SUCCEEDED", status.get("SUCCEEDED", 0)],
        ["Execution FAILED", status.get("FAILED", 0)],
        ["AgentVersion", ", ".join(f"v{k}: {v}" for k, v in versions.items())],
        ["Drift", drift_text],
        ["Trace có read_file/write_file (toàn bộ)",
         f"{noise_all.sum()}/{len(df)} ({_pct(noise_all.sum(), len(df))})"],
        ["Trace có hook feedback (toàn bộ)",
         f"{df['hook_feedback'].sum()}/{len(df)} "
         f"({_pct(df['hook_feedback'].sum(), len(df))})"],
        ["Câu trả lời cuối bị derail (toàn bộ)",
         f"{df['derailed'].sum()}/{len(df)} "
         f"({_pct(df['derailed'].sum(), len(df))})"],
    ]).splitlines()
    out.append("")

    out += ["## 2. Theo kịch bản (phạm vi suite)", ""]
    out.append(_table(
        ["Kịch bản", "n", "OK", "p50 (s)", "token in", "token out",
         "derail", "nhiễu file", "hook feedback"],
        scenario_rows(off),
    ))
    out.append("")

    out += ["## 3. Độ trễ (giây)", ""]
    out.append(_table(["Kịch bản", "n", "min", "p50", "p90", "max"],
                      latency_rows(off)))
    out.append("")

    out += ["## 4. Token", ""]
    tok_in, tok_out = off["tokens_in"].dropna(), off["tokens_out"].dropna()
    rows = []
    for label, series in (("input", tok_in), ("output", tok_out)):
        rows.append([label, f"{series.median():.0f}",
                     f"{series.quantile(0.9):.0f}", f"{series.max():.0f}"])
    out.append(_table(["Loại", "median", "p90", "max"], rows))
    out.append("")
    turn_rows = []
    for turn, group in off.groupby("turn"):
        if len(group) >= 3:
            turn_rows.append([int(turn), len(group),
                              f"{_median(group['tokens_in']) or 0:.0f}"])
    if turn_rows:
        out.append("Token vào theo lượt trong session:")
        out.append("")
        out += _table(["Lượt", "n", "token in (median)"], turn_rows).splitlines()
        out.append("")

    out += ["## 5. Sử dụng tool", ""]
    rows, unused = tool_rows(off, contract, builtin)
    out.append(_table(["Tool", "Tỷ lệ trace dùng", "Loại"], rows))
    out.append("")
    out.append(f"Tool contract chưa từng dùng: "
               f"{', '.join(f'`{n}`' for n in unused) or 'không có'}")
    out.append("")

    out += ["## 6. Nhiễu file-tool (deepagents built-in)", ""]
    noisy = off[(off["read_file"] + off["write_file"]) > 0]
    out += _table(["Mục", "Giá trị"], [
        ["Trace có read_file/write_file", f"{len(noisy)}/{len(off)} "
                                          f"({_pct(len(noisy), len(off))})"],
        ["Tổng span read_file", int(off["read_file"].sum())],
        ["Tổng span write_file", int(off["write_file"].sum())],
        ["Trace có hook feedback",
         f"{off['hook_feedback'].sum()}/{len(off)} "
         f"({_pct(off['hook_feedback'].sum(), len(off))})"],
        ["Câu trả lời cuối bị derail",
         f"{off['derailed'].sum()}/{len(off)} "
         f"({_pct(off['derailed'].sum(), len(off))})"],
    ]).splitlines()
    out.append("")
    targets = Counter()
    for answer in off["answer"]:
        for match in re.findall(r"File '([^']+)' not found", str(answer)):
            targets[match] += 1
    if targets:
        out.append("Đường dẫn model bịa nhiều nhất:")
        out.append("")
        out += _table(["Lần", "Đường dẫn"],
                      [[n, f"`{p}`"] for p, n in targets.most_common(top)]
                      ).splitlines()
        out.append("")

    out += ["## 7. Fan-out tool mỗi lượt model", ""]
    n_turns, median, mean, maximum, dist = fanout_stats(off)
    out += _table(["n lượt", "median", "mean", "max", "phân bố"], [
        [n_turns, median, mean, maximum,
         ", ".join(f"{k}:{v}" for k, v in dist.items())],
    ]).splitlines()
    out.append("")

    out += ["## 8. Prompt gọi tên tool vs giọng khách", ""]
    style_rows = []
    for explicit, group in off.groupby("explicit"):
        label = "explicit (gọi tên tool)" if explicit else "natural (giọng khách)"
        style_rows.append([
            label, len(group),
            _pct(group["derailed"].sum(), len(group)),
            _pct(((group["read_file"] + group["write_file"]) > 0).sum(),
                 len(group)),
            f"{(_median(group['duration_ms']) or 0) / 1000:.0f}",
        ])
    out += _table(["Kiểu prompt", "n", "derail", "nhiễu file", "p50 (s)"],
                  sorted(style_rows)).splitlines()
    out.append("")

    b = booking_stats(off)
    out += ["## 9. Booking: cổng xác nhận", ""]
    out += _table(["Lượt", "Số lượng", "Ghi chú"], [
        ["Session", b["sessions"], ""],
        ["Lượt hỏi giá", b["quote"],
         f"sạch {b['clean']}, tự ghi {b['early_write']}, "
         f"derail {b['derailed']}"],
        ["Lượt xác nhận/follow-up", b["confirm"],
         f"có create_booking {b['confirm_booking']}, "
         f"derail {b['confirm_derailed']}"],
    ]).splitlines()
    out.append("")

    m = memory_stats(off)
    out += ["## 10. Memory 3 lượt", ""]
    out += _table(["Mục", "Giá trị"], [
        ["Session", m["sessions"]],
        ["Session có nhắc mã ở lượt sau", f"{m['sessions_hit']}/{m['sessions']}"],
        ["Lượt sau nhắc đúng mã", f"{m['stated']}/{m['later']}"],
        ["Rendered history chứa mã", f"{m['history']}/{m['history_total']}"],
    ]).splitlines()
    out.append("")

    out += ["## 11. Ngoại lệ", ""]
    slow = off.nlargest(top, "duration_ms")
    out.append("Chậm nhất:")
    out.append("")
    out += _table(["Execution", "Kịch bản", "Giây", "token in", "tool"],
                  [[r["execution_id"], r["scenario"],
                    f"{r['duration_ms'] / 1000:.0f}",
                    f"{r['tokens_in']:.0f}" if pd.notna(r["tokens_in"]) else "-",
                    ", ".join(f"`{t}`" for t in r["tools"][:6]) or "-"]
                   for _, r in slow.iterrows()]).splitlines()
    out.append("")
    big = off.nlargest(top, "tokens_in")
    out.append("Context lớn nhất:")
    out.append("")
    out += _table(["Execution", "Kịch bản", "token in", "token out", "giây"],
                  [[r["execution_id"], r["scenario"],
                    f"{r['tokens_in']:.0f}",
                    f"{r['tokens_out']:.0f}" if pd.notna(r["tokens_out"]) else "-",
                    f"{r['duration_ms'] / 1000:.0f}"]
                   for _, r in big.iterrows()]).splitlines()
    out.append("")

    nested = off[off["nested_tool_spans"] > 0]
    out += ["## 12. Ghi chú đọc số liệu", ""]
    out.append(
        f"- {len(nested)}/{len(off)} trace có span tool lồng cùng tên "
        f"({int(off['nested_tool_spans'].sum())} span). Đây là instrumentation "
        "lồng nhau của deepagents (ToolNode → wrapper → tool), không phải "
        "gọi MCP hai lần; `audit_logs` vì vậy cũng có 2 row/tool."
    )
    out.append(
        "- Đường dẫn `/large_tool_results/...` đến từ middleware message-"
        "eviction của deepagents; model 3B tự bịa thêm tên file rồi gọi "
        "`read_file` (filesystem thật) và thất bại."
    )
    out.append(
        "- `--cleanup` chỉ xoá row Postgres; run/trace MLflow vẫn còn, nên "
        "báo cáo chạy lại được sau khi dọn DB."
    )
    out.append("")
    return "\n".join(out) + "\n"


# --- CLI ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking URI (default: env "
                             "MLFLOW_TRACKING_URI or http://localhost:5000)")
    parser.add_argument("--experiment", default="agentic-runtime",
                        help="experiment name or id (default: agentic-runtime)")
    parser.add_argument("--session-prefix", default="sess-hotel-",
                        help="only runs whose session_id starts with this "
                             "(default: sess-hotel-)")
    parser.add_argument("--agent", default=None,
                        help="only runs tagged with this agent_id")
    parser.add_argument("--since", default=None,
                        help="only runs started on/after this ISO date")
    parser.add_argument("--exclude-scenarios", default="adhoc,probe",
                        help="comma-separated scenarios kept out of the "
                             "statistics (default: adhoc,probe)")
    parser.add_argument("--contract-tools", default=",".join(HOTEL_TOOLS),
                        help="comma-separated MCP tool names")
    parser.add_argument("--builtin-tools", default=",".join(BUILTIN_TOOLS),
                        help="comma-separated built-in tool names")
    parser.add_argument("--memory-code-regex", default=r"HOTEL-[A-Z0-9]+",
                        help="regex of the code the memory scenario recalls")
    parser.add_argument("--max-traces", type=int, default=5000,
                        help="upper bound for run/trace searches")
    parser.add_argument("--top", type=int, default=5,
                        help="rows in outlier tables (default: 5)")
    parser.add_argument("--out", default="reports/hotel-mlflow-report.md",
                        help="Markdown output path ('' = stdout)")
    parser.add_argument("--csv", default="reports/hotel-mlflow-traces.csv",
                        help="per-trace CSV path ('' = skip)")
    args = parser.parse_args()

    load_dotenv()
    tracking = (args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
                or "http://localhost:5000")
    os.environ["MLFLOW_TRACKING_URI"] = tracking
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    import mlflow

    mlflow.set_tracking_uri(tracking)
    experiment_id, experiment_name = resolve_experiment(args.experiment)
    since = None
    if args.since:
        since = pd.Timestamp(args.since, tz="UTC")

    contract = set(_split(args.contract_tools))
    builtin = set(_split(args.builtin_tools))
    excluded = set(_split(args.exclude_scenarios))
    memory_regex = re.compile(args.memory_code_regex)

    print(f"MLflow {tracking} · experiment {experiment_name} "
          f"(id={experiment_id})")
    runs = load_runs(experiment_id, args.session_prefix, args.agent, since,
                     args.max_traces)
    if runs.empty:
        raise SystemExit(f"no runs with session prefix {args.session_prefix!r}")
    traces = load_traces(experiment_id, args.max_traces)
    df = collect(runs, traces, prefix=args.session_prefix, contract=contract,
                 builtin=builtin, memory_regex=memory_regex)
    if df.empty:
        raise SystemExit("runs matched but no execution traces did; "
                         "run the manual scripts first")

    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tracking_uri": tracking,
        "experiment": experiment_name,
        "experiment_id": experiment_id,
        "prefix": args.session_prefix,
        "agent": args.agent,
        "since": args.since,
        "runs": len(runs),
    }
    report = render(df, meta, contract=contract, builtin=builtin,
                    top=args.top, excluded=excluded)

    suite = df[~df["scenario"].isin(excluded)]
    print(f"runs={len(runs)} traces_matched={len(df)} "
          f"suite={len(suite)} sessions={suite['session_id'].nunique()}")
    print(f"scenarios: {', '.join(sorted(df['scenario'].unique()))}")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"csv  -> {path}")
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"report -> {path}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
