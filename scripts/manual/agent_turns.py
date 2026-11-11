"""Prove the Deep Agent runs multiple model/tool turns inside one execution.

The platform keeps **no conversation memory between executions** by
design (README section 8): every ``POST /executions`` is a new
conversation. So "multi-turn" here means the agent loop *inside* a single
execution — the model is called again after its tools return, and the
next model turn can react to those results.

Turn evidence comes from the middleware tables, not from the prose:

- ``llm_usage``  — one row per model call, written at ``on_llm_end``;
- ``audit_logs`` — one row per tool call, written when the tool starts.

Sorting both by ``created_at`` yields the timeline
``LLM turn 1 -> its tools -> LLM turn 2 -> its tools -> ...``, because an
``llm_usage`` row is written when the model call *finishes* and the audit
rows are written after the model emitted its tool calls. The final model
turn answers without tools.

By default this runs the seeded ``system_admin`` mission (autonomous,
admin_bot) with a read-only request that needs several tools. PASS means:

- the execution ended SUCCEEDED;
- ``llm_calls >= 2`` (an agent loop happened, not just one model call);
- ``tool_calls >= 2``;
- every audit row has a terminal status (``success``/``error``) — no
  dangling ``started`` row.

``--strict`` additionally requires tools in at least two *different*
model turns, i.e. the model saw tool output and then asked for more
tools. This is opt-in because a small local model often emits every tool
call in one assistant message: you then still get two model calls
(plan+act, then answer) but only one tool-bearing turn. That is a
model-quality signal, not a runtime bug.

Usage (from the repo root, with the API + worker running):

    python scripts/manual/agent_turns.py
    python scripts/manual/agent_turns.py --min-turns 3 --min-tools 4
    python scripts/manual/agent_turns.py --strict
    python scripts/manual/agent_turns.py --request "Kiểm tra sức khỏe hệ thống"
    python scripts/manual/agent_turns.py --cleanup   # delete this run's rows

Exit code: 0 when the checks pass, 1 otherwise. The script needs the
project venv because it reads the middleware tables through
``runtime.middleware`` / ``app.*`` (SQLAlchemy), not an HTTP endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _api import (  # noqa: E402
    call,
    get_result,
    require_worker,
    submit,
    wait_for,
)

MISSION_ID = "system_admin"

# Read-only, several tools, no ids invented by the caller. It is the same
# shape that produced 4 tool calls in 2 model turns in the pilot runs.
DEFAULT_REQUEST = (
    "Liệt kê các mission hiện có. Sau đó lấy chi tiết mission fault_analysis. "
    "Tiếp theo liệt kê 5 execution gần nhất. Cuối cùng kiểm tra sức khỏe hệ thống."
)

TERMINAL_AUDIT_STATUSES = {"success", "error"}


def load_timeline(execution_id: str) -> tuple[list[dict], list[dict]]:
    """Read the run's model calls and tool calls as plain dicts.

    Converts inside the session so the rows can be grouped after the
    session is closed (no detached-instance errors).
    """
    from app.core.db import SessionLocal
    from runtime.middleware.queries import (
        get_audit_for_execution,
        get_usage_for_execution,
    )

    db = SessionLocal()
    try:
        usage = [
            {
                "id": row.id,
                "model": row.model,
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "total_tokens": row.total_tokens,
                "usage_source": row.usage_source,
                "created_at": row.created_at,
            }
            for row in get_usage_for_execution(db, execution_id)
        ]
        audits = [
            {
                "id": row.id,
                "tool_name": row.tool_name,
                "args_truncated": row.args_truncated,
                "status": row.status,
                "error": row.error,
                "duration_ms": row.duration_ms,
                "created_at": row.created_at,
            }
            for row in get_audit_for_execution(db, execution_id)
        ]
    finally:
        db.close()
    return usage, audits


def group_turns(usage: list[dict], audits: list[dict]) -> tuple[list[dict], list[dict]]:
    """Attach each tool call to the model turn that requested it.

    A tool row belongs to the latest model call at or before its
    ``created_at``. Rows before the first model call are returned
    separately as unattributed (should stay empty).
    """
    turns = [{"usage": row, "tools": []} for row in usage]
    unattributed: list[dict] = []
    for audit in audits:
        turn_index = -1
        for index, row in enumerate(usage):
            if audit["created_at"] >= row["created_at"]:
                turn_index = index
        if turn_index < 0:
            unattributed.append(audit)
        else:
            turns[turn_index]["tools"].append(audit)
    return turns, unattributed


def token_summary(usage: dict) -> str:
    total = usage["total_tokens"]
    if total is None:
        return f"tokens=unreported (source={usage['usage_source']})"
    return (
        f"tokens={total} "
        f"(prompt={usage['prompt_tokens']} "
        f"completion={usage['completion_tokens']} "
        f"source={usage['usage_source']})"
    )


def print_turn(turn_number: int, turn: dict) -> None:
    usage = turn["usage"]
    print(f"  turn {turn_number}  model={usage['model']} {token_summary(usage)}  "
          f"tools={len(turn['tools'])}")
    for tool in turn["tools"]:
        duration = ("?" if tool["duration_ms"] is None
                    else f"{tool['duration_ms']:.0f}ms")
        args = (tool["args_truncated"] or "{}").replace("\n", " ")
        print(f"    - {tool['tool_name']:<24} {tool['status']:<8} "
              f"{duration:>7}  args={args[:120]}")
        if tool["status"] not in TERMINAL_AUDIT_STATUSES:
            print(f"      status is not terminal ({tool['status']})")
        if tool["error"]:
            print(f"      error: {tool['error'][:200]}")


def evaluate(execution: dict, turns: list[dict], unattributed: list[dict],
             args: argparse.Namespace) -> list[str]:
    """Return the list of failed checks (empty means PASS)."""
    problems: list[str] = []
    llm_calls = len(turns)
    tool_calls = sum(len(turn["tools"]) for turn in turns) + len(unattributed)
    tool_turns = [
        number for number, turn in enumerate(turns, start=1) if turn["tools"]
    ]
    non_terminal = [
        tool for turn in turns for tool in turn["tools"]
        if tool["status"] not in TERMINAL_AUDIT_STATUSES
    ] + [
        tool for tool in unattributed
        if tool["status"] not in TERMINAL_AUDIT_STATUSES
    ]

    if execution["status"] != "SUCCEEDED":
        problems.append(
            f"execution ended {execution['status']} (error={execution.get('error')})"
        )
    if llm_calls < args.min_turns:
        problems.append(
            f"only {llm_calls} model call(s), need >= {args.min_turns} "
            f"(--min-turns)"
        )
    if tool_calls < args.min_tools:
        problems.append(
            f"only {tool_calls} tool call(s), need >= {args.min_tools} "
            f"(--min-tools)"
        )
    if non_terminal:
        names = ", ".join(f"{tool['tool_name']}={tool['status']}"
                          for tool in non_terminal)
        problems.append(f"audit rows without a terminal status: {names}")
    if unattributed:
        names = ", ".join(tool["tool_name"] for tool in unattributed)
        problems.append(
            f"{len(unattributed)} tool call(s) happened before the first "
            f"recorded model turn (timeline unreliable): {names}"
        )
    if args.strict and len(tool_turns) < 2:
        problems.append(
            f"only {len(tool_turns)} tool-bearing turn(s) "
            f"({tool_turns or 'none'}); --strict needs tools in >= 2 model turns"
        )
    return problems


def cleanup(execution_id: str) -> None:
    """Delete this run's result, usage, audit, and execution rows."""
    from app import models
    from app.core.db import SessionLocal

    db = SessionLocal()
    try:
        results = db.query(models.ExecutionResult).filter_by(
            execution_id=execution_id
        ).delete(synchronize_session=False)
        usage = db.query(models.LLMUsage).filter_by(
            execution_id=execution_id
        ).delete(synchronize_session=False)
        audits = db.query(models.AuditLog).filter_by(
            execution_id=execution_id
        ).delete(synchronize_session=False)
        executions = db.query(models.Execution).filter_by(
            id=execution_id
        ).delete(synchronize_session=False)
        db.commit()
        print(f"  cleanup: execution={executions} result={results} "
              f"llm_usage={usage} audit_logs={audits}")
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--request", default=DEFAULT_REQUEST,
                        help="admin request text sent to system_admin")
    parser.add_argument("--min-turns", type=int, default=2,
                        help="minimum model calls (default 2)")
    parser.add_argument("--min-tools", type=int, default=2,
                        help="minimum tool calls (default 2)")
    parser.add_argument("--strict", action="store_true",
                        help="also require tools in >= 2 different model turns")
    parser.add_argument("--cleanup", action="store_true",
                        help="delete this run's execution/result/usage/audit rows")
    parser.add_argument("--timeout", type=float, default=300,
                        help="seconds to wait for the run (default 300)")
    args = parser.parse_args()

    require_worker()
    status, mission = call("GET", f"/missions/{MISSION_ID}")
    if status != 200:
        sys.exit(
            f"mission {MISSION_ID!r} not found (HTTP {status}) — seed the DB first "
            f"(seeds/missions/system_admin.yaml)"
        )

    print(f"=== multi-turn: mission={MISSION_ID} "
          f"(min-turns={args.min_turns}, min-tools={args.min_tools}, "
          f"strict={args.strict})")
    print(f"  request: {args.request}")

    execution = submit(MISSION_ID, {"request": args.request})
    execution_id = execution["execution_id"]
    print(f"  execution_id={execution_id} status={execution['status']}")

    passed = False
    try:
        execution, elapsed = wait_for(execution_id, timeout=args.timeout)
        result = get_result(execution_id)
        output = result.get("output") or {}
        print(f"  final after {elapsed:.1f}s: status={execution['status']} "
              f"agent={execution.get('agent_id')} "
              f"version={execution.get('agent_version')} "
              f"error={execution.get('error')}")

        usage, audits = load_timeline(execution_id)
        turns, unattributed = group_turns(usage, audits)
        tool_calls = sum(len(turn["tools"]) for turn in turns) + len(unattributed)
        total_tokens = sum(row["total_tokens"] or 0 for row in usage)
        print(f"  model calls={len(turns)}  tool calls={tool_calls}  "
              f"tokens={total_tokens}")
        for number, turn in enumerate(turns, start=1):
            print_turn(number, turn)
        for tool in unattributed:
            print(f"  (before turn 1) {tool['tool_name']} {tool['status']}")

        actions = output.get("actions_taken") or []
        if actions:
            print(f"  actions_taken={json.dumps(actions, ensure_ascii=False)}")
            audited = {tool["tool_name"] for tool in audits}
            missing = [action for action in actions if action not in audited]
            if missing:
                print(f"  note: actions with no audit row: {missing}")

        answer = output.get("answer") or output.get("summary")
        if answer:
            print(f"  answer={json.dumps(answer, ensure_ascii=False)[:1200]}")

        problems = evaluate(execution, turns, unattributed, args)
        if problems:
            print("  checks: FAIL")
            for problem in problems:
                print(f"    - {problem}")
        else:
            tool_turns = [
                number for number, turn in enumerate(turns, start=1)
                if turn["tools"]
            ]
            print(f"  checks: PASS — {len(turns)} model turn(s), "
                  f"tools in turn(s) {tool_turns or 'none'}")
            passed = True
    finally:
        if args.cleanup:
            cleanup(execution_id)

    print(f"\n{'PASS' if passed else 'FAIL'}: multi-turn agent loop")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
