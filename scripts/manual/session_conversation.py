"""Drive a real multi-turn **conversation** through sessions (Option B).

Sessions are the one place the platform keeps conversation memory: all
executions sharing a ``session_id`` are one conversation, and a later
run is shown the prior turns (``docs/sessions.md``). Without a
``session_id`` every execution is still its own one-turn session, exactly
as before.

This script proves the memory through the real API + worker:

1. turn 1 asks the admin bot to list missions and remember a random code;
2. turn 2 (same session) lists again and asks for the code — it can only
   answer from the session history, because the code is not in turn 2's
   request and no tool stores it;
3. turn 3 asks for a one-sentence summary plus the code, after a health
   check, so the run has **two** prior turns to read back;
4. ``GET /sessions/{id}`` must report every execution, oldest first.

Each turn names a tool to call on purpose. ``system_admin`` declares the
``stop_completion`` hook (Ralph-loop guard): when the agent tries to stop
without any tool call, the hook forces another model turn. The small
local model then falls back to the only no-arg tool (``system_health``)
and its output tends to drown out the history. Naming a tool keeps the
turn deterministic while still requiring the model to read the prior
turns for the code.

``--control`` adds one execution in a **fresh** session asking the same
recall question; its answer must NOT contain the code (the control is
random per run, so a lucky guess is very unlikely).

Usage (from the repo root, with the API + worker running):

    python scripts/manual/session_conversation.py
    python scripts/manual/session_conversation.py --control
    python scripts/manual/session_conversation.py --secret 7391
    python scripts/manual/session_conversation.py --cleanup
    python scripts/manual/session_conversation.py --session-id sess-demo-1

Exit code: 0 when the session recalls the code and the session listing
is consistent, 1 otherwise. Needs the project venv for the SQLAlchemy
cleanup (``--cleanup``).
"""

from __future__ import annotations

import argparse
import json
import secrets
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
AGENT_ID = "admin_bot"


def make_turns(secret: str) -> list[str]:
    return [
        f"Hãy gọi list_missions một lần, sau đó ghi nhớ mã bí mật của cuộc "
        f"hội thoại này là {secret} và xác nhận ngắn gọn.",
        "Hãy gọi list_missions một lần nữa, sau đó trả lời: mã bí mật ở "
        "lượt trước là gì?",
        "Hãy gọi system_health một lần, sau đó tóm tắt cuộc hội thoại này "
        "trong đúng một câu và nhắc lại mã bí mật.",
    ]


def answer_of(output: dict) -> str:
    return output.get("answer") or output.get("summary") or ""


def run_turn(session_id: str, request: str, timeout: float) -> dict:
    """Submit one turn in the session and wait for its terminal state."""
    execution = submit(MISSION_ID, {"request": request}, session_id=session_id)
    execution, elapsed = wait_for(execution["execution_id"], timeout=timeout)
    result = get_result(execution["execution_id"])
    output = result.get("output") or {}
    return {
        "execution": execution,
        "elapsed": elapsed,
        "answer": answer_of(output),
        "actions": output.get("actions_taken") or [],
        "evidence": result.get("evidence") or output.get("evidence") or [],
    }


def get_session(session_id: str) -> dict | None:
    status, payload = call("GET", f"/sessions/{session_id}")
    if status == 404:
        return None
    if status != 200 or payload is None:
        raise RuntimeError(f"GET /sessions/{session_id} -> HTTP {status}: {payload}")
    return payload


def print_turn(number: int, request: str, turn: dict) -> None:
    execution = turn["execution"]
    print(f"\n[turn {number}] {execution['execution_id']} "
          f"{execution['status']} in {turn['elapsed']:.1f}s "
          f"session={execution.get('session_id')}")
    print(f"  request: {request}")
    print(f"  actions: {json.dumps(turn['actions'], ensure_ascii=False)}")
    print(f"  answer: {json.dumps(turn['answer'][:500], ensure_ascii=False)}")


def cleanup(session_id: str) -> None:
    """Delete the session's executions and their result/usage/audit rows."""
    from app import models
    from app.core.db import SessionLocal

    db = SessionLocal()
    try:
        execution_ids = [
            row.id for row in
            db.query(models.Execution).filter_by(session_id=session_id)
        ]
        if execution_ids:
            db.query(models.ExecutionResult).filter(
                models.ExecutionResult.execution_id.in_(execution_ids)
            ).delete(synchronize_session=False)
            db.query(models.LLMUsage).filter(
                models.LLMUsage.execution_id.in_(execution_ids)
            ).delete(synchronize_session=False)
            db.query(models.AuditLog).filter(
                models.AuditLog.execution_id.in_(execution_ids)
            ).delete(synchronize_session=False)
            db.query(models.Execution).filter(
                models.Execution.id.in_(execution_ids)
            ).delete(synchronize_session=False)
        db.commit()
        print(f"cleanup: session {session_id}, "
              f"{len(execution_ids)} execution(s) and their result/usage/audit rows")
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--session-id", default=None,
                        help="reuse a session id (default: sess-manual-<8 hex>)")
    parser.add_argument("--secret", default=None,
                        help="code to remember (default: random 4 digits)")
    parser.add_argument("--control", action="store_true",
                        help="also run the recall question in a fresh session "
                             "(must NOT know the code)")
    parser.add_argument("--cleanup", action="store_true",
                        help="delete this session's execution rows afterwards")
    parser.add_argument("--timeout", type=float, default=300,
                        help="seconds to wait per turn (default 300)")
    args = parser.parse_args()

    require_worker()
    status, mission = call("GET", f"/missions/{MISSION_ID}")
    if status != 200:
        sys.exit(f"mission {MISSION_ID!r} not found (HTTP {status}) — run `make seed`")

    session_id = args.session_id or f"sess-manual-{secrets.token_hex(4)}"
    secret = args.secret or f"{secrets.randbelow(9000) + 1000}"
    turns = make_turns(secret)
    expected_executions = len(turns)

    print("=== session conversation (Option B)")
    print(f"  session_id={session_id}  secret={secret}  turns={len(turns)}")

    problems: list[str] = []
    ran: list[dict] = []
    try:
        for number, request in enumerate(turns, start=1):
            turn = run_turn(session_id, request, args.timeout)
            ran.append(turn)
            print_turn(number, request, turn)
            if turn["execution"]["status"] != "SUCCEEDED":
                problems.append(
                    f"turn {number} ended {turn['execution']['status']} "
                    f"(error={turn['execution'].get('error')})"
                )

        # Turn 2 can only know the code from the session history.
        if secret not in ran[1]["answer"]:
            problems.append(
                f"turn 2 did not recall {secret!r} from session history "
                f"(answer={ran[1]['answer'][:200]!r})"
            )
        # Turn 3 has two prior turns to read back.
        if secret not in ran[2]["answer"]:
            problems.append(
                f"turn 3 did not carry the code over two prior turns "
                f"(answer={ran[2]['answer'][:200]!r})"
            )

        session = get_session(session_id)
        if session is None:
            problems.append(f"GET /sessions/{session_id} returned 404")
        else:
            print("\n  session:")
            print(f"    agent={session.get('agent_id')} "
                  f"mission={session.get('mission_id')} "
                  f"executions={session.get('execution_count')}")
            if session.get("agent_id") != AGENT_ID:
                problems.append(
                    f"session agent is {session.get('agent_id')!r}, "
                    f"expected {AGENT_ID!r}"
                )
            if session.get("mission_id") != MISSION_ID:
                problems.append(
                    f"session mission is {session.get('mission_id')!r}, "
                    f"expected {MISSION_ID!r}"
                )
            if session.get("execution_count") != expected_executions:
                problems.append(
                    f"session lists {session.get('execution_count')} execution(s), "
                    f"expected {expected_executions}"
                )
            listed = [e["execution_id"] for e in session.get("executions") or []]
            expected = [turn["execution"]["execution_id"] for turn in ran]
            if listed != expected:
                problems.append(
                    f"session executions are not the conversation in order: "
                    f"{listed} != {expected}"
                )
            for execution in session.get("executions") or []:
                if execution.get("status") != "SUCCEEDED":
                    problems.append(
                        f"session execution {execution['execution_id']} is "
                        f"{execution.get('status')}"
                    )

        if args.control:
            print("\n  control: same question in a FRESH session")
            control = run_turn(
                f"sess-control-{secrets.token_hex(4)}",
                "Hãy gọi list_missions một lần, sau đó trả lời: mã bí mật tôi "
                "đã đưa cho bạn là gì? Nếu chưa từng được cho, hãy nói là "
                "không biết.",
                args.timeout,
            )
            print(f"    {control['execution']['execution_id']} "
                  f"{control['execution']['status']} in {control['elapsed']:.1f}s")
            print(f"    answer: {json.dumps(control['answer'][:300], ensure_ascii=False)}")
            if secret in control["answer"]:
                problems.append(
                    f"control session knew {secret!r} without history — "
                    f"session isolation is broken"
                )
            # The control execution is not part of the demo session, clean
            # it up separately when --cleanup is set.
            if args.cleanup:
                cleanup(control["execution"]["session_id"])
    finally:
        if args.cleanup:
            cleanup(session_id)

    if problems:
        print("\n  checks: FAIL")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print(f"\n  checks: PASS — {expected_executions} turns share one session "
              f"and the code survived every turn")
    print(f"\n{'PASS' if not problems else 'FAIL'}: session conversation")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
