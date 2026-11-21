"""Human-in-the-loop interrupts (MDA ``interrupt_on``).

A mission may declare, in ``execution.interrupt_on``, the tools that
must pause for a human decision before they run::

    execution:
      interrupt_on:
        delete_mission: true
        cancel_execution:
          allowed_decisions: [approve, reject]
          description: "Cancelling an execution is irreversible."

The values are the ``define_deep_agent`` / ``create_deep_agent``
``interrupt_on`` mapping: a tool name to ``true`` (interrupt before
the call, default decisions approve/reject) or to an
``InterruptOnConfig`` dict (``allowed_decisions`` and ``description``
only — ``when`` is a callable and cannot come from YAML).

Mechanics (LangGraph-native, three levels):

* the deep agent is built with ``interrupt_on`` + a checkpointer
  (``runtime.deep_agent.build_deep_agent``). When a gated tool is
  called the agent pauses with an ``__interrupt__`` whose value is a
  ``HITLRequest`` (batched mode: one interrupt per model turn).
* ``deep_agent_node`` turns that into a mission-graph ``interrupt()``
  pause, so the whole run stops and the worker persists status
  ``INTERRUPTED`` with the pending calls in the result.
* ``POST /executions/{id}/resume`` re-dispatches the SAME execution
  with a decision (``approve`` / ``reject``); ``run_mission`` resumes
  the mission graph with ``Command(resume=decision)`` and the node
  resumes the deep agent with ``Command(resume=HITLResponse(...))``.

The existing permission-request flow (``request_permission`` hook,
``GET/POST /permissions``) is NOT replaced: it still gates
``delete_mission`` / ``cancel_execution`` for admin_bot by
re-dispatching a fresh execution. ``interrupt_on`` is the general,
resume-in-place mechanism for any tool; a mission may use either.

Checkpointer requirement: a pause only survives a worker restart
when the checkpointer is Postgres (``CHECKPOINT_ENABLED=true`` +
the ``postgres-checkpoint`` extra). Without it, interrupts are
fail-closed outside dev/test: a mission that declares
``interrupt_on`` refuses to run rather than silently running the
gated tool unapproved. In dev/test the fallback is a process-local
``InMemorySaver`` singleton (resume only works in the same process).

Flow mode (``flow_subgraph.enabled``) does NOT support
``interrupt_on`` yet: the flow compiler would need its own
checkpoint/resume plumbing (flow subgraph + per-flow agent threads).
A mission that declares both fails fast with a clear error.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Execution status for a run paused on a human decision. Stored in
# Execution.status (a free string column, no migration) and in
# MissionState["status"]; the worker leaves finished_at unset while
# the run waits.
INTERRUPTED = "INTERRUPTED"

ALLOWED_DECISIONS = ("approve", "reject")
# YAML InterruptOnConfig keys (``when`` is a callable — not from YAML).
_CONFIG_KEYS = ("allowed_decisions", "description")


def parse_interrupt_on(raw: Any) -> dict[str, Any] | None:
    """Normalize ``execution.interrupt_on`` into a ``create_deep_agent`` mapping.

    Returns None when nothing is declared. Raises ValueError on a
    malformed declaration so a typo fails fast at run start instead
    of silently leaving a tool ungated.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("execution.interrupt_on must be a mapping of tool name to true/config")
    normalized: dict[str, Any] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("execution.interrupt_on keys must be non-empty tool names")
        if value is True:
            normalized[name] = True
        elif isinstance(value, dict):
            unknown = sorted(set(value) - set(_CONFIG_KEYS))
            if unknown:
                raise ValueError(
                    f"interrupt_on[{name}] has unsupported keys: {unknown} "
                    "(only allowed_decisions and description come from YAML)"
                )
            decisions = value.get("allowed_decisions")
            if decisions is not None:
                if not isinstance(decisions, list) or not decisions:
                    raise ValueError(
                        f"interrupt_on[{name}].allowed_decisions must be a non-empty list"
                    )
                bad = [d for d in decisions if d not in ALLOWED_DECISIONS]
                if bad:
                    raise ValueError(
                        f"interrupt_on[{name}] has unsupported decisions: {bad} "
                        f"(v1 supports {list(ALLOWED_DECISIONS)})"
                    )
            normalized[name] = dict(value)
        else:
            raise ValueError(
                f"interrupt_on[{name}] must be true or a config mapping, got {value!r}"
            )
    return normalized or None


def extract_interrupt_requests(interrupt_value: Any) -> list[dict[str, Any]]:
    """Turn an ``__interrupt__`` value into plain pending-call dicts.

    Two shapes occur: the deep agent's ``HITLRequest`` (batched mode,
    ``action_requests`` with ``name``/``args``) and the mission-level
    payload this module builds (``{"requests": [{"tool", "args"}]}`` —
    see ``interrupt_payload``). Each entry is ``{"tool": name, "args":
    {...}}`` — JSON-safe for the result. Unknown shapes yield an
    empty list (the pause still happened; the resume needs only the
    count, see ``build_resume_command``).
    """
    requests: list[dict[str, Any]] = []
    value = interrupt_value
    if isinstance(value, list):
        for item in value:
            inner = getattr(item, "value", item)
            requests.extend(extract_interrupt_requests(inner))
        return requests
    if isinstance(value, dict) and isinstance(value.get("requests"), list):
        for req in value["requests"]:
            if isinstance(req, dict) and isinstance(req.get("tool"), str):
                args = req.get("args") or {}
                if not isinstance(args, dict):
                    args = {"value": str(args)}
                requests.append({"tool": req["tool"], "args": args})
        return requests
    action_requests = getattr(value, "action_requests", None)
    if action_requests is None and isinstance(value, dict):
        action_requests = value.get("action_requests")
    for req in action_requests or []:
        if isinstance(req, dict):
            name = req.get("name")
            args = req.get("args") or {}
        else:
            name = getattr(req, "name", None)
            args = getattr(req, "args", None) or {}
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(args, dict):
            args = {"value": str(args)}
        requests.append({"tool": name, "args": args})
    return requests


def interrupt_payload(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """The payload stored in state["interrupt"] and the result."""
    return {
        "requests": requests,
        "allowed_decisions": list(ALLOWED_DECISIONS),
    }


def build_resume_command(decision: str, request_count: int):
    """Build ``Command(resume=HITLResponse(...))`` for a decision.

    One decision per pending call, in order (batched mode matches
    decisions to calls by position). ``approve`` runs every pending
    call; ``reject`` skips them with a reviewer message.
    """
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"resume decision must be one of {list(ALLOWED_DECISIONS)}")
    if request_count < 1:
        raise ValueError("cannot resume: no pending interrupted calls")
    from langchain.agents.middleware.human_in_the_loop import (
        ApproveDecision,
        HITLResponse,
        RejectDecision,
    )
    from langgraph.types import Command

    if decision == "approve":
        decisions = [ApproveDecision(type="approve") for _ in range(request_count)]
    else:
        decisions = [
            RejectDecision(type="reject", message="Rejected by human reviewer")
            for _ in range(request_count)
        ]
    return Command(resume=HITLResponse(decisions=decisions))


_memory_checkpointer: Any | None = None


def _memory_checkpointer_singleton() -> Any:
    """Process-local InMemorySaver shared by pause and resume (dev/test only)."""
    global _memory_checkpointer
    if _memory_checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        _memory_checkpointer = InMemorySaver()
    return _memory_checkpointer


@contextlib.contextmanager
def interrupt_checkpointer():
    """Yield the checkpointer an interrupt_on run needs, or None.

    Postgres first (durable across worker restarts). In dev/test,
    when Postgres is unavailable, the process-local memory saver.
    Anywhere else, None — the caller must fail closed.
    """
    from runtime.mission_graph import get_checkpointer

    with get_checkpointer() as checkpointer:
        if checkpointer is not None:
            yield checkpointer
            return
    from app.config import settings

    if settings.environment in ("dev", "test") or "PYTEST_CURRENT_TEST" in os.environ:
        logger.warning(
            "interrupt_on: no Postgres checkpointer; using process-local "
            "memory checkpointer (resume only works in this process)"
        )
        yield _memory_checkpointer_singleton()
        return
    yield None


def agent_thread_id(execution_id: str) -> str:
    """Stable thread id for a run's deep agent (pause and resume must match)."""
    return f"{execution_id}:deep-agent"
