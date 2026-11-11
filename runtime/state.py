"""Mission execution state.

One state object per execution (no conversation memory across executions).
`evidence` and `errors` accumulate via the `add` reducer so parallel or
repeated nodes append instead of overwriting.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict


class MissionState(TypedDict, total=False):
    execution_id: str
    mission_id: str
    mission: dict
    input: dict
    # Option B session (docs/sessions.md): the session this execution
    # belongs to, and its prior turns — {"turns": [...], "omitted": int,
    # "total_prior": int} as loaded by session_service, or None when
    # the run has no session history (fresh session, capability-only
    # callers that pass nothing). deep_agent_node renders the turns
    # into the system prompt; the capability subgraph ignores them.
    session_id: str | None
    session_history: dict | None
    # "deterministic" | "hybrid" | "autonomous" (from mission.execution.mode)
    mode: str
    # Option C: the AgentVersion that ran this execution (None when no
    # version was active or the mission names no agent), and whether
    # the live tool manifest had drifted from that version's pinned
    # manifest. Resolved once in run_mission for every execution, so
    # capability-only runs record their version too.
    agent_version: int | None
    agent_version_drift: bool
    # Hooks: one record per hook dispatch in this execution (event,
    # tool, decision actions, messages) — see runtime/hooks/.
    hook_events: Annotated[list, add]
    plan: Annotated[list, add]
    evidence: Annotated[list, add]
    result: dict | None
    # QUEUED | RUNNING | SUCCEEDED | FAILED | CANCELLED
    status: str
    errors: Annotated[list, add]
    notes: Annotated[list, add]
