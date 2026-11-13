"""The plan gate (Phase 3a): the skill map stops being advice and becomes
the tool allowlist, enforced at the call boundary.

The architecture proposal (docs, 2026-11-13) diagnosed the task scenario
as a harness bug: an advisory YAML map plus a hard test assertion is
incoherent — either the map is enforced or the assertion goes. The gate
is the enforcement: a ``langchain`` ``AgentMiddleware`` built from the
run's parsed ``SkillMap`` s, sitting in the per-run middleware list next
to the hook middleware and the built-in exclusion (``deep_agent_node``).

Rules, per map:

- The gate tracks one *current node* per map, starting at the map's
  ``start``. A tool call is allowed when its node is the current node,
  the next tool node in map order, a branch target of the current node,
  a branch target of the *start* node (re-route — the classifier node is
  always re-enterable, so one execution can serve a second request), or
  an already-visited node (re-visit). Reasoning-only nodes (no tools)
  are passed through: a branch to ``booking-dates`` resolves to
  ``check-availability``, the next tool node after it.
- A call outside that set is DENIED with an error ``ToolMessage`` naming
  the allowed next tools — the agent corrects itself on the next turn
  instead of the tool running. Denials are counted; past
  ``MAX_DENIALS`` the gate raises ``PlanGateExceededError`` so the run
  fails with a diagnostic instead of looping until step exhaustion.
- A tool in no map node is ALLOWED with a warning (fail-open): a missing
  map entry is a config bug, not a reason to break a live run. Map
  coverage of the tool contract is enforced by tests instead
  (``test_reception_map_covers_all_14_contract_tools``).

Scope is per mission, like the built-in exclusion: ``PLAN_GATE_MISSIONS``
(hotel only, pilot). ``fault_analysis`` keeps its advisory map until its
map is verified against the gate — the mechanism is generic, the rollout
is not. Preconditions between tools (e.g. ``create_task`` requiring a
prior ``list_staff``) are deliberately NOT here: the allowlist already
forces that order for the task flow, and the per-rule choice
(data dependency vs gate vs delete) belongs to Phase 3b.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage

from runtime.skills.map_plan import SkillMap

logger = logging.getLogger(__name__)

# Pilot scope: hotel only. fault_analysis adopts the gate once its
# data-access map is verified against these rules (Phase 3b).
PLAN_GATE_MISSIONS = frozenset({"hotel_reception"})

# Denials past this count in one execution raise instead of returning
# another error message — a run that cannot get back on plan fails with
# a diagnostic, it does not spin until the step limit.
MAX_DENIALS = 5


class PlanGateExceededError(Exception):
    """The agent kept calling out-of-plan tools past the denial cap."""


class _MapState:
    """Gate state for one SkillMap: tool -> node, current node, visited."""

    def __init__(self, skill_map: SkillMap):
        self.map = skill_map
        # First claimant wins, same convention as compute_plan_trace.
        self.tool_to_node: dict[str, str] = {}
        for node in skill_map.nodes:
            for tool in node.tools:
                self.tool_to_node.setdefault(tool, node.id)
        self._index = {node.id: i for i, node in enumerate(skill_map.nodes)}
        self.current = skill_map.start
        self.visited: set[str] = set()

    def _next_tool_node(self, from_node_id: str) -> str | None:
        """First node strictly after ``from_node_id`` that has tools."""
        start = self._index[from_node_id]
        for node in self.map.nodes[start + 1 :]:
            if node.tools:
                return node.id
        return None

    def _expand(self, node_id: str) -> str | None:
        """A branch target resolves to itself when it has tools, else to
        the next tool node after it (reasoning-only nodes are passed
        through, never landed on)."""
        node = self.map.node_by_id(node_id)
        if node is None:
            return None
        if node.tools:
            return node_id
        return self._next_tool_node(node_id)

    def _branch_targets(self, node_id: str) -> set[str]:
        node = self.map.node_by_id(node_id)
        out: set[str] = set()
        for branch in (node.branches if node is not None else []):
            expanded = self._expand(branch.goto)
            if expanded is not None:
                out.add(expanded)
        return out

    def reachable(self) -> set[str]:
        """Tool nodes callable from the current node: the next tool node
        in order, the current node's branch targets, the start node's
        branch targets (re-route), and every visited tool node."""
        out: set[str] = set()
        nxt = self._next_tool_node(self.current)
        if nxt is not None:
            out.add(nxt)
        out |= self._branch_targets(self.current)
        out |= self._branch_targets(self.map.start)
        out |= self.visited
        return out

    def allowed_tools_text(self) -> str:
        """The reachable tool nodes' tools, in map order, as
        ``tool (node)`` pairs — what a denial message names."""
        reachable = self.reachable()
        parts: list[str] = []
        for node in self.map.nodes:
            if node.id in reachable:
                for tool in node.tools:
                    parts.append(f"{tool} ({node.id})")
        # The current node's own tools are always callable too.
        current = self.map.node_by_id(self.current)
        if current is not None:
            for tool in current.tools:
                label = f"{tool} ({current.id})"
                if label not in parts:
                    parts.append(label)
        return ", ".join(parts)


class PlanGate(AgentMiddleware):
    """Deny-by-default gate over the run's skill maps.

    One instance per run (built in ``deep_agent_node``); its state is
    per-execution, so concurrent runs never share a gate. ``denied_count``
    and ``visited_nodes`` are read by the node for the execution notes.
    """

    def __init__(
        self,
        skill_maps: list[SkillMap],
        mission_id: str | None = None,
        execution_id: str = "",
        max_denials: int = MAX_DENIALS,
    ):
        self.skill_maps = list(skill_maps)
        self.mission_id = mission_id
        self.execution_id = execution_id
        self.max_denials = max_denials
        self.denied_count = 0
        self._states = [_MapState(m) for m in self.skill_maps]

    @property
    def visited_nodes(self) -> dict[str, list[str]]:
        return {s.map.skill_name: sorted(s.visited) for s in self._states}

    def _state_for(self, tool_name: str) -> _MapState | None:
        for state in self._states:
            if tool_name in state.tool_to_node:
                return state
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Any],
    ) -> ToolMessage | Any:
        tool_call = request.tool_call or {}
        tool_name = str(tool_call.get("name") or "")
        tool_call_id = str(tool_call.get("id") or "")
        state = self._state_for(tool_name)
        if state is None:
            # No map claims this tool: allow, loudly. Coverage of the
            # tool contract is a test concern, not a runtime one.
            logger.warning(
                "Plan gate: tool %r is in no plan map (mission %s); allowing",
                tool_name, self.mission_id,
            )
            return handler(request)
        node_id = state.tool_to_node[tool_name]
        if node_id == state.current or node_id in state.reachable():
            state.visited.add(node_id)
            state.current = node_id
            return handler(request)
        self.denied_count += 1
        if self.denied_count > self.max_denials:
            raise PlanGateExceededError(
                f"Plan gate: {self.denied_count} out-of-plan tool calls "
                f"(mission {self.mission_id}, execution {self.execution_id or '-'}) "
                f"— last denied: '{tool_name}' (node '{node_id}') while the plan "
                f"is at node '{state.current}'. The agent could not get back on "
                "plan; see the denial messages for the allowed next tools."
            )
        logger.info(
            "Plan gate denied tool call %s (node %s, plan at %s)",
            tool_name, node_id, state.current,
        )
        return ToolMessage(
            content=(
                f"Plan gate: cannot call '{tool_name}' now — it belongs to node "
                f"'{node_id}', but the plan is at node '{state.current}'. "
                f"Call one of: {state.allowed_tools_text()}."
            ),
            tool_call_id=tool_call_id,
            status="error",
        )


def plan_gate(
    skill_maps: list[SkillMap] | None,
    mission_id: str | None,
    execution_id: str = "",
    max_denials: int = MAX_DENIALS,
) -> PlanGate | None:
    """A PlanGate for the run's maps, or None when the mission keeps its
    advisory map (every mission except ``PLAN_GATE_MISSIONS``) or when
    no map was selected."""
    if mission_id not in PLAN_GATE_MISSIONS:
        return None
    maps = [m for m in (skill_maps or []) if m is not None and m.nodes]
    if not maps:
        return None
    return PlanGate(
        maps, mission_id=mission_id, execution_id=execution_id,
        max_denials=max_denials,
    )
