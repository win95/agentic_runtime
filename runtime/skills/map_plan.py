"""Skill maps — a skill's guidance as a plan the Deep Agent follows.

Option C (tran 2026-11-11): the map is a PLAN, not a program. The agent
receives the rendered plan in its instructions, may deviate when the
evidence justifies it (and must say why), and at the end reports the
nodes it visited. The trace is computed deterministically from the run's
tool calls — no LLM cooperation is needed to audit it.

A map lives in the skill's frontmatter under ``map:``::

    map:
      start: identify
      nodes:
        - id: identify
          goal: Parse asset_id and date range from the input.
          tools: []
          branches:
            - when: no faults found
              goto: postgres-maintenance
          stop_when: asset_id missing or invalid

Rules: the default flow is the next node in list order; a branch is a
jump taken when its ``when`` condition holds (plain words — the agent
decides). A node with no tools is a reasoning-only step. Unknown tool
names and unknown ``goto`` targets are dropped with a warning, never
fatal; a skill without a map works exactly as before (``map=None``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MapBranch:
    """One conditional jump: when ``when`` holds, go to ``goto``."""

    when: str
    goto: str


@dataclass
class MapNode:
    """One step of the plan: what to achieve, with which tools.

    ``terminal`` marks a flow-ending node: the router honors a
    ``NEXT: END`` declaration only there. At any other node END means
    the model judged the request handled mid-flow (e.g. task-details
    after merely restating the issue) — the router falls back to the
    next node in order instead. A node that needs more information
    from the guest ends the turn with ``NEXT: WAIT`` (always honored).
    """

    id: str
    goal: str = ""
    tools: list[str] = field(default_factory=list)
    branches: list[MapBranch] = field(default_factory=list)
    stop_when: str = ""
    terminal: bool = False


@dataclass
class SkillMap:
    """A parsed skill map: an ordered node list plus its start node."""

    skill_name: str
    start: str
    nodes: list[MapNode] = field(default_factory=list)

    @property
    def node_ids(self) -> list[str]:
        return [node.id for node in self.nodes]

    def node_by_id(self, node_id: str) -> MapNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None


def parse_skill_map(
    frontmatter: dict | None,
    skill_name: str,
    known_tools: set[str] | None = None,
) -> SkillMap | None:
    """Parse the ``map:`` frontmatter block into a SkillMap, or None.

    Never raises: malformed maps log a warning and return None (the
    skill then works exactly as before, map-less). Unknown tool names
    are dropped with a warning when ``known_tools`` is given; unknown
    ``goto`` targets drop their branch. Unknown extra keys are ignored
    with a debug log. ``known_tools=None`` means "no tool validation"
    (accept every named tool).
    """
    if not isinstance(frontmatter, dict):
        return None
    raw = frontmatter.get("map")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning("Skill %s: 'map' must be a mapping; ignoring", skill_name)
        return None
    for key in raw:
        if key not in {"start", "nodes"}:
            logger.debug("Skill %s: ignoring unknown map key %r", skill_name, key)
    nodes_raw = raw.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        logger.warning("Skill %s: 'map.nodes' must be a non-empty list; ignoring", skill_name)
        return None
    nodes: list[MapNode] = []
    seen_ids: set[str] = set()
    for entry in nodes_raw:
        if not isinstance(entry, dict):
            logger.warning("Skill %s: map node must be a mapping; skipping %r", skill_name, entry)
            continue
        node_id = entry.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            logger.warning("Skill %s: map node without a non-empty 'id'; skipping", skill_name)
            continue
        node_id = node_id.strip()
        if node_id in seen_ids:
            logger.warning(
                "Skill %s: duplicate map node id %r; keeping the first",
                skill_name, node_id,
            )
            continue
        seen_ids.add(node_id)
        for key in entry:
            if key not in {"id", "goal", "tools", "branches", "stop_when", "terminal"}:
                logger.debug("Skill %s: ignoring unknown map node key %r", skill_name, key)
        goal = entry.get("goal")
        goal = str(goal).strip() if goal is not None else ""
        tools: list[str] = []
        tools_raw = entry.get("tools") or []
        if not isinstance(tools_raw, list):
            logger.warning(
                "Skill %s: node %r 'tools' must be a list; ignoring",
                skill_name, node_id,
            )
            tools_raw = []
        for tool in tools_raw:
            if not isinstance(tool, str) or not tool.strip():
                logger.warning(
                    "Skill %s: node %r has a non-string tool; dropping",
                    skill_name, node_id,
                )
                continue
            tool = tool.strip()
            if known_tools is not None and tool not in known_tools:
                logger.warning(
                    "Skill %s: node %r names unknown tool %r; dropping it",
                    skill_name, node_id, tool,
                )
                continue
            if tool not in tools:
                tools.append(tool)
        branches: list[MapBranch] = []
        branches_raw = entry.get("branches") or []
        if not isinstance(branches_raw, list):
            logger.warning(
                "Skill %s: node %r 'branches' must be a list; ignoring",
                skill_name, node_id,
            )
            branches_raw = []
        for branch in branches_raw:
            if not isinstance(branch, dict):
                logger.warning(
                    "Skill %s: node %r branch must be a mapping; dropping",
                    skill_name, node_id,
                )
                continue
            when = branch.get("when")
            goto = branch.get("goto")
            if not isinstance(when, str) or not when.strip():
                logger.warning(
                    "Skill %s: node %r branch without 'when'; dropping",
                    skill_name, node_id,
                )
                continue
            if not isinstance(goto, str) or not goto.strip():
                logger.warning(
                    "Skill %s: node %r branch without 'goto'; dropping",
                    skill_name, node_id,
                )
                continue
            branches.append(MapBranch(when=when.strip(), goto=goto.strip()))
        stop_when = entry.get("stop_when")
        stop_when = str(stop_when).strip() if stop_when is not None else ""
        terminal = entry.get("terminal") is True
        nodes.append(
            MapNode(id=node_id, goal=goal, tools=tools, branches=branches,
                    stop_when=stop_when, terminal=terminal)
        )
    if not nodes:
        logger.warning("Skill %s: map has no usable nodes; ignoring", skill_name)
        return None
    # Second pass: a branch's goto must name a real node.
    for node in nodes:
        kept: list[MapBranch] = []
        for branch in node.branches:
            if branch.goto not in seen_ids:
                logger.warning(
                    "Skill %s: node %r branch targets unknown node %r; dropping it",
                    skill_name, node.id, branch.goto,
                )
                continue
            kept.append(branch)
        node.branches = kept
    start = raw.get("start")
    if not isinstance(start, str) or start.strip() not in seen_ids:
        if start is not None:
            logger.warning(
                "Skill %s: map 'start' %r is not a node id; using %r",
                skill_name, start, nodes[0].id,
            )
        start = nodes[0].id
    else:
        start = start.strip()
    return SkillMap(skill_name=skill_name, start=start, nodes=nodes)


def render_plan(skill_map: SkillMap, enforced: bool = False) -> str:
    """Render a map as a compact plan block for the agent's instructions.

    One line per node: id, goal, tools, branches as ``if <when> -> <goto>``,
    and stop conditions. The header states the Option C contract: follow
    the plan, deviate only with a stated reason, report visited nodes and
    deviations at the end. With ``enforced=True`` (Phase 3a, the plan gate
    is active for this mission) the header says so instead: tool order is
    refused at the call boundary, so advisory-deviation text would be
    incoherent next to a hard gate.
    """
    if enforced:
        header = (
            f"Plan — skill '{skill_map.skill_name}'. This plan is ENFORCED by "
            "the plan gate: a tool call outside the current step is refused, "
            "and the refusal names the tools you may call next. Follow the "
            "nodes in order; re-route only through the start node's branches. "
            "At the end, report the plan nodes you visited."
        )
    else:
        header = (
            f"Plan — skill '{skill_map.skill_name}'. Follow this plan; you may deviate "
            "when the evidence justifies it, but state why in your summary. At the end, "
            "report the plan nodes you visited and any deviations."
        )
    lines = [
        header,
        f"Start at '{skill_map.start}'. Default flow is the next node in order.",
    ]
    for node in skill_map.nodes:
        tools = ", ".join(node.tools) if node.tools else "none"
        line = f"- {node.id}: {node.goal} [tools: {tools}]"
        for branch in node.branches:
            line += f" — if {branch.when}, go to {branch.goto}"
        if node.stop_when:
            line += f" — stop when: {node.stop_when}"
        lines.append(line)
    return "\n".join(lines)


def compute_plan_trace(skill_map: SkillMap, called_tools: list[str]) -> dict:
    """Trace a run against a map, deterministically from tool calls.

    ``called_tools`` is the run's tool-call sequence (in call order, may
    repeat). A node counts as visited when any of its tools was called;
    a tool claimed by two nodes counts for the first node in map order.
    Deviations are tools that belong to no node of this map. Failed
    calls still count — the agent attempted the node.
    """
    ordered = list(dict.fromkeys(called_tools))
    tool_to_node: dict[str, str] = {}
    for node in skill_map.nodes:
        for tool in node.tools:
            tool_to_node.setdefault(tool, node.id)
    visited: list[str] = []
    tools_by_node: dict[str, list[str]] = {}
    deviations: list[dict] = []
    for tool in ordered:
        node_id = tool_to_node.get(tool)
        if node_id is None:
            deviations.append({"tool": tool})
            continue
        if node_id not in visited:
            visited.append(node_id)
        tools_by_node.setdefault(node_id, []).append(tool)
    skipped = [node.id for node in skill_map.nodes if node.id not in visited]
    return {
        "visited": visited,
        "tools_by_node": tools_by_node,
        "skipped": skipped,
        "deviations": deviations,
    }


def plan_trace_note(skill_name: str, trace: dict) -> str:
    """One-line human summary of a plan trace, for execution notes."""
    parts: list[str] = []
    visited = trace.get("visited") or []
    if visited:
        parts.append("visited " + " -> ".join(str(v) for v in visited))
    skipped = trace.get("skipped") or []
    if skipped:
        parts.append("skipped " + ", ".join(str(s) for s in skipped))
    deviations = trace.get("deviations") or []
    if deviations:
        parts.append(
            "deviation: "
            + ", ".join(f"{d.get('tool')} (not in map)" for d in deviations)
        )
    if not parts:
        return f"Plan ({skill_name}): no tools called"
    return f"Plan ({skill_name}): " + "; ".join(parts)
