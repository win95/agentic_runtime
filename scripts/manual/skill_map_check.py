"""Confirm the Deep Agent runs per its skill maps — through the real API.

Two throwaway autonomous missions are created, run, and deleted again
(the API refuses to delete a mission that has executions, so the rows are
removed straight from Postgres):

- ``map``     — ``skill_map_test``, ``skills=["data-access"]``. The skill
  declares a ``map:`` in ``skills/data-access/SKILL.md``, so the Deep Agent
  gets the rendered plan and the result must carry
  ``plan_trace["data-access"]``. The trace is checked against the local
  map for **structural** consistency (the deterministic contract):
    * ``visited`` and ``skipped`` partition the map nodes, no overlap;
    * ``tools_by_node`` only names visited nodes and only that node's tools;
    * ``deviations`` are tools outside the map;
    * every successful ``actions_taken`` tool appears in the trace.
  Separately, **conformance** is reported: how many tool-bearing nodes the
  agent actually visited, in map order, and how many deviations it made.
  The local small model usually deviates (``read_file``/``task``), so
  conformance only fails the run with ``--strict``.
- ``mapless`` — ``skill_nomap_test``, ``skills=["fault-investigation"]``.
  That skill has no ``map:``, so the result must NOT contain ``plan_trace``:
  map-less skills behave exactly as before.

``identify`` and ``synthesize`` have no tools, so they are never "visited"
in the trace by design — they are reasoning-only steps and are reported
separately, not counted as skipped work.

Usage (from the repo root, with the API + worker running):

    python scripts/manual/skill_map_check.py
    python scripts/manual/skill_map_check.py --case map
    python scripts/manual/skill_map_check.py --case mapless
    python scripts/manual/skill_map_check.py --case map --strict
    python scripts/manual/skill_map_check.py --keep   # leave the missions

Exit code: 0 when the structural contract holds (and, with ``--strict``,
when the agent followed the plan); 1 otherwise. The script needs the
project venv (SQLAlchemy cleanup imports app.* and the map ground truth
comes from runtime.skills).
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
    print_result,
    require_worker,
    submit,
    wait_for,
)

SKILL_NAME = "data-access"
MAP_MISSION = "skill_map_test"
NOMAP_MISSION = "skill_nomap_test"
MISSION_IDS = [MAP_MISSION, NOMAP_MISSION]
O21_MONTH = {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"}

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "asset_id": {"type": "string"},
        "start_date": {"type": "string", "format": "date"},
        "end_date": {"type": "string", "format": "date"},
    },
    "required": ["asset_id", "start_date", "end_date"],
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "root_causes": {"type": "array"},
        "evidence": {"type": "array"},
        "recommendations": {"type": "array"},
        # Not a product output — exposed so this script can prove every
        # successful tool call is accounted for in the plan trace.
        "actions_taken": {"type": "array"},
    },
}

CAPABILITIES = [
    "fault_history",
    "maintenance_analysis",
    "knowledge_retrieval",
    "sensor_data",
    "asset_relations",
]


def mission_payload(mission_id: str, name: str, skills: list[str]) -> dict:
    """Autonomous clone of ``fault_analysis`` with a fixed skill list."""
    return {
        "id": mission_id,
        "name": name,
        "objective": "Analyze equipment faults and identify root causes.",
        "input_schema": INPUT_SCHEMA,
        "output_schema": OUTPUT_SCHEMA,
        "execution_config": {
            "mode": "autonomous",
            "agent_id": "fault_investigator",
            "skills": skills,
        },
        "capabilities": CAPABILITIES,
        "agent_id": "fault_investigator",
    }


def cleanup(mission_ids: list[str]) -> None:
    """Delete the temp missions' executions/results and mission rows."""
    from app import models
    from app.core.db import SessionLocal

    db = SessionLocal()
    try:
        for mission_id in mission_ids:
            execution_ids = [
                row.id for row in
                db.query(models.Execution).filter_by(mission_id=mission_id)
            ]
            if execution_ids:
                db.query(models.ExecutionResult).filter(
                    models.ExecutionResult.execution_id.in_(execution_ids)
                ).delete(synchronize_session=False)
            executions = db.query(models.Execution).filter_by(
                mission_id=mission_id
            ).delete(synchronize_session=False)
            missions = db.query(models.Mission).filter_by(id=mission_id).delete(
                synchronize_session=False
            )
            if executions or missions:
                print(f"cleanup: removed mission {mission_id} "
                      f"and {executions} execution(s)")
        db.commit()
    finally:
        db.close()


def create_mission(mission: dict) -> bool:
    status, payload = call("POST", "/missions", mission)
    if status == 409:
        print(f"mission {mission['id']} already exists — run without --keep "
              f"to clean it up first, or delete it by hand")
        return False
    if status != 201:
        print(f"create mission {mission['id']} -> HTTP {status}: {payload}")
        return False
    print(f"created mission {mission['id']} "
          f"(mode=autonomous, skills={mission['execution_config']['skills']})")
    return True


def load_pilot_map():
    """Ground truth: the map the runtime itself loads from the repo."""
    from runtime.skills import load_skills

    skills = {skill.name: skill for skill in load_skills()}
    skill = skills.get(SKILL_NAME)
    if skill is None:
        raise SystemExit(f"skill {SKILL_NAME!r} not found under skills/")
    if skill.map is None:
        raise SystemExit(
            f"skill {SKILL_NAME!r} has no map: block — nothing to check"
        )
    return skill.map


def run_mission(mission_id: str, timeout: float) -> tuple[dict, dict, dict]:
    """Submit, wait, and return (execution, result-payload, output dict)."""
    execution = submit(mission_id, O21_MONTH)
    execution_id = execution["execution_id"]
    print(f"  execution_id={execution_id} status={execution['status']}")
    execution, elapsed = wait_for(execution_id, timeout=timeout)
    print(f"  final after {elapsed:.1f}s")
    result = get_result(execution_id)
    output = result.get("output") or {}
    print_result(execution, result)
    return execution, result, output


def check_trace_structure(trace: dict, skill_map) -> list[str]:
    """Deterministic contract checks; returns a list of violations."""
    problems: list[str] = []
    node_ids = skill_map.node_ids
    known = set(node_ids)

    for key in ("visited", "tools_by_node", "skipped", "deviations"):
        if key not in trace:
            problems.append(f"trace is missing {key!r}")
    if problems:
        return problems

    visited = trace["visited"]
    skipped = trace["skipped"]
    tools_by_node = trace["tools_by_node"]
    deviations = trace["deviations"]
    if not isinstance(visited, list) or not isinstance(skipped, list):
        return ["visited/skipped must be lists"]
    if not isinstance(tools_by_node, dict) or not isinstance(deviations, list):
        return ["tools_by_node must be an object and deviations a list"]
    if len(set(visited)) != len(visited):
        problems.append(f"visited has duplicates: {visited}")
    if len(set(skipped)) != len(skipped):
        problems.append(f"skipped has duplicates: {skipped}")
    if set(visited) & set(skipped):
        problems.append(
            f"nodes both visited and skipped: {sorted(set(visited) & set(skipped))}"
        )
    unknown = (set(visited) | set(skipped)) - known
    if unknown:
        problems.append(f"trace names nodes outside the map: {sorted(unknown)}")
    missing = known - (set(visited) | set(skipped))
    if missing:
        problems.append(f"trace omits map nodes: {sorted(missing)}")

    all_map_tools = {tool for node in skill_map.nodes for tool in node.tools}
    for node_id, tools in tools_by_node.items():
        if node_id not in visited:
            problems.append(f"tools_by_node has unvisited node {node_id!r}")
            continue
        if node_id not in known:
            problems.append(f"tools_by_node has unknown node {node_id!r}")
            continue
        node = skill_map.node_by_id(node_id)
        extra = set(tools) - set(node.tools)
        if extra:
            problems.append(
                f"node {node_id!r} traced tools it does not declare: {sorted(extra)}"
            )
        if not tools:
            problems.append(f"visited node {node_id!r} has no traced tool")

    for deviation in deviations:
        tool = deviation.get("tool") if isinstance(deviation, dict) else None
        if not tool:
            problems.append(f"malformed deviation entry: {deviation!r}")
            continue
        if tool in all_map_tools:
            problems.append(
                f"deviation {tool!r} actually belongs to the map"
            )

    return problems


def check_actions_in_trace(trace: dict, actions_taken: list) -> list[str]:
    """Every successful tool call must be classified by the trace."""
    traced = {tool for tools in trace.get("tools_by_node", {}).values()
              for tool in tools}
    traced |= {
        deviation.get("tool") for deviation in trace.get("deviations", [])
        if isinstance(deviation, dict) and deviation.get("tool")
    }
    return [
        f"successful action {action!r} is missing from the trace"
        for action in actions_taken or []
        if action not in traced
    ]


def report_trace(trace: dict, skill_map, actions_taken: list,
                 structural: list[str], strict: bool) -> bool:
    """Print the trace, its verdict, and return True when the case passes."""
    node_ids = skill_map.node_ids
    visited = trace.get("visited") or []
    skipped = trace.get("skipped") or []
    tools_by_node = trace.get("tools_by_node") or {}
    deviations = trace.get("deviations") or []
    tool_nodes = [node.id for node in skill_map.nodes if node.tools]
    visited_tool_nodes = [node_id for node_id in tool_nodes if node_id in visited]

    order = [node_ids.index(node_id) for node_id in visited if node_id in node_ids]
    in_map_order = order == sorted(order)

    print(f"  plan: {SKILL_NAME} ({len(node_ids)} nodes, "
          f"{len(tool_nodes)} with tools)")
    print(f"  map order: {' -> '.join(node_ids)}")
    print(f"  visited: {' -> '.join(visited) if visited else '(none)'}")
    print(f"  skipped: {', '.join(skipped) if skipped else '(none)'}")
    print("  per node: " + (", ".join(
        f"{node_id}=[{', '.join(tools_by_node.get(node_id, []))}]"
        for node_id in visited
    ) or "(none)"))
    print("  deviations: " + (", ".join(
        str(deviation.get("tool")) for deviation in deviations
    ) or "(none)"))
    print(f"  actions_taken: {json.dumps(actions_taken, ensure_ascii=False)}")

    if structural:
        print("  structural invariants: FAIL")
        for problem in structural:
            print(f"    - {problem}")
        return False
    print("  structural invariants: PASS")

    deviations_count = len(deviations)
    complete = len(visited_tool_nodes) == len(tool_nodes)
    followed = complete and not deviations and in_map_order
    if followed:
        print(f"  plan conformance: FOLLOWED "
              f"({len(visited_tool_nodes)}/{len(tool_nodes)} tool nodes, "
              f"0 deviations)")
    else:
        reasons = [
            f"{len(visited_tool_nodes)}/{len(tool_nodes)} tool nodes visited",
            f"{deviations_count} deviation(s)",
        ]
        if not in_map_order:
            reasons.append("visited out of map order")
        print("  plan conformance: NOT FOLLOWED (" + ", ".join(reasons) + ")")

    if strict and not followed:
        print("  strict: FAIL — the agent did not follow the plan")
        return False
    return True


def run_map_case(timeout: float, strict: bool) -> bool:
    print(f"\n=== map: skill '{SKILL_NAME}' becomes a plan + plan_trace")
    skill_map = load_pilot_map()
    if not create_mission(mission_payload(
        MAP_MISSION, "Skill Map Test", [SKILL_NAME]
    )):
        return False
    execution, _result, output = run_mission(MAP_MISSION, timeout)
    if execution["status"] != "SUCCEEDED":
        print(f"  FAIL: execution ended {execution['status']} "
              f"(error={execution.get('error')})")
        return False

    trace_by_skill = output.get("plan_trace")
    if not isinstance(trace_by_skill, dict):
        print("  FAIL: result has no plan_trace object")
        return False
    trace = trace_by_skill.get(SKILL_NAME)
    if not isinstance(trace, dict):
        print(f"  FAIL: plan_trace has no {SKILL_NAME!r} entry "
              f"(keys={sorted(trace_by_skill)})")
        return False

    structural = check_trace_structure(trace, skill_map)
    structural += check_actions_in_trace(trace, output.get("actions_taken") or [])
    return report_trace(
        trace, skill_map, output.get("actions_taken") or [], structural, strict
    )


def run_mapless_case(timeout: float) -> bool:
    print("\n=== mapless: a skill without a map stays unchanged")
    if not create_mission(mission_payload(
        NOMAP_MISSION, "Skill No-Map Test", ["fault-investigation"]
    )):
        return False
    execution, _result, output = run_mission(NOMAP_MISSION, timeout)
    if execution["status"] != "SUCCEEDED":
        print(f"  FAIL: execution ended {execution['status']} "
              f"(error={execution.get('error')})")
        return False
    if "plan_trace" in output:
        print(f"  FAIL: mapless skill produced plan_trace: "
              f"{json.dumps(output['plan_trace'], ensure_ascii=False)}")
        return False
    print("  plan_trace: absent (expected — skill has no map)")
    print("  PASS")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case", default="all", choices=["all", "map", "mapless"],
                        help="which scenario to run (default: all)")
    parser.add_argument("--strict", action="store_true",
                        help="the map case also requires full plan conformance")
    parser.add_argument("--keep", action="store_true",
                        help="keep the temp missions and execution rows")
    parser.add_argument("--timeout", type=float, default=600,
                        help="seconds to wait per run (default 600)")
    args = parser.parse_args()

    require_worker()
    if not args.keep:
        cleanup(MISSION_IDS)

    passed = True
    try:
        if args.case in ("all", "map"):
            passed = run_map_case(args.timeout, args.strict) and passed
        if args.case in ("all", "mapless"):
            passed = run_mapless_case(args.timeout) and passed
    finally:
        if not args.keep:
            cleanup(MISSION_IDS)

    print(f"\n{'PASS' if passed else 'FAIL'}: skill-map check")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
