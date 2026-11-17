"""Scaffold a new agent + mission + skill skeleton from templates.

Usage:
    python -m scripts.new_agent my_agent
    make new-agent NAME=my_agent
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "seeds" / "templates"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("name must contain at least one letter or digit")
    return s


def _title(slug: str) -> str:
    return slug.replace("_", " ").title()


def render(template_name: str, **kwargs: str) -> str:
    path = TEMPLATES / template_name
    text = path.read_text(encoding="utf-8")
    return text.format(**kwargs)


def scaffold(name: str, *, capability: str | None = None) -> dict[str, Path]:
    agent_id = _slug(name)
    if agent_id in ("hotel_receptionist", "admin_bot", "fault_investigator"):
        raise ValueError(f"refusing to overwrite built-in agent {agent_id!r}")
    mission_id = f"{agent_id}_mission"
    skill_slug = agent_id.replace("_bot", "").replace("_", "-")
    cap = capability or agent_id.replace("_bot", "")
    ctx = {
        "agent_id": agent_id,
        "agent_title": _title(agent_id),
        "mission_id": mission_id,
        "mission_title": _title(mission_id),
        "mission_objective": f"Run the {_title(agent_id)} mission.",
        "skill_slug": skill_slug,
        "skill_title": _title(skill_slug),
        "capability": cap,
    }
    agents_dir = REPO / "seeds" / "agents"
    missions_dir = REPO / "seeds" / "missions"
    skill_dir = REPO / "skills" / skill_slug
    agent_path = agents_dir / f"{agent_id}.yaml"
    mission_path = missions_dir / f"{mission_id}.yaml"
    skill_path = skill_dir / "SKILL.md"
    if agent_path.exists() or mission_path.exists():
        raise FileExistsError(
            f"seed files already exist for {agent_id!r} — remove them first"
        )
    agent_path.write_text(render("agent.yaml.tpl", **ctx), encoding="utf-8")
    mission_path.write_text(render("mission.yaml.tpl", **ctx), encoding="utf-8")
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(render("SKILL.md.tpl", **ctx), encoding="utf-8")
    return {
        "agent": agent_path,
        "mission": mission_path,
        "skill": skill_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="agent slug, e.g. concierge_bot")
    parser.add_argument(
        "--capability",
        help="registry capability name (default: derived from agent name)",
    )
    args = parser.parse_args(argv)
    try:
        paths = scaffold(args.name, capability=args.capability)
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Created:")
    for key, path in paths.items():
        print(f"  {key}: {path.relative_to(REPO)}")
    print(
        "\nNext: add tools to allowed_tools, set harness overrides if needed, "
        "then run:\n"
        f"  make seed-agent AGENT={paths['agent'].stem} "
        f"MISSION={paths['mission'].stem}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())