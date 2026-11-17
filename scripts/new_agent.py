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
IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


def validate_agent_name(name: str) -> str:
    raw = name.strip()
    if not raw or raw in (".", "..") or "/" in raw or "\\" in raw:
        raise ValueError("name must be a simple identifier, not a path")
    slug = re.sub(r"[^a-z0-9_]+", "_", raw.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not IDENTIFIER_RE.match(slug):
        raise ValueError(
            "name must normalize to a lowercase identifier "
            "[a-z][a-z0-9_]* (letters, digits, underscores)"
        )
    return slug


def _title(slug: str) -> str:
    return slug.replace("_", " ").title()


def render(template_name: str, **kwargs: str) -> str:
    path = TEMPLATES / template_name
    text = path.read_text(encoding="utf-8")
    return text.format(**kwargs)


def scaffold(
    name: str,
    *,
    capability: str | None = None,
    repo_root: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    root = repo_root or REPO
    agent_id = validate_agent_name(name)
    mission_id = f"{agent_id}_mission"
    skill_slug = agent_id.removesuffix("_bot").replace("_", "-")
    cap = capability or agent_id.removesuffix("_bot")
    ctx = {
        "agent_id": agent_id,
        "agent_title": _title(agent_id),
        "mission_id": mission_id,
        "mission_title": _title(mission_id),
        "mission_objective": f"Run the {_title(agent_id)} mission.",
        "skill_slug": skill_slug,
        "skill_title": _title(skill_slug),
        "capability": cap,
        "skill_keywords": f"{skill_slug} {agent_id} mission",
    }
    agents_dir = root / "seeds" / "agents"
    missions_dir = root / "seeds" / "missions"
    skill_dir = root / "skills" / skill_slug
    agent_path = agents_dir / f"{agent_id}.yaml"
    mission_path = missions_dir / f"{mission_id}.yaml"
    skill_path = skill_dir / "SKILL.md"
    if not force:
        if agent_path.exists() or mission_path.exists():
            raise FileExistsError(
                f"seed files already exist for {agent_id!r} — use --force or remove them"
            )
        if skill_path.exists() or (skill_dir.exists() and any(skill_dir.iterdir())):
            raise FileExistsError(
                f"skill directory already exists: {skill_dir.relative_to(root)}"
            )
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    mission_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing seed/skill files",
    )
    args = parser.parse_args(argv)
    try:
        paths = scaffold(args.name, capability=args.capability, force=args.force)
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