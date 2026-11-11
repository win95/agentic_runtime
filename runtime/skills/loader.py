"""Skills — task playbooks the Deep Agent loads when relevant (Layer 4).

Self-hosted equivalent of the "skills" piece in LangSmith Managed Deep
Agents. Skills live as ``skills/<name>/SKILL.md`` files (YAML frontmatter
+ markdown playbook, per the Agent Skills spec). Selection is
deterministic: explicit mission-declared skills win; otherwise keyword
scoring over the mission + request text with a threshold. No LLM call is
made just to pick skills.

Delivery uses the native ``deepagents`` mechanism (verified against the
installed 0.7.34 package with a recording fake model):
``create_deep_agent(skills=["/skills/"])`` adds ``SkillsMiddleware`` over
the default ``StateBackend`` (in-memory — no real filesystem access is
granted); the selected skill files are handed to the agent via
``invoke(files={...})`` and the middleware progressively discloses them
(skill index in the system prompt, full text on read).

A skill may also carry a ``map:`` frontmatter block (see
``runtime/skills/map_plan.py``): a plan the Deep Agent follows, rendered
into its instructions and traced deterministically from its tool calls.
Skills without a map work exactly as before (``skill.map is None``).
"""

from __future__ import annotations

import logging
import pathlib
import re

import yaml

from runtime.skills.map_plan import SkillMap, parse_skill_map

logger = logging.getLogger(__name__)

# Backend path the native SkillsMiddleware reads from (see module docstring).
SKILLS_SOURCE = "/skills/"

# A skill is selected by keyword matching when this many of its distinct
# keywords appear in the mission + request text.
DEFAULT_THRESHOLD = 1

# Repo-root ``skills/`` directory (this file lives in runtime/skills/).
DEFAULT_SKILLS_DIR = pathlib.Path(__file__).resolve().parents[2] / "skills"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _normalize_token(token: str) -> str:
    """Lowercase singular form: "executions" -> "execution"."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens of ``text``, singularized, length >= 2."""
    return {
        _normalize_token(t) for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2
    }


class Skill:
    """One parsed skill: frontmatter metadata + full file text.

    ``map`` is the parsed ``map:`` frontmatter block (``SkillMap``) when
    the skill declares one, else None — map-less skills are unaffected.
    """

    def __init__(self, name: str, description: str, keywords: list[str], raw: str,
                 map: SkillMap | None = None):
        self.name = name
        self.description = description
        self.keywords = keywords
        self.raw = raw
        self.map = map

    @property
    def body(self) -> str:
        """Playbook markdown without the frontmatter block."""
        match = _FRONTMATTER_RE.match(self.raw)
        return self.raw[match.end():] if match else self.raw

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Skill(name={self.name!r}, keywords={self.keywords!r})"


def _parse_keywords(value: object) -> list[str]:
    """Frontmatter ``keywords`` as a list or a space/comma-separated string."""
    if value is None:
        return []
    if isinstance(value, list):
        raw = [str(v) for v in value]
    else:
        raw = re.split(r"[\s,]+", str(value))
    seen: list[str] = []
    for token in _tokens(" ".join(raw)):
        if token not in seen:
            seen.append(token)
    return seen


_KNOWN_TOOLS_CACHE: set[str] | None = None
_KNOWN_TOOLS_TRIED = False


def _known_tool_names() -> set[str] | None:
    """Registered tool names for map validation, best-effort.

    Importing the tools package registers the built-ins (and any MCP
    tools already loaded by the caller) in the global registry. When
    the import fails — e.g. a test process without the infra stack —
    return None and skip tool validation (structure is still checked).
    Tried once per process; a later caller that needs validation reuses
    the cached set.
    """
    global _KNOWN_TOOLS_CACHE, _KNOWN_TOOLS_TRIED
    if _KNOWN_TOOLS_TRIED:
        return _KNOWN_TOOLS_CACHE
    _KNOWN_TOOLS_TRIED = True
    try:
        import runtime.tools  # noqa: F401  (registers built-in tools)
        from runtime.tools.registry import registry

        _KNOWN_TOOLS_CACHE = {entry.name for entry in registry.list()}
    except Exception as exc:  # noqa: BLE001 — validation is best-effort
        logger.debug("Skill map tool validation skipped: %s", exc)
        _KNOWN_TOOLS_CACHE = None
    return _KNOWN_TOOLS_CACHE


def _parse_skill_file(path: pathlib.Path) -> Skill | None:
    """Parse one ``SKILL.md``; None (with a warning) when unusable."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Skipping skill %s: unreadable (%s)", path, exc)
        return None
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        logger.warning("Skipping skill %s: no YAML frontmatter found", path)
        return None
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        logger.warning("Skipping skill %s: invalid YAML (%s)", path, exc)
        return None
    if not isinstance(frontmatter, dict):
        logger.warning("Skipping skill %s: frontmatter is not a mapping", path)
        return None
    name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()
    if not name or not description:
        logger.warning("Skipping skill %s: missing required 'name'/'description'", path)
        return None
    return Skill(
        name=name,
        description=description,
        keywords=_parse_keywords(frontmatter.get("keywords")),
        raw=raw,
        map=parse_skill_map(frontmatter, name, known_tools=_known_tool_names()),
    )


def load_skills(skills_dir: pathlib.Path | None = None) -> list[Skill]:
    """Load every ``*/SKILL.md`` under ``skills_dir`` (sorted by name).

    Invalid files are skipped with a warning, never raised — a broken
    skill must not fail a run. A missing directory loads as zero skills.
    """
    directory = pathlib.Path(skills_dir) if skills_dir is not None else DEFAULT_SKILLS_DIR
    if not directory.is_dir():
        logger.debug("Skills directory %s not found; no skills loaded", directory)
        return []
    skills: list[Skill] = []
    for path in sorted(directory.glob("*/SKILL.md")):
        skill = _parse_skill_file(path)
        if skill is not None:
            skills.append(skill)
    skills.sort(key=lambda s: s.name)
    return skills


def _match_text(mission: dict, input_data: dict) -> str:
    """Text the keyword matcher reads: mission identity + request."""
    parts = [
        mission.get("mission_id") or "",
        mission.get("objective") or "",
        " ".join(mission.get("capabilities") or []),
        " ".join(str(v) for v in (input_data or {}).values()),
    ]
    return " ".join(parts)


def select_skills(
    mission: dict,
    input_data: dict | None = None,
    skills: list[Skill] | None = None,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[Skill]:
    """Select skills for one run, deterministically.

    Explicit declaration wins: a mission listing skills in
    ``mission["skills"]`` or ``mission["execution"]["skills"]`` gets
    exactly those (unknown names are skipped with a warning). Otherwise
    keyword scoring: a skill is selected when at least ``threshold`` of
    its distinct keywords appear in the mission + request text; ties
    break by score, then name. Selection is logged at debug level with
    the per-skill scores so it is explainable without an LLM call.
    """
    available = skills if skills is not None else load_skills()
    by_name = {skill.name: skill for skill in available}
    declared = mission.get("skills") or (mission.get("execution") or {}).get("skills") or []
    if declared:
        selected: list[Skill] = []
        for name in declared:
            skill = by_name.get(str(name))
            if skill is None:
                logger.warning(
                    "Mission %s declares unknown skill %r; skipping",
                    mission.get("mission_id"),
                    name,
                )
                continue
            selected.append(skill)
        logger.debug(
            "Skills for mission %s (declared): %s",
            mission.get("mission_id"),
            [s.name for s in selected],
        )
        return selected
    tokens = _tokens(_match_text(mission, input_data or {}))
    scored = [
        (skill, len(set(skill.keywords) & tokens)) for skill in available
    ]
    scored.sort(key=lambda item: (-item[1], item[0].name))
    selected = [skill for skill, score in scored if score >= threshold]
    logger.debug(
        "Skills for mission %s (matched): %s (scores=%s)",
        mission.get("mission_id"),
        [s.name for s in selected],
        {skill.name: score for skill, score in scored},
    )
    return selected


def skill_files_for_run(selected: list[Skill]) -> dict:
    """``files`` payload for ``agent.invoke``: the selected SKILL.md
    contents, keyed by their backend paths (``/skills/<name>/SKILL.md``).

    Values are ``FileData`` dicts (``{"content", "encoding": "utf-8"}``),
    which is what the default ``StateBackend`` stores.
    """
    return {
        f"{SKILLS_SOURCE}{skill.name}/SKILL.md": {
            "content": skill.raw,
            "encoding": "utf-8",
        }
        for skill in selected
    }
