"""requirements.txt must list every pyproject core dependency (Docker --no-deps)."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
REQUIREMENTS = REPO / "requirements.txt"


def _normalize_pkg(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _package_from_requirement_line(line: str) -> str | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    head = re.split(r"[<>=!;\[]", text, maxsplit=1)[0].strip()
    if not head:
        return None
    return _normalize_pkg(head)


def _package_from_pyproject_spec(spec: str) -> str:
    text = spec.strip()
    head = re.split(r"[<>=!;\[]", text, maxsplit=1)[0].strip()
    return _normalize_pkg(head)


def _pinned_requirement_names() -> set[str]:
    names: set[str] = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        pkg = _package_from_requirement_line(line)
        if pkg:
            names.add(pkg)
    return names


def _pyproject_core_dependencies() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    raw = data.get("project", {}).get("dependencies") or []
    return [_package_from_pyproject_spec(str(spec)) for spec in raw]


def test_pyproject_dependencies_pinned_in_requirements_txt():
    """Docker installs with ``pip install --no-deps -e .`` — freeze must be complete."""
    pinned = _pinned_requirement_names()
    missing = [dep for dep in _pyproject_core_dependencies() if dep not in pinned]
    assert not missing, (
        "requirements.txt missing packages from pyproject [project].dependencies: "
        + ", ".join(sorted(missing))
    )