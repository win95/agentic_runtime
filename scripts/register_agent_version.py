"""Register (deploy) an agent version (Option C).

Snapshots an agent's full config — system prompt, model, temperature,
allowed tools, and the tool manifest — as a new immutable
AgentVersion, and activates it (unless --no-activate). A later change
to the agent YAML (or to a tool's description) produces a new version;
rollback is activating an older one:

    python scripts/register_agent_version.py --agent-id admin_bot
    python scripts/register_agent_version.py --all
    python scripts/register_agent_version.py --agent-id fault_investigator --no-activate

Options:
  --prompt-text-file PATH  use this file's text as the prompt instead
                           of the YAML's system_prompt.
  --register-prompt        also register the prompt in the MLflow Prompt
                           Registry first (via scripts/register_prompts.py)
                           and record its version on the snapshot.

The agent row is re-upserted from its YAML first, so editing the YAML
and running this script is the whole deploy flow.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.db import SessionLocal, init_db  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[1] / "seeds" / "agents"


def _upsert_agent_from_yaml(db, agent_id: str):
    """Re-read the agent YAML into the agents table. Returns the row."""
    from app import models
    from seeds.load_missions import load_agent_file

    path = BASE / f"{agent_id}.yaml"
    if not path.is_file():
        raise SystemExit(f"No agent YAML found: {path}")
    agent = db.merge(models.Agent(**load_agent_file(path)))
    db.commit()
    return agent


def register_agent_version(agent_id: str, *, activate: bool = True,
                           prompt_text: str | None = None,
                           register_prompt: bool = False) -> int:
    from app.services import agent_version_service

    if register_prompt:
        from scripts.register_prompts import main as register_prompts_main

        try:
            rc = register_prompts_main(["--prompt", agent_id])
        except SystemExit as exc:
            rc = int(exc.code or 1)
        if rc != 0:
            print(f"WARNING: prompt registration failed (exit {rc}); "
                  "the version is still created from the YAML text.")
    init_db()
    db = SessionLocal()
    try:
        _upsert_agent_from_yaml(db, agent_id)
        version, created = agent_version_service.create_version(
            db, agent_id, source="cli", activate=activate,
            prompt_text=prompt_text, record_mlflow_version=register_prompt,
        )
        mlflow_note = (
            f", MLflow prompt v{version.prompt_registry_version}"
            if version.prompt_registry_version else ""
        )
        print(
            f"Agent {agent_id}: version v{version.version} "
            f"({'created' if created else 'already active, no change'})"
            f" — model {version.model}, temp {version.temperature}, "
            f"{len(version.allowed_tools)} allowed tools, "
            f"manifest {version.manifest_sha256[:12]}"
            f"{', ACTIVE' if version.is_active else ', inactive'}"
            f"{mlflow_note}"
        )
        logger.info("Agent %s: version v%d (%s)", agent_id, version.version,
                    "created" if created else "unchanged")
        return 0
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register an agent version (Option C).")
    parser.add_argument("--agent-id", default=None, help="agent id to version")
    parser.add_argument("--all", action="store_true", help="version every agent YAML")
    parser.add_argument("--no-activate", action="store_true",
                        help="create the version but do not activate it")
    parser.add_argument("--prompt-text-file", default=None,
                        help="use this file's text as the prompt")
    parser.add_argument("--register-prompt", action="store_true",
                        help="also register the prompt in MLflow first")
    args = parser.parse_args(argv)

    if bool(args.agent_id) == bool(args.all):
        parser.error("give exactly one of --agent-id or --all")
    if args.all and args.prompt_text_file:
        parser.error("--prompt-text-file cannot be used with --all "
                     "(one text cannot fit every agent)")
    prompt_text = None
    if args.prompt_text_file:
        prompt_text = Path(args.prompt_text_file).read_text(encoding="utf-8")

    agent_ids = (
        sorted(p.stem for p in BASE.glob("*.yaml"))
        if args.all else [args.agent_id]
    )
    for agent_id in agent_ids:
        register_agent_version(
            agent_id, activate=not args.no_activate,
            prompt_text=prompt_text, register_prompt=args.register_prompt,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
