"""Register system prompts in the MLflow Prompt Registry.

The registry holds a DOWNSTREAM COPY of the prompts for review and
rollback only — the runtime never reads it (docs/agent-versioning.md).
The prompts that actually run are the pinned text in an AgentVersion,
or, when no version is active, the hardcoded constants in
``runtime/deep_agent.py`` (kept identical to the YAML seed text).

Note (Option C): the prompt alone does not pin an agent's behaviour —
model, allowed tools and tool descriptions matter too. The full
versioned unit is an AgentVersion (see docs/agent-versioning.md);
``scripts/register_agent_version.py`` snapshots one (and can register
the prompt here first with --register-prompt).

Usage:
    python scripts/register_prompts.py [--tracking-uri URI] [--alias production]
    python scripts/register_prompts.py --prompt admin_bot

Requires a reachable MLflow 3.x server. When the server is unreachable the
script prints a clear one-line error and exits non-zero (no traceback).

To roll out a new prompt version: edit the prompt text in the agent YAML
(``seeds/agents/*.yaml``) and its mirror constant in
``runtime/deep_agent.py``, run this script again (it creates a new
immutable version), then move the alias:
    python scripts/register_prompts.py --alias production
Then re-seed (or run ``scripts/register_agent_version.py``) so the new
text becomes a new AgentVersion — editing the registry alone changes
nothing at runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPTS = {
    "fault_investigator": {"mission": "fault_analysis", "role": "system"},
    "admin_bot": {"mission": "system_admin", "role": "system"},
}
DEFAULT_ALIAS = "production"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register system prompts in MLflow (default: all prompts)."
    )
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or http://localhost:5000)")
    parser.add_argument("--alias", default=DEFAULT_ALIAS,
                        help="Alias to point at the new version (default: production)")
    parser.add_argument("--prompt", default=None, choices=sorted(PROMPTS),
                        help="Register only this prompt (default: all)")
    args = parser.parse_args(argv)

    from runtime.deep_agent import ADMIN_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT

    templates = {
        "fault_investigator": DEFAULT_SYSTEM_PROMPT,
        "admin_bot": ADMIN_SYSTEM_PROMPT,
    }
    names = [args.prompt] if args.prompt else sorted(PROMPTS)
    try:
        import mlflow

        mlflow.set_tracking_uri(args.tracking_uri)
        from mlflow.genai import register_prompt, set_prompt_alias

        for name in names:
            version = register_prompt(
                name=name,
                template=templates[name],
                commit_message=f"{name} system prompt (from runtime/deep_agent.py)",
                tags=PROMPTS[name],
            )
            set_prompt_alias(name, args.alias, version.version)
            print(f"Registered prompt '{name}' version {version.version} (alias '{args.alias}').")
            print(f"View it at {args.tracking_uri.rstrip('/')}/#/prompts/{name}")
    except Exception as exc:  # noqa: BLE001 — server down, bad URI, etc.
        print(f"ERROR: could not register prompt in MLflow at {args.tracking_uri}: {exc}")
        print("Start the MLflow 3.x server first "
              "(docker compose --profile observability up -d mlflow),")
        print("or pass --tracking-uri sqlite:///tmp/mlflow-test/mlflow.db for a local backend.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
