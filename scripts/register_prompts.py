"""Register the fault_investigator system prompt in the MLflow Prompt Registry.

The registered prompt becomes the Deep Agent's system prompt (loaded by
``runtime/prompts.py`` via the ``production`` alias, falling back to
``@latest`` and then to the hardcoded default in ``runtime/deep_agent.py``).

Usage:
    python scripts/register_prompts.py [--tracking-uri URI] [--alias production]

Requires a reachable MLflow 3.x server. When the server is unreachable the
script prints a clear one-line error and exits non-zero (no traceback).

To roll out a new prompt version: edit DEFAULT_SYSTEM_PROMPT in
``runtime/deep_agent.py``, run this script again (it creates a new immutable
version), then move the alias:
    python scripts/register_prompts.py --alias production
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPT_NAME = "fault_investigator"
DEFAULT_ALIAS = "production"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register the fault_investigator system prompt in MLflow."
    )
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or http://localhost:5000)")
    parser.add_argument("--alias", default=DEFAULT_ALIAS,
                        help="Alias to point at the new version (default: production)")
    args = parser.parse_args(argv)

    from runtime.deep_agent import DEFAULT_SYSTEM_PROMPT

    try:
        import mlflow

        mlflow.set_tracking_uri(args.tracking_uri)
        from mlflow.genai import register_prompt, set_prompt_alias

        version = register_prompt(
            name=PROMPT_NAME,
            template=DEFAULT_SYSTEM_PROMPT,
            commit_message="fault_investigator system prompt (from runtime/deep_agent.py)",
            tags={"mission": "fault_analysis", "role": "system"},
        )
        set_prompt_alias(PROMPT_NAME, args.alias, version.version)
    except Exception as exc:  # noqa: BLE001 — server down, bad URI, etc.
        print(f"ERROR: could not register prompt in MLflow at {args.tracking_uri}: {exc}")
        print("Start the MLflow 3.x server first "
              "(docker compose --profile observability up -d mlflow),")
        print("or pass --tracking-uri sqlite:///tmp/mlflow-test/mlflow.db for a local backend.")
        return 3
    print(f"Registered prompt '{PROMPT_NAME}' version {version.version} (alias '{args.alias}').")
    print(f"View it at {args.tracking_uri.rstrip('/')}/#/prompts/{PROMPT_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
