"""Phase 0 retrieval eval — run the plain-vector baseline through the harness.

Fake-only: deterministic BoW embeddings + in-memory fake Qdrant, so it
runs with no LiteLLM proxy, no Qdrant server and no live embedding
model. Prints the baseline numbers table; with --tracking-uri it also
logs them to MLflow (a dead server only warns, never fails the run).

Usage:
    python -m scripts.eval_retrieval [--tracking-uri URI] [--output FILE]
    make eval-retrieval-phase0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.retrieval_baseline import build_baseline_retriever  # noqa: E402
from runtime.retrieval_harness import (  # noqa: E402
    format_results_table,
    load_golden_set,
    run_retrieval_eval,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 retrieval baseline eval.")
    parser.add_argument(
        "--tracking-uri", default=None,
        help="MLflow tracking URI (default: skip MLflow, print results only).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write the full results JSON to this file.",
    )
    parser.add_argument(
        "--golden-set", default=None,
        help="Path to the golden set JSON (default: scripts/golden_set_fault_analysis.json).",
    )
    args = parser.parse_args()

    golden = load_golden_set(args.golden_set)
    retriever = build_baseline_retriever()
    result = run_retrieval_eval(
        retriever,
        golden["cases"],
        retriever_name="plain_vector_baseline",
        golden_set_version=golden.get("version", "unknown"),
        tracking_uri=args.tracking_uri,
        params={"embedding": "deterministic-bow-over-seeded-doc-vocab (fake)"},
    )
    print(format_results_table(result))
    print("\nNOTE: these are FAKE-SETUP numbers (deterministic BoW stand-in "
          "vectors over the 7 seeded docs), not live measurements. Live "
          "numbers come from the user's machine with the real embedding "
          "model + seeded Qdrant.")
    if result.get("mlflow_run_id"):
        print(f"MLflow run: {result['mlflow_run_id']}")
    elif args.tracking_uri:
        print("MLflow logging failed or was skipped; results above are unaffected.")
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Full results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
