"""Contrastive fine-tune of the embedding model (track A).

Method: sentence-transformers + MultipleNegativesRankingLoss on
EmbeddingExample JSONL (query / positive / negatives). Every run logs
params/metrics to the MLflow experiment from the config.

Training libraries are imported LAZILY inside main() — importing this
module (or running --help) works without torch installed. With no
torch/sentence-transformers, the script exits with a clear message
instead of a traceback.

Usage:
    python -m training.embedding.train --config training/configs/embedding.yaml
    python -m training.embedding.train --model <name> --data <file> --output <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_CONFIG = "training/configs/embedding.yaml"


def _load_config(path: str) -> dict:
    import yaml

    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contrastive fine-tune of the embedding model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file.")
    parser.add_argument("--model", default=None, help="Base model (overrides config).")
    parser.add_argument("--data", default=None, help="EmbeddingExample JSONL (overrides config).")
    parser.add_argument("--output", default=None, help="Output dir (overrides config).")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    model_name = args.model or config.get("base_model", "")
    data_path = args.data or config.get("data", "training/data/synthetic_embedding.jsonl")
    output_dir = args.output or config.get("output_dir", "training/output/embedding-v1")

    if not model_name or "PLACEHOLDER" in model_name.upper():
        print("ERROR: no base model configured.")
        print("Set base_model in training/configs/embedding.yaml to the real")
        print("embedding model (see docs/fine-tuning-lifecycle.md, open decision 1).")
        return 2

    try:
        import torch  # noqa: F401
        from sentence_transformers import InputExample, SentenceTransformer, losses

        try:
            from sentence_transformers import (
                SentenceTransformerTrainer,
                SentenceTransformerTrainingArguments,
            )
        except ImportError:  # older 3.x layout
            from sentence_transformers.trainer import SentenceTransformerTrainer
            from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    except ModuleNotFoundError as exc:
        print(f"ERROR: missing training dependency: {exc.name}")
        print("Install the training deps first (needs a machine with GPU for real runs):")
        print("  pip install -r training/requirements-training.txt")
        print("This environment has no torch/GPU — training cannot run here.")
        return 2

    from training.common.schemas import EmbeddingExample, validate_jsonl

    examples, errors = validate_jsonl(data_path, EmbeddingExample)
    if errors:
        print(f"ERROR: {len(errors)} invalid line(s) in {data_path}; fix the dataset first:")
        for err in errors[:5]:
            print(f"  line {err['line']}: {err['error']}")
        return 2
    if not examples:
        print(f"ERROR: no valid examples in {data_path}.")
        return 2
    print(f"Loaded {len(examples)} embedding examples from {data_path}")

    train_examples = [
        InputExample(texts=[ex.query, ex.positive] + list(ex.negatives))
        for ex in examples
        if ex.split == "train"
    ]
    if not train_examples:
        print("ERROR: no train-split examples in the dataset.")
        return 2

    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 16))
    lr = float(config.get("learning_rate", 2e-5))

    model = SentenceTransformer(model_name)
    loss = losses.MultipleNegativesRankingLoss(model)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        save_strategy="epoch",
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=train_examples, loss=loss
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved fine-tuned embedding model to {output_dir}")

    try:
        import mlflow

        mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", "http://localhost:5000"))
        mlflow.set_experiment(config.get("mlflow_experiment", "agentic-runtime-finetune"))
        with mlflow.start_run(run_name=f"embedding-{Path(output_dir).name}"):
            mlflow.log_params(
                {
                    "track": "embedding",
                    "base_model": model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "n_train": len(train_examples),
                }
            )
            mlflow.log_artifact(output_dir)
        print("Logged training run to MLflow.")
    except Exception as exc:  # noqa: BLE001 — MLflow is observability, never fatal
        print(f"WARNING: could not log to MLflow ({exc}); model is still saved locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
