"""SFT of the main LLM with LoRA (track B).

Method: TRL SFTTrainer + PEFT LoRA, configured by a YAML file
(training/configs/llm_sft.yaml). Trains on SFTExample JSONL (chat
messages with optional tool calls). Every run logs params/metrics to
the MLflow experiment from the config.

All training libraries are imported LAZILY inside main() — importing
this module (or running --help) works without torch installed. With no
torch/TRL/PEFT, or with no GPU, the script exits with a clear message
instead of a traceback. Pass --allow-cpu to run a tiny smoke run on
CPU (slow, for pipeline checks only).

Usage:
    python -m training.llm.train_sft --config training/configs/llm_sft.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_CONFIG = "training/configs/llm_sft.yaml"


def _load_config(path: str) -> dict:
    import yaml

    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _format_messages(messages: list[dict]) -> str:
    """Render chat messages as plain text (no tokenizer chat template needed).

    Tool calls are rendered as ``<tool_call name args>`` lines so the
    model learns the call format even before a tokenizer is involved.
    """
    import json

    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "tool":
            lines.append(f"[tool result] {content}")
            continue
        lines.append(f"[{role}] {content}")
        for call in msg.get("tool_calls") or []:
            args_json = json.dumps(call.get("arguments", {}), ensure_ascii=False)
            lines.append(f"[tool_call] {call.get('name')} {args_json}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SFT of the main LLM with LoRA.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file.")
    parser.add_argument("--data", default=None, help="SFTExample JSONL (overrides config).")
    parser.add_argument("--output", default=None, help="Output dir (overrides config).")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow a CPU smoke run (slow).")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    model_name = config.get("base_model", "")
    data_path = args.data or config.get("data", "training/data/synthetic_sft.jsonl")
    output_dir = args.output or config.get("output_dir", "training/output/llm-sft-v1")

    if not model_name or "PLACEHOLDER" in model_name.upper():
        print("ERROR: no base model configured.")
        print("Set base_model in training/configs/llm_sft.yaml to the real")
        print("main LLM (see docs/fine-tuning-lifecycle.md, open decision 2).")
        return 2

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ModuleNotFoundError as exc:
        print(f"ERROR: missing training dependency: {exc.name}")
        print("Install the training deps first (needs a machine with GPU for real runs):")
        print("  pip install -r training/requirements-training.txt")
        print("This environment has no torch/GPU — training cannot run here.")
        return 2

    if not torch.cuda.is_available() and not args.allow_cpu:
        print("ERROR: no GPU detected (torch.cuda.is_available() is False).")
        print("LLM SFT needs a GPU. Run on the training machine, or pass")
        print("--allow-cpu for a tiny CPU smoke run (slow, pipeline check only).")
        return 2

    from training.common.schemas import SFTExample, validate_jsonl

    examples, errors = validate_jsonl(data_path, SFTExample)
    if errors:
        print(f"ERROR: {len(errors)} invalid line(s) in {data_path}; fix the dataset first:")
        for err in errors[:5]:
            print(f"  line {err['line']}: {err['error']}")
        return 2
    train_examples = [ex for ex in examples if ex.split == "train"]
    if not train_examples:
        print("ERROR: no train-split examples in the dataset.")
        return 2
    print(f"Loaded {len(train_examples)} train SFT examples from {data_path}")

    lora = config.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        task_type="CAUSAL_LM",
    )
    dataset = Dataset.from_list(
        [
            {"text": _format_messages([m.model_dump() for m in ex.messages])}
            for ex in train_examples
        ]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=float(config.get("epochs", 3)),
        per_device_train_batch_size=int(config.get("batch_size", 4)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        logging_steps=int(config.get("logging_steps", 10)),
        save_strategy="epoch",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved fine-tuned LLM (LoRA adapter) to {output_dir}")

    try:
        import mlflow

        mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", "http://localhost:5000"))
        mlflow.set_experiment(config.get("mlflow_experiment", "agentic-runtime-finetune"))
        with mlflow.start_run(run_name=f"llm-sft-{Path(output_dir).name}"):
            mlflow.log_params(
                {
                    "track": "llm-sft",
                    "base_model": model_name,
                    "lora_r": peft_config.r,
                    "lora_alpha": peft_config.lora_alpha,
                    "epochs": config.get("epochs", 3),
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
