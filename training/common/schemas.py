"""Dataset contracts for the three fine-tuning tracks.

These Pydantic schemas are the law: every dataset (synthetic or real)
must validate against them before training or evaluation. A record that
does not fit its schema is dropped and logged, never silently fixed.

Tracks:
- EmbeddingExample: contrastive retrieval (query / positive / negatives).
- SFTExample: supervised fine-tuning of the main LLM (chat messages with
  optional tool calls and tool results).
- ToolUseExample: the tool-use slice (context / available tools /
  expected tool call).

All examples carry ``source`` ("synthetic" for generated data, "trace"
or similar for real operational data) and ``split`` (train/val/test).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

Split = Literal["train", "val", "test"]


class EmbeddingExample(BaseModel):
    """One contrastive retrieval example (track A)."""

    id: str = Field(default="", description="Stable id, e.g. 'emb-001'.")
    query: str = Field(min_length=1, description="User query (Vietnamese).")
    positive: str = Field(min_length=1, description="The correct document.")
    negatives: list[str] = Field(
        default_factory=list,
        description="Near-miss documents that are NOT the answer.",
    )
    source: str = Field(default="synthetic", description="synthetic | trace | manual.")
    split: Split = "train"


class ChatMessage(BaseModel):
    """One chat turn in an SFT example."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)
    tool_calls: list["ToolCall"] = Field(default_factory=list)
    tool_call_id: str = ""


class ToolCall(BaseModel):
    """One tool call made by the assistant."""

    name: str = Field(min_length=1)
    arguments: dict = Field(default_factory=dict)


class SFTExample(BaseModel):
    """One supervised fine-tuning example (track B)."""

    id: str = Field(default="", description="Stable id, e.g. 'sft-001'.")
    mission_id: str = Field(min_length=1, description="E.g. 'fault_analysis'.")
    messages: list[ChatMessage] = Field(
        min_length=2, description="At least a user turn and an assistant turn."
    )
    source: str = Field(default="synthetic")
    split: Split = "train"


class ToolUseExample(BaseModel):
    """One tool-use evaluation/training example (track C)."""

    id: str = Field(default="", description="Stable id, e.g. 'tool-001'.")
    mission_id: str = Field(min_length=1, description="E.g. 'fault_analysis'.")
    context: str = Field(min_length=1, description="Request / state summary.")
    available_tools: list[str] = Field(
        min_length=1, description="Tools the agent is allowed to call."
    )
    expected_tool: str | None = Field(
        default=None,
        description="Tool that should be called; null means call NO tool.",
    )
    expected_args: dict = Field(
        default_factory=dict, description="Arguments of the expected call."
    )
    source: str = Field(default="synthetic")
    split: Split = "train"


SCHEMA_BY_TRACK = {
    "embedding": EmbeddingExample,
    "sft": SFTExample,
    "tool_use": ToolUseExample,
}


def validate_jsonl(path: str | Path, model: type[BaseModel]) -> tuple[list[BaseModel], list[dict]]:
    """Validate a JSONL file against a schema.

    Returns ``(valid_examples, errors)`` where each error is
    ``{"line": n, "raw": <parsed json or text>, "error": <message>}``.
    Invalid lines are reported, never fixed up silently.
    """
    path = Path(path)
    valid: list[BaseModel] = []
    errors: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": lineno, "raw": line, "error": f"bad JSON: {exc}"})
                continue
            try:
                valid.append(model.model_validate(raw))
            except ValidationError as exc:
                errors.append({"line": lineno, "raw": raw, "error": str(exc)})
    return valid, errors


def write_jsonl(path: str | Path, examples: list[BaseModel]) -> int:
    """Write validated examples as JSONL. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json(ensure_ascii=False) + "\n")
    return len(examples)
