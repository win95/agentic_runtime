"""Tests for the fine-tuning scaffold — offline, no torch, no GPU, no server.

Covers: dataset schemas (accept good / reject bad), synthetic
generators (counts + valid schema), YAML configs (parse + required
keys), lazy imports of the training scripts (importable without torch),
collect_from_traces conversion, and tool-use metric computation.
"""

import importlib
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from training.common import synthetic
from training.common.collect_from_traces import _demo_records, collect
from training.common.schemas import (
    EmbeddingExample,
    SFTExample,
    ToolUseExample,
    validate_jsonl,
)
from training.llm.eval_tool_use import _mock_predict, evaluate

N = synthetic.N_PER_TRACK


def test_embedding_schema_accepts_good_rejects_bad():
    good = {
        "id": "emb-001",
        "query": "lỗi cửa O21?",
        "positive": "tài liệu cửa",
        "negatives": ["tài liệu khác"],
        "source": "synthetic",
        "split": "train",
    }
    assert EmbeddingExample.model_validate(good).query.startswith("lỗi cửa")
    with pytest.raises(ValidationError):
        EmbeddingExample.model_validate({"query": "", "positive": "x"})
    with pytest.raises(ValidationError):
        EmbeddingExample.model_validate({"query": "q", "positive": "p", "split": "nope"})


def test_sft_schema_requires_messages():
    with pytest.raises(ValidationError):
        SFTExample.model_validate({"mission_id": "fault_analysis", "messages": []})
    ok = SFTExample.model_validate(
        {
            "mission_id": "fault_analysis",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
    )
    assert len(ok.messages) == 2


def test_tool_use_schema_allows_null_expected_tool():
    ok = ToolUseExample.model_validate(
        {
            "mission_id": "fault_analysis",
            "context": "đặt lịch họp",
            "available_tools": ["search_fault_history"],
            "expected_tool": None,
            "expected_args": {},
        }
    )
    assert ok.expected_tool is None


def test_synthetic_counts_and_validity():
    emb = synthetic.generate_embedding()
    sft = synthetic.generate_sft()
    tool = synthetic.generate_tool_use()
    assert len(emb) == len(sft) == len(tool) == N
    assert all(isinstance(e, EmbeddingExample) for e in emb)
    assert all(isinstance(e, SFTExample) for e in sft)
    assert all(isinstance(e, ToolUseExample) for e in tool)
    assert all(e.source == "synthetic" for e in emb + sft + tool)
    # Traps: the last 3 tool-use cases must expect NO tool call.
    assert all(e.expected_tool is None for e in tool[-3:])
    assert all(e.expected_tool is not None for e in tool[:-3])
    # Splits cover train/val/test.
    assert {e.split for e in emb} == {"train", "val", "test"}


def test_validate_jsonl_round_trip(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"id": "emb-001", "query": "q", "positive": "p", "negatives": []}\n'
        '{"query": "", "positive": "p"}\n'
        "not json\n",
        encoding="utf-8",
    )
    valid, errors = validate_jsonl(path, EmbeddingExample)
    assert len(valid) == 1
    assert len(errors) == 2
    assert errors[0]["line"] == 2
    assert errors[1]["line"] == 3


def test_configs_load_and_have_required_keys():
    import yaml

    emb = yaml.safe_load(open("training/configs/embedding.yaml", encoding="utf-8"))
    llm = yaml.safe_load(open("training/configs/llm_sft.yaml", encoding="utf-8"))
    for cfg in (emb, llm):
        assert cfg["base_model"]
        assert cfg["data"]
        assert cfg["output_dir"]
        assert cfg["mlflow_experiment"] == "agentic-runtime-finetune"
        assert int(cfg["epochs"]) >= 1
    assert "lora" in llm and int(llm["lora"]["r"]) >= 1


def test_training_scripts_import_without_torch():
    """Train scripts must import with no torch installed (lazy imports)."""
    for mod in (
        "training.embedding.train",
        "training.embedding.eval_retrieval",
        "training.llm.train_sft",
        "training.llm.eval_tool_use",
    ):
        importlib.import_module(mod)


def test_training_scripts_help_without_torch():
    """--help must print usage (argparse runs before any torch import)."""
    for script in (
        "training.embedding.train",
        "training.embedding.eval_retrieval",
        "training.llm.train_sft",
    ):
        proc = subprocess.run(
            [sys.executable, "-m", script, "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert "usage:" in proc.stdout.lower()


def test_train_scripts_fail_cleanly_without_torch(tmp_path):
    """With no torch, train scripts exit 2 with a clear message (no traceback)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.embedding.train",
            "--model",
            "some-model",
            "--data",
            "x.jsonl",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "sentence-transformers" in proc.stdout or "training" in proc.stdout.lower()
    assert "Traceback" not in proc.stdout
    assert "Traceback" not in proc.stderr

    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "base_model: some-model\ndata: x.jsonl\noutput_dir: out\n"
        "epochs: 1\nlora: {r: 8, alpha: 16, dropout: 0.0}\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "training.llm.train_sft", "--config", str(cfg)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stdout
    assert "Traceback" not in proc.stderr


def test_collect_from_traces_demo():
    sft, tool_use, stats = collect(_demo_records())
    assert stats == {"records": 3, "sft": 2, "tool_use": 1, "skipped": 1}
    assert len(sft) == 2
    assert all(isinstance(e, SFTExample) for e in sft)
    assert all(e.source == "trace" for e in sft)
    assert len(tool_use) == 1
    assert tool_use[0].expected_tool == "search_fault_history"
    assert tool_use[0].expected_args["asset_id"] == "O21"


def test_collect_skips_failed_runs():
    sft, tool_use, stats = collect(
        [{"name": "x", "success": 0, "validation_passed": 0, "report": "r"}]
    )
    assert stats["sft"] == 0 and stats["skipped"] == 1
    assert sft == [] and tool_use == []


def test_tool_use_metrics_with_mock_predictions():
    cases = synthetic.generate_tool_use()
    predictions = {
        case.id: {"predicted_tool": t, "predicted_args": a}
        for i, case in enumerate(cases)
        for t, a in [_mock_predict(case, i)]
    }
    metrics = evaluate(cases, predictions)
    # 20 cases. Wrong tool at (i+1)%5==0 -> i=4,9,14,19; only the first
    # three are tool cases (case 20 is a trap). 17 tool cases - 3 = 14.
    assert metrics["correct_tool_rate"] == 0.7
    # Wrong args at (i+1)%7==0 -> i=6,13 (both right tool). 14 - 2 = 12.
    assert metrics["correct_args_rate"] == 0.6
    assert metrics["out_of_scope_rate"] == 0.0
    assert metrics["n_traps"] == 3
    # Traps: i=17,18 pass (mock returns None); i=19 gets a wrong tool.
    assert metrics["trap_pass_rate"] == round(2 / 3, 4)


def test_tool_use_metrics_detect_out_of_scope():
    cases = synthetic.generate_tool_use()[:3]
    predictions = {
        cases[0].id: {"predicted_tool": "delete_database", "predicted_args": {}},
        cases[1].id: {
            "predicted_tool": cases[1].expected_tool,
            "predicted_args": dict(cases[1].expected_args),
        },
        cases[2].id: {"predicted_tool": None, "predicted_args": {}},
    }
    metrics = evaluate(cases, predictions)
    assert metrics["out_of_scope_rate"] == round(1 / 3, 4)
    assert metrics["correct_tool_rate"] == round(1 / 3, 4)


def test_eval_tool_use_cli_on_synthetic(tmp_path):
    """End-to-end CLI run on synthetic data with mock predictions."""
    data = tmp_path / "tool.jsonl"
    data.write_text(
        "\n".join(c.model_dump_json(ensure_ascii=False) for c in synthetic.generate_tool_use()),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.llm.eval_tool_use",
            "--data",
            str(data),
            "--mock-predictions",
            "--no-mlflow",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    metrics = json.loads(proc.stdout[proc.stdout.index("{"):])
    assert metrics["correct_tool_rate"] == 0.7
    assert metrics["mock"] is True
