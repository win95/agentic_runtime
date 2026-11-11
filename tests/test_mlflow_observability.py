"""MLflow 3.x observability tests — tracing setup, prompt registry write.

Offline only: no MLflow server, no LiteLLM proxy, no real key. The
register round-trip test uses a throwaway local SQLite backend (real
MLflow code, no network).

Note: the runtime never READS prompts from the registry
(docs/agent-versioning.md) — the registry only holds a downstream copy
for review/rollback, so there is no read-path test here.
"""

import importlib.util
from pathlib import Path

import mlflow
import pytest

import infra.mlflow_setup as mlflow_setup
from app.config import settings
from runtime.deep_agent import ADMIN_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_mlflow_setup():
    mlflow_setup.reset_mlflow_setup_for_tests()
    yield
    mlflow_setup.reset_mlflow_setup_for_tests()


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_returns_false_without_server(monkeypatch):
    monkeypatch.setattr(settings, "mlflow_tracking_uri", "http://127.0.0.1:1")
    monkeypatch.setattr(settings, "mlflow_tracing_enabled", True)
    assert mlflow_setup.setup_mlflow_tracing() is False
    assert mlflow_setup.is_mlflow_tracing_enabled() is False


def test_autolog_called_once(monkeypatch):
    import mlflow.langchain

    calls = {"n": 0}

    def _fake_autolog(*args, **kwargs):
        calls["n"] += 1

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(mlflow.langchain, "autolog", _fake_autolog)
    monkeypatch.setattr(settings, "mlflow_tracing_enabled", True)
    assert mlflow_setup.setup_mlflow_tracing() is True
    assert mlflow_setup.setup_mlflow_tracing() is True
    assert calls["n"] == 1


def test_setup_disabled_by_env(monkeypatch):
    monkeypatch.setattr(settings, "mlflow_tracing_enabled", False)
    assert mlflow_setup.setup_mlflow_tracing() is False


def test_register_prompts_sqlite_roundtrip(tmp_path):
    """Write path still works: register_prompts.py stores the code
    constants in the registry (the runtime never reads them back)."""
    import mlflow.genai as genai

    module = _load_script("register_prompts")
    db_path = tmp_path / "mlflow.db"
    try:
        assert module.main(["--tracking-uri", f"sqlite:///{db_path}"]) == 0
        fault = genai.load_prompt("prompts:/fault_investigator@production")
        assert fault.template == DEFAULT_SYSTEM_PROMPT
        admin = genai.load_prompt("prompts:/admin_bot@production")
        assert admin.template == ADMIN_SYSTEM_PROMPT
    finally:
        mlflow.set_tracking_uri("http://localhost:5000")


def test_register_prompts_fails_cleanly_without_server(capsys):
    module = _load_script("register_prompts")
    code = module.main(["--tracking-uri", "http://127.0.0.1:1"])
    out = capsys.readouterr().out
    assert code == 3
    assert "ERROR" in out
    assert "Traceback" not in out


def test_genai_skips_without_judge_routing(monkeypatch, capsys):
    module = _load_script("eval_mission")
    monkeypatch.setattr(settings, "llm_api_key", "")
    result = module.run_genai_eval([], mock_llm=True, tracking_uri="http://127.0.0.1:1")
    out = capsys.readouterr().out
    assert result is None
    assert "SKIPPED" in out
