"""MVP tests — LiteLLM factory, Qdrant helpers, report, langgraph.json
and the MLflow eval harness (mock mode). No infra, no real key, no
network calls anywhere in this file."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def _dummy_settings(**overrides):
    values = {
        "llm_api_base": "http://localhost:4000",
        "llm_api_key": "dummy-key",
        "llm_model": "my-model",
        "llm_temperature": 0.2,
        "embedding_api_base": "",
        "embedding_api_key": "",
        "embedding_model": "my-embed",
    }
    values.update(overrides)
    return Settings(**values)


def test_factory_builds_chat_model_from_env():
    from runtime.llm_factory import build_chat_model

    model = build_chat_model(_dummy_settings())
    assert model.model_name == "my-model"
    assert model.openai_api_base == "http://localhost:4000"
    assert model.temperature == 0.2


def test_factory_builds_embeddings_from_env():
    from runtime.llm_factory import build_embeddings

    embeddings = build_embeddings(_dummy_settings())
    assert embeddings.model == "my-embed"
    assert embeddings.openai_api_base == "http://localhost:4000"


def test_factory_fails_fast_without_key():
    from runtime.llm_factory import build_chat_model, build_embeddings

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        build_chat_model(_dummy_settings(llm_api_key=""))
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        build_embeddings(_dummy_settings(embedding_model=""))


def test_qdrant_helpers_with_mocked_client(monkeypatch):
    import infra.qdrant_client as qc

    client = MagicMock()
    # Collection missing, then present with dim 3.
    client.get_collection.side_effect = [
        Exception("Not found"),
        SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=3)))),
        SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=3)))),
    ]
    client.retrieve.return_value = []
    client.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(id="p1", score=0.9, payload={"doc_id": "d1", "title": "T"})]
    )
    monkeypatch.setattr(qc, "get_client", lambda: client)

    class FakeEmbeddings:
        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    embeddings = FakeEmbeddings()
    assert qc.probe_dimension(embeddings=embeddings) == 3
    assert qc.ensure_collection(3) == "created"
    result = qc.upsert_documents(
        [{"id": "d1", "title": "T", "content": "text"}], embeddings=embeddings
    )
    assert result == {"upserted": 1, "skipped": 0}
    docs = qc.search_documents("query", embeddings=embeddings)
    assert docs[0]["doc_id"] == "d1"
    assert docs[0]["score"] == 0.9


def test_qdrant_dimension_mismatch_raises(monkeypatch):
    import infra.qdrant_client as qc

    client = MagicMock()
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=5)))
    )
    monkeypatch.setattr(qc, "get_client", lambda: client)
    with pytest.raises(RuntimeError, match="vector size 5"):
        qc.ensure_collection(3)


def test_report_builder_contains_sections():
    from runtime.report import build_markdown_report

    report = build_markdown_report(
        "exec-1",
        {"mission_id": "fault_analysis", "version": "1.0.0"},
        {"asset_id": "O21"},
        {
            "status": "SUCCEEDED",
            "result": {"root_causes": ["Door sensor wear"], "recommendations": ["Replace sensor"]},
            "evidence": [
                {"source": "fault_history", "kind": "fault", "summary": "F-101", "stub": False}
            ],
        },
    )
    assert "# Execution report — exec-1" in report
    assert "Door sensor wear" in report
    assert "F-101" in report
    assert "Replace sensor" in report


def test_langgraph_json_exposes_compiled_graph():
    import runtime.mission_graph as mg

    assert mg.mission_graph is not None
    assert hasattr(mg.mission_graph, "invoke")
    spec = json.loads((ROOT / "langgraph.json").read_text())
    assert spec["graphs"]["mission_graph"] == "./runtime/mission_graph.py:mission_graph"
    assert spec["env"] == ".env"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_mission", ROOT / "scripts" / "eval_mission.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_mock_mode_logs_to_mlflow(monkeypatch, tmp_path):
    import infra.repositories as repos
    import runtime.deep_agent as deep_agent_module

    monkeypatch.setattr(
        repos, "search_fault_history",
        lambda *a, **k: ([{"id": "fh-1", "code": "F-101", "title": "Door sensor fault"}], False),
    )
    monkeypatch.setattr(
        repos, "get_maintenance_records",
        lambda *a, **k: ([{"id": "mr-1", "action": "Sensor cleaned"}], False),
    )
    monkeypatch.setattr(
        repos, "search_documents",
        lambda *a, **k: ([{"doc_id": "d1", "title": "Door sensor doc"}], False),
    )
    monkeypatch.setattr(
        repos, "query_asset_relationships",
        lambda *a, **k: ([{"from": "O21", "type": "HAS_COMPONENT", "to": "motor-M2"}], False),
    )

    eval_module = _load_eval_module()
    eval_module.install_mock_llm()
    assert deep_agent_module.build_deep_agent() is not None

    case = eval_module.TEST_CASES[0]
    result = eval_module.run_case(case, mock_llm=True, model_name="mock")
    assert result["success"] == 1
    assert result["validation_passed"] == 1
    assert result["evidence_count"] >= 3
    assert result["llm"] == "mock"

    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    run_id = eval_module.log_to_mlflow(result, tracking_uri)
    assert run_id

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    assert run.data.params["llm"] == "mock"
    assert run.data.metrics["success"] == 1.0
    assert run.data.metrics["validation_passed"] == 1.0
