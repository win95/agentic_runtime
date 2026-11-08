"""Settings tests — defaults and env loading (fresh Settings instances)."""

from app.config import Settings


def test_settings_defaults():
    settings = Settings()
    assert settings.llm_api_base == "http://localhost:4000"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_temperature == 0.2
    assert settings.embedding_model == "text-embedding-3-small"
    # Embedding base/key fall back to the LLM proxy values when empty.
    assert settings.embedding_api_base == ""
    assert settings.resolved_embedding_api_base == "http://localhost:4000"
    assert settings.resolved_embedding_api_key == ""
    assert settings.minio_reports_bucket == "reports"
    assert settings.langsmith_tracing is False
    assert settings.langsmith_endpoint == "https://api.smith.langchain.com"
    assert settings.checkpoint_enabled is False
    assert settings.execution_timeout_seconds == 3600
    # Secrets default to empty — never hardcoded.
    assert settings.llm_api_key == ""
    assert settings.embedding_api_key == ""
    assert settings.api_key == ""


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-model-123")
    monkeypatch.setenv("LLM_API_BASE", "http://proxy.example:4000")
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embed-456")
    monkeypatch.setenv("QDRANT_COLLECTION", "test-docs")
    settings = Settings()
    assert settings.llm_model == "test-model-123"
    assert settings.llm_api_base == "http://proxy.example:4000"
    assert settings.embedding_model == "test-embed-456"
    assert settings.qdrant_collection == "test-docs"
    # Embedding falls back to the LLM proxy.
    assert settings.resolved_embedding_api_base == "http://proxy.example:4000"
    assert settings.resolved_embedding_api_key == "dummy-key"
