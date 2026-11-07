"""Settings tests — defaults and env loading (fresh Settings instances)."""

from app.config import Settings


def test_settings_defaults():
    settings = Settings()
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.langsmith_tracing is False
    assert settings.checkpoint_enabled is False
    assert settings.execution_timeout_seconds == 3600
    # Secrets default to empty — never hardcoded.
    assert settings.openai_api_key == ""
    assert settings.api_key == ""


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-model-123")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("QDRANT_COLLECTION", "test-docs")
    settings = Settings()
    assert settings.llm_model == "test-model-123"
    assert settings.llm_provider == "anthropic"
    assert settings.qdrant_collection == "test-docs"
