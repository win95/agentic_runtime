"""Settings tests — defaults and env loading (fresh Settings instances).

Item 10 (2026-11-11): Settings must be constructed with
_env_file=None and the env vars the assertions depend on cleared,
so these tests do not depend on the machine's .env file (or on
foreign env vars leaking in from the shell). A .env present in the
cwd must not change the outcome.
"""

from app.config import Settings

# Env vars whose defaults test_settings_defaults asserts.
_DEFAULTED_VARS = [
    "LLM_API_BASE",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_MODEL_INDEX",
    "EMBEDDING_MODEL_QUERY",
    "EMBEDDING_API_BASE",
    "EMBEDDING_API_KEY",
    "API_KEY",
    "MINIO_REPORTS_BUCKET",
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
    "CHECKPOINT_ENABLED",
    "EXECUTION_TIMEOUT_SECONDS",
    "QDRANT_COLLECTION",
]


def test_settings_defaults(monkeypatch):
    for var in _DEFAULTED_VARS:
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
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
    settings = Settings(_env_file=None)
    assert settings.llm_model == "test-model-123"
    assert settings.llm_api_base == "http://proxy.example:4000"
    assert settings.embedding_model == "test-embed-456"
    assert settings.qdrant_collection == "test-docs"
    # Embedding falls back to the LLM proxy.
    assert settings.resolved_embedding_api_base == "http://proxy.example:4000"
    assert settings.resolved_embedding_api_key == "dummy-key"


def test_embedding_model_per_part_resolution(monkeypatch):
    for var in ("EMBEDDING_MODEL", "EMBEDDING_MODEL_INDEX", "EMBEDDING_MODEL_QUERY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.embedding_model_index == ""
    assert settings.embedding_model_query == ""
    # No overrides: both parts share EMBEDDING_MODEL.
    assert settings.resolved_embedding_model_index == "text-embedding-3-small"
    assert settings.resolved_embedding_model_query == "text-embedding-3-small"

    # A custom shared model is the fallback for both parts.
    monkeypatch.setenv("EMBEDDING_MODEL", "shared-custom")
    settings = Settings(_env_file=None)
    assert settings.resolved_embedding_model_index == "shared-custom"
    assert settings.resolved_embedding_model_query == "shared-custom"

    # Each override applies to its own part only.
    monkeypatch.setenv("EMBEDDING_MODEL_INDEX", "index-model")
    settings = Settings(_env_file=None)
    assert settings.resolved_embedding_model_index == "index-model"
    assert settings.resolved_embedding_model_query == "shared-custom"
    monkeypatch.setenv("EMBEDDING_MODEL_QUERY", "query-model")
    settings = Settings(_env_file=None)
    assert settings.resolved_embedding_model_index == "index-model"
    assert settings.resolved_embedding_model_query == "query-model"
