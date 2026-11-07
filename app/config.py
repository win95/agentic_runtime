"""Application settings — everything comes from the environment.

No secret is ever hardcoded. Copy `.env.example` to `.env` for local dev.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Platform
    app_name: str = "agentic-runtime"
    environment: str = "dev"
    # Empty => API-key auth disabled (DEV ONLY). Production must set API_KEY.
    api_key: str = ""

    # PostgreSQL (missions, agents, tools, executions, results)
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/agentic_runtime"

    # Celery + Redis
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # LLM — primary provider is OpenAI (langchain-openai).
    # Alternative: LLM_PROVIDER=anthropic + `pip install -e ".[anthropic]"`.
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # LangSmith — observability only, never the source of execution state.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "agentic-runtime"

    # Qdrant (document embeddings / RAG)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "technical_documents"

    # FalkorDB (knowledge graph)
    falkor_host: str = "localhost"
    falkor_port: int = 6379

    # MinIO (files, artifacts, reports)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "agentic-runtime"
    minio_secure: bool = False

    # MLflow (experiments, optional)
    mlflow_tracking_uri: str = "http://localhost:5000"

    # MCP — generic server URL, configured by env only. Empty = no MCP tools.
    mcp_server_url: str = ""
    mcp_transport: str = "streamable_http"

    # Runtime
    checkpoint_enabled: bool = False
    execution_timeout_seconds: int = 3600


settings = Settings()
