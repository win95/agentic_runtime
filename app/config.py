"""Application settings — everything comes from the environment.

No secret is ever hardcoded. Copy `.env.example` to `.env` for local dev.

LLM access goes through the user's LiteLLM proxy, which speaks the
OpenAI-compatible protocol. The four values the user must supply are
LLM_API_BASE, LLM_API_KEY, LLM_MODEL and EMBEDDING_MODEL (see
`.env.example` and README). Placeholders below are only examples of the
*shape* of a value — they are not real endpoints or model names.
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

    # --- LLM via LiteLLM (OpenAI-compatible) ---
    # Base URL of the user's LiteLLM proxy, e.g. http://localhost:4000.
    # Placeholder default only — the user MUST set the real URL.
    llm_api_base: str = "http://localhost:4000"
    # API key for the LiteLLM proxy. Empty by default; required at runtime.
    llm_api_key: str = ""
    # Model name AS REGISTERED ON THE USER'S LiteLLM PROXY
    # (e.g. gpt-4o-mini). The proxy routes by this name.
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2

    # --- Embeddings via LiteLLM (OpenAI-compatible) ---
    # Empty base/key => reuse the LLM proxy values above.
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    # Embedding model name AS REGISTERED ON THE USER'S LiteLLM PROXY
    # (e.g. text-embedding-3-small). The vector dimension is detected
    # from a probe embedding at collection-creation time, never hardcoded.
    embedding_model: str = "text-embedding-3-small"

    # LangSmith — observability only, never the source of execution state.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "agentic-runtime"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

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
    minio_reports_bucket: str = "reports"
    minio_secure: bool = False

    # MLflow (primary observability: tracing, eval, prompt registry)
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "agentic-runtime"
    mlflow_tracing_enabled: bool = True

    # MCP — generic server URL, configured by env only. Empty = no MCP tools.
    mcp_server_url: str = ""
    mcp_transport: str = "streamable_http"
    # Layer 4: which capability each MCP tool belongs to, as a JSON
    # object {"tool_name": "capability"}. An MCP tool with no mapping
    # here is DENIED (never handed to an agent) — MCP tool descriptions
    # are untrusted prompt content, so unknown tools are not trusted.
    # MCP tools are treated as writes (read_only=False) because their
    # side effects are unknown to this codebase.
    mcp_tool_capabilities: str = ""

    # --- Slack notifications (Incoming Webhook, #agent-activity) ---
    # Webhook URL for the workspace's Incoming Webhook app. Empty =
    # notifications disabled (the notifier is a silent no-op); the URL
    # is a secret, it lives only in .env, never in the repo.
    slack_webhook_url: str = ""
    # Documentary only for now: an Incoming Webhook posts to the channel
    # configured when the webhook was created, not to this value. Kept
    # so the intended target channel is explicit in config.
    slack_channel: str = "#agent-activity"
    # Master switch: False disables all Slack notifications even when
    # a webhook URL is set.
    slack_notifications_enabled: bool = True

    # Runtime
    checkpoint_enabled: bool = False
    execution_timeout_seconds: int = 3600

    # Batch ingest (Option B): files in a local inbox dir, or in the
    # MinIO ingest-inbox bucket when ingest_use_minio is true. Beat
    # runs the batch every ingest_interval_seconds (default 30 min).
    ingest_interval_seconds: int = 1800
    ingest_inbox_dir: str = "ingest/inbox"
    ingest_use_minio: bool = False
    ingest_minio_bucket: str = "ingest-inbox"

    # --- Resolved (effective) values ---
    @property
    def resolved_embedding_api_base(self) -> str:
        return self.embedding_api_base or self.llm_api_base

    @property
    def resolved_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key


settings = Settings()
