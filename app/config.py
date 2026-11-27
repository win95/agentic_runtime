"""Application settings — everything comes from the environment.

No secret is ever hardcoded. Copy `.env.example` to `.env` for local dev.

LLM access goes through the user's LiteLLM proxy, which speaks the
OpenAI-compatible protocol. The four values the user must supply are
LLM_API_BASE, LLM_API_KEY, LLM_MODEL and EMBEDDING_MODEL (see
`.env.example` and README). Placeholders below are only examples of the
*shape* of a value — they are not real endpoints or model names.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Platform
    app_name: str = "agentic-runtime"
    environment: str = "dev"
    # Empty => API-key auth disabled (DEV ONLY). Production must set API_KEY.
    api_key: str = ""
    # JWT secret shared with the user service (services/user_service):
    # the service signs tokens with it, this API verifies them. Empty =
    # JWT auth is not configured (Bearer tokens are ignored). The two
    # values MUST match wherever both services run.
    jwt_secret: str = ""

    # User service (services/user_service) — for the Lark user sync
    # (POST /lark/users/sync, agent_lark_me D-LARK-11). The platform
    # calls the service's admin identity endpoints with a shared
    # service key (X-Service-Key), NOT a minted admin JWT: the user
    # service requires a JWT sub to be an existing User row, and the
    # platform cannot know an admin's id. The key must match
    # USER_SERVICE_API_KEY in the user service's env.
    user_service_url: str = "http://localhost:8001"
    user_service_api_key: str = ""

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
    # Per-part embedding model overrides (optional). Each part of the
    # retrieval pipeline may use its own embedding model later; when an
    # override is empty/unset, that part falls back to EMBEDDING_MODEL
    # above, so today everything still shares one model. The two models
    # MUST produce compatible vectors (same dimension / vector space):
    # changing the index model requires re-indexing Qdrant, and a query
    # model with a different dimension cannot search the collection.
    embedding_model_index: str = ""
    embedding_model_query: str = ""

    # LangSmith — observability only, never the source of execution state.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "agentic-runtime"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Qdrant (document embeddings / RAG)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "technical_documents"

    # Neo4j (ontology projection / knowledge graph)
    graph_database_backend: str = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "changeme"
    neo4j_database: str = "neo4j"
    graph_features_enabled: bool = True
    projection_drain_interval_seconds: int = 60
    projection_drain_batch_size: int = 32
    # Poison handling: a projection row that fails this many times is
    # marked status="dead" and never re-claimed (see docs/ontology.md).
    projection_max_attempts: int = 5

    # MinIO (files, artifacts, reports)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "agentic-runtime"
    minio_reports_bucket: str = "reports"
    minio_documents_bucket: str = "documents"
    minio_secure: bool = False
    minio_region: str = "us-east-1"
    # Presigned GET URLs for artifact objects (browser/host access). Empty
    # endpoint => reuse MINIO_ENDPOINT (e.g. localhost:9000 on the host).
    minio_presign_endpoint: str = ""
    minio_presign_secure: bool = False
    minio_artifacts_presign_ttl_seconds: int = 86400

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

    # --- OpenSandbox (per-session sandbox backend, docs/sandbox.md) ---
    # The Deep Agent's file tools + `execute` run inside one
    # OpenSandbox sandbox per session — but ONLY when sandbox_enabled
    # is true AND the mission opts in (execution.sandbox: true) AND
    # the execution has a session_id. Everything else keeps the
    # in-memory StateBackend. The server must be running first
    # (see docs/sandbox.md); the API key is a secret, it lives only
    # in .env, never in the repo. The image MUST contain python3:
    # deepagents' file helpers run a small Python script inside the
    # sandbox.
    sandbox_enabled: bool = False
    sandbox_domain: str = "localhost:8080"
    sandbox_api_key: str = ""
    sandbox_image: str = "python:3.12-slim"
    # True when the server runs in Docker and this client on the host
    # (sandbox container IPs unreachable — traffic goes via the server
    # proxy instead).
    sandbox_use_server_proxy: bool = False
    # Beat reaps sandboxes idle longer than this (seconds).
    sandbox_idle_ttl_seconds: int = 3600

    # Mission harness index: ``db`` reads missions.execution_config.harness
    # from Postgres (per-row seed YAML fallback when the key is missing);
    # ``seed`` loads only seeds/missions/*.yaml (tests / no DB).
    harness_source: Literal["db", "seed"] = "db"
    harness_index_ttl_seconds: int = 30
    harness_index_db_timeout_seconds: float = 2.0
    # Runtime
    checkpoint_enabled: bool = False
    execution_timeout_seconds: int = 3600
    # Celery: must exceed worst-case task runtime so unacked work is not redelivered early.
    celery_visibility_timeout_seconds: int = 0  # 0 => execution_timeout + 600
    # Mark RUNNING executions stale after this many seconds (reaper / reclaim).
    execution_stale_running_seconds: int = 7200
    # Flow subgraph: each flow may run multiple model+tool steps; 80 is
    # above the historical 50 default to tolerate multi-tool flows
    # without changing the small-model router contract.
    graph_recursion_limit: int = 80
    llm_request_timeout_seconds: int = 120
    llm_request_max_retries: int = 2
    guest_default_locale: str = "en"
    # Guest channel: when reply_to_guest is missing after reprompt (locale templates).
    guest_reply_generic_fallback: str = (
        "We could not deliver a reply at this time. "
        "Please try again or contact the front desk."
    )
    guest_reply_generic_fallback_vi: str = (
        "Hiện chưa gửi được phản hồi. "
        "Vui lòng thử lại hoặc liên hệ lễ tân để được hỗ trợ."
    )
    policy_unavailable_reply_template: str = (
        "The requested policy information is not available in our documents. "
        "Please contact the front desk for assistance."
    )
    policy_unavailable_reply_template_vi: str = (
        "Chưa tìm thấy thông tin chính sách phù hợp trong tài liệu. "
        "Vui lòng liên hệ lễ tân để được hỗ trợ."
    )
    guest_quote_booking_offer: str = (
        "If you would like to book, let us know and we can proceed."
    )
    guest_quote_booking_offer_vi: str = (
        "Nếu anh/chị muốn đặt phòng, vui lòng cho biết để chúng tôi hỗ trợ tiếp."
    )

    # Batch ingest (Option B): files in a local inbox dir, or in the
    # MinIO ingest-inbox bucket when ingest_use_minio is true. Beat
    # runs the batch every ingest_interval_seconds (default 30 min).
    ingest_interval_seconds: int = 1800
    ingest_inbox_dir: str = "ingest/inbox"
    ingest_use_minio: bool = False
    ingest_minio_bucket: str = "ingest-inbox"

    # --- Retrieval / ingestion / projection ops values ---
    # Moved here from module-level literals (2026-11-17 hardcode
    # cleanup): every default below equals the old literal, so
    # behaviour is unchanged unless the env overrides it. Modules
    # keep their old constant names as aliases of these settings.
    # Graph expansion caps + budgets (runtime/retrieval_graph.py).
    retrieval_hub_degree: int = 50
    retrieval_hub_located_in_weight: float = 0.15
    retrieval_hop_decay: float = 0.6
    retrieval_max_fanout_per_type: int = 10
    retrieval_max_expansion_nodes: int = 50
    retrieval_default_timeout_seconds: float = 0.15
    retrieval_live_adjacency_timeout_seconds: float = 0.30
    retrieval_max_live_adjacency_nodes: int = 200
    retrieval_gazetteer_refresh_min_interval_seconds: int = 60
    # One shared retriever per execution (runtime/retrieval_live.py).
    retrieval_max_execution_retrievers: int = 32
    # Context assembly budgets (runtime/retrieval_context.py).
    retrieval_total_budget_tokens: int = 3000
    retrieval_graph_reserve_tokens: int = 200
    retrieval_max_triples: int = 20
    retrieval_max_chunks_per_doc: int = 2
    # Expansion gating thresholds (runtime/retrieval_expansion.py).
    retrieval_max_variants: int = 3
    retrieval_low_top1_vec: float = 0.35
    retrieval_small_margin_vec: float = 0.15
    # Personalized PageRank (runtime/retrieval_ppr.py).
    retrieval_ppr_restart_prob: float = 0.5
    retrieval_ppr_max_iter: int = 200
    retrieval_ppr_tol: float = 1e-9
    # Process embedding cache (runtime/embedding_cache.py).
    embedding_cache_ttl_seconds: int = 300
    embedding_cache_max_entries: int = 512
    # Qdrant payload batching + substrate cache (infra/qdrant_client.py).
    qdrant_payload_batch_size: int = 100
    qdrant_substrate_ttl_seconds: int = 60

    # --- Resolved (effective) values ---
    @property
    def resolved_embedding_api_base(self) -> str:
        return self.embedding_api_base or self.llm_api_base

    @property
    def resolved_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key

    @property
    def resolved_embedding_model_index(self) -> str:
        """Embedding model for index-time work (empty override = shared)."""
        return self.embedding_model_index or self.embedding_model

    @property
    def resolved_embedding_model_query(self) -> str:
        """Embedding model for query-time work (empty override = shared)."""
        return self.embedding_model_query or self.embedding_model

    @property
    def resolved_celery_visibility_timeout(self) -> int:
        if self.celery_visibility_timeout_seconds > 0:
            return self.celery_visibility_timeout_seconds
        return self.execution_timeout_seconds + 600

    @property
    def resolved_minio_presign_endpoint(self) -> str:
        return self.minio_presign_endpoint or self.minio_endpoint


settings = Settings()
