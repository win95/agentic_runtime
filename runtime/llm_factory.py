"""LLM + embedding model factory — provider switch (Layer 4/6).

Two providers, both OpenAI-compatible, selected by ``LLM_PROVIDER``:

- ``litellm`` (default): the user's LiteLLM proxy. Base URL, key and
  model names come from LLM_API_BASE / LLM_API_KEY / LLM_MODEL.
- ``cloudflare``: Cloudflare Workers AI via its OpenAI-compatible
  /ai/v1 endpoint — no proxy needed. Needs CLOUDFLARE_ACCOUNT_ID
  (not a secret) + CLOUDFLARE_API_TOKEN (a Workers AI API token);
  LLM_MODEL must then be a Workers AI model name such as
  @cf/qwen/qwen3.8-27b and EMBEDDING_MODEL such as
  @cf/qwen/qwen3-embedding-0.6b. See docs/cloudflare-workers-ai.md.

The factory builds `langchain_openai` clients pointed at the
selected provider's base URL. Nothing here is hardcoded: base URL,
key and model names all come from `app.config.settings` (env).
Missing configuration fails FAST with an actionable error at
execution time — never at import time, and never with a silent
fallback to a different model.
"""

from __future__ import annotations

from app.config import Settings, settings


def missing_llm_config(cfg: Settings | None = None) -> list[str]:
    """Env var names that are still missing for chat-model use."""
    cfg = cfg or settings
    missing: list[str] = []
    if cfg.llm_provider == "cloudflare":
        # The resolved base URL is non-empty even without an account
        # id, so the id itself must be checked, not the URL.
        if not cfg.cloudflare_account_id:
            missing.append("CLOUDFLARE_ACCOUNT_ID")
        if not cfg.cloudflare_api_token:
            missing.append("CLOUDFLARE_API_TOKEN")
    else:
        if not cfg.llm_api_base:
            missing.append("LLM_API_BASE")
        if not cfg.llm_api_key:
            missing.append("LLM_API_KEY")
    if not cfg.llm_model:
        missing.append("LLM_MODEL")
    return missing


def missing_embedding_config(cfg: Settings | None = None) -> list[str]:
    """Env var names that are still missing for embedding use."""
    cfg = cfg or settings
    missing: list[str] = []
    if cfg.llm_provider == "cloudflare":
        if not cfg.cloudflare_account_id:
            missing.append("CLOUDFLARE_ACCOUNT_ID")
        if not cfg.cloudflare_api_token and not cfg.embedding_api_key:
            missing.append("CLOUDFLARE_API_TOKEN (or EMBEDDING_API_KEY)")
    else:
        if not cfg.resolved_embedding_api_base:
            missing.append("EMBEDDING_API_BASE (or LLM_API_BASE)")
        if not cfg.resolved_embedding_api_key:
            missing.append("EMBEDDING_API_KEY (or LLM_API_KEY)")
    if not cfg.embedding_model:
        missing.append("EMBEDDING_MODEL")
    return missing


def require_llm_config(cfg: Settings | None = None) -> None:
    """Raise a clear, actionable error when chat config is incomplete."""
    cfg = cfg or settings
    missing = missing_llm_config(cfg)
    if missing:
        if cfg.llm_provider == "cloudflare":
            hint = (
                "LLM_PROVIDER=cloudflare needs CLOUDFLARE_ACCOUNT_ID + "
                "CLOUDFLARE_API_TOKEN in .env (see .env.example): the "
                "account id is not a secret, the token is a Workers AI "
                "API token; LLM_MODEL must be a Workers AI model name "
                "such as @cf/qwen/qwen3.8-27b."
            )
        else:
            hint = (
                "Set them in .env (see .env.example): LLM_API_BASE is "
                "your LiteLLM proxy URL, LLM_API_KEY its key, LLM_MODEL "
                "the model name registered on that proxy."
            )
        raise ValueError("LLM is not configured: missing " + ", ".join(missing) + ". " + hint)


def require_embedding_config(cfg: Settings | None = None) -> None:
    """Raise a clear, actionable error when embedding config is incomplete."""
    cfg = cfg or settings
    missing = missing_embedding_config(cfg)
    if missing:
        if cfg.llm_provider == "cloudflare":
            hint = (
                "LLM_PROVIDER=cloudflare needs CLOUDFLARE_ACCOUNT_ID + "
                "CLOUDFLARE_API_TOKEN in .env (see .env.example); "
                "EMBEDDING_MODEL must be a Workers AI embedding model "
                "such as @cf/qwen/qwen3-embedding-0.6b. EMBEDDING_API_BASE/"
                "EMBEDDING_API_KEY override the provider when set."
            )
        else:
            hint = (
                "Set them in .env (see .env.example): EMBEDDING_MODEL is "
                "the embedding model name registered on your LiteLLM "
                "proxy; EMBEDDING_API_BASE/EMBEDDING_API_KEY default to "
                "the LLM proxy values when left empty."
            )
        raise ValueError(
            "Embeddings are not configured: missing " + ", ".join(missing) + ". " + hint
        )


def build_chat_model(cfg: Settings | None = None):
    """Build the chat model via the selected provider (OpenAI-compatible).

    ``max_tokens`` is the context budget's output reserve
    (OUTPUT_RESERVE_TOKENS): one response can never eat more than the
    tokens set aside for the answer.
    """
    from langchain_openai import ChatOpenAI

    from runtime.context_budget import OUTPUT_RESERVE_TOKENS

    cfg = cfg or settings
    require_llm_config(cfg)
    return ChatOpenAI(
        base_url=cfg.resolved_llm_api_base,
        api_key=cfg.resolved_llm_api_key,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        max_tokens=OUTPUT_RESERVE_TOKENS,
        timeout=cfg.llm_request_timeout_seconds,
        max_retries=cfg.llm_request_max_retries,
    )


def build_embeddings(cfg: Settings | None = None, *, model: str | None = None):
    """Build the embedding model via the selected provider (OpenAI-compatible).

    `model` overrides the configured EMBEDDING_MODEL for one part of the
    pipeline (index vs query); None means the shared configured model.
    The base EMBEDDING_MODEL is still required — the override only
    replaces it for that part, it never supplies the base config.
    """
    from langchain_openai import OpenAIEmbeddings

    cfg = cfg or settings
    require_embedding_config(cfg)
    # The LiteLLM proxy serves a custom embedding model: tiktoken cannot
    # load its tokenizer (embedding calls failed with a 422), and the
    # proxy does not expose a context length to check against — both
    # checks must be off for the proxy to work at all.
    return OpenAIEmbeddings(
        base_url=cfg.resolved_embedding_api_base,
        api_key=cfg.resolved_embedding_api_key,
        model=model or cfg.embedding_model,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
