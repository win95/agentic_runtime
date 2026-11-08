"""LLM + embedding model factory — LiteLLM, OpenAI-compatible (Layer 4/6).

All access goes through the user's LiteLLM proxy, which speaks the
OpenAI protocol. The factory builds `langchain_openai` clients pointed at
the proxy URL with the model names registered on that proxy.

Nothing here is hardcoded: base URL, key and model names all come from
`app.config.settings` (env). Missing configuration fails FAST with an
actionable error at execution time — never at import time, and never with
a silent fallback to a different model.
"""

from __future__ import annotations

from app.config import Settings, settings


def missing_llm_config(cfg: Settings | None = None) -> list[str]:
    """Env var names that are still missing for chat-model use."""
    cfg = cfg or settings
    missing: list[str] = []
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
    if not cfg.resolved_embedding_api_base:
        missing.append("EMBEDDING_API_BASE (or LLM_API_BASE)")
    if not cfg.resolved_embedding_api_key:
        missing.append("EMBEDDING_API_KEY (or LLM_API_KEY)")
    if not cfg.embedding_model:
        missing.append("EMBEDDING_MODEL")
    return missing


def require_llm_config(cfg: Settings | None = None) -> None:
    """Raise a clear, actionable error when chat config is incomplete."""
    missing = missing_llm_config(cfg)
    if missing:
        raise ValueError(
            "LLM is not configured: missing "
            + ", ".join(missing)
            + ". Set them in .env (see .env.example): "
            "LLM_API_BASE is your LiteLLM proxy URL, LLM_API_KEY its key, "
            "LLM_MODEL the model name registered on that proxy."
        )


def require_embedding_config(cfg: Settings | None = None) -> None:
    """Raise a clear, actionable error when embedding config is incomplete."""
    missing = missing_embedding_config(cfg)
    if missing:
        raise ValueError(
            "Embeddings are not configured: missing "
            + ", ".join(missing)
            + ". Set them in .env (see .env.example): "
            "EMBEDDING_MODEL is the embedding model name registered on "
            "your LiteLLM proxy; EMBEDDING_API_BASE/EMBEDDING_API_KEY "
            "default to the LLM proxy values when left empty."
        )


def build_chat_model(cfg: Settings | None = None):
    """Build the chat model via the LiteLLM proxy (OpenAI-compatible)."""
    from langchain_openai import ChatOpenAI

    cfg = cfg or settings
    require_llm_config(cfg)
    return ChatOpenAI(
        base_url=cfg.llm_api_base,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
    )


def build_embeddings(cfg: Settings | None = None):
    """Build the embedding model via the LiteLLM proxy (OpenAI-compatible)."""
    from langchain_openai import OpenAIEmbeddings

    cfg = cfg or settings
    require_embedding_config(cfg)
    return OpenAIEmbeddings(
        base_url=cfg.resolved_embedding_api_base,
        api_key=cfg.resolved_embedding_api_key,
        model=cfg.embedding_model,
    )
