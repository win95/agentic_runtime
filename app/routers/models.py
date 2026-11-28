"""Model Registry — which LLM models the runtime is configured to use.

Models go through the provider selected by LLM_PROVIDER (OpenAI-compatible):
``litellm`` (default) is the user's LiteLLM proxy; ``cloudflare`` is
Cloudflare Workers AI via its /ai/v1 endpoint. The provider, base URL, key
and model names come from env (see docs/cloudflare-workers-ai.md).
Anthropic is only a documented alternative: point the proxy at it and
register the model name on the proxy — no code change is needed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import schemas
from app.config import settings
from app.core.security import require_api_key

router = APIRouter(
    prefix="/models", tags=["models"], dependencies=[Depends(require_api_key)]
)


def _provider_label() -> str:
    """Provider name as configured (never hardcoded to one provider)."""
    if settings.llm_provider == "cloudflare":
        return "cloudflare (openai-compatible)"
    return "litellm (openai-compatible)"


def _chat_note() -> str:
    if settings.llm_provider == "cloudflare":
        return "Chat model, via Cloudflare Workers AI / LLM_MODEL."
    return "Chat model, via LLM_API_BASE / LLM_MODEL."


def _embedding_note() -> str:
    if settings.llm_provider == "cloudflare":
        return "Embedding model, via Cloudflare Workers AI / EMBEDDING_MODEL."
    return "Embedding model, via EMBEDDING_MODEL."


@router.get("", response_model=list[schemas.ModelInfo])
def list_models():
    provider = _provider_label()
    return [
        schemas.ModelInfo(
            provider=provider,
            model=settings.llm_model,
            default=True,
            note=_chat_note(),
        ),
        schemas.ModelInfo(
            provider=provider,
            model=settings.embedding_model,
            default=False,
            note=_embedding_note(),
        ),
    ]
