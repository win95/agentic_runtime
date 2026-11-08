"""Model Registry — which LLM models the runtime is configured to use.

All models go through the user's LiteLLM proxy (OpenAI-compatible).
The proxy URL, key and model names come from env (LLM_API_BASE /
LLM_API_KEY / LLM_MODEL / EMBEDDING_MODEL). Anthropic is only a
documented alternative: point the proxy at it and register the model
name on the proxy — no code change is needed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import schemas
from app.config import settings
from app.core.security import require_api_key

router = APIRouter(
    prefix="/models", tags=["models"], dependencies=[Depends(require_api_key)]
)


@router.get("", response_model=list[schemas.ModelInfo])
def list_models():
    return [
        schemas.ModelInfo(
            provider="litellm (openai-compatible)",
            model=settings.llm_model,
            default=True,
            note="Chat model, via LLM_API_BASE / LLM_MODEL.",
        ),
        schemas.ModelInfo(
            provider="litellm (openai-compatible)",
            model=settings.embedding_model,
            default=False,
            note="Embedding model, via EMBEDDING_MODEL.",
        ),
    ]
