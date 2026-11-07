"""Model Registry — which LLM models the runtime is configured to use.

Primary provider is OpenAI (langchain-openai). Anthropic is the documented
alternative and needs the `anthropic` extra:
`pip install -e ".[anthropic]"` + LLM_PROVIDER=anthropic.
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
    provider = settings.llm_provider
    configured = [
        schemas.ModelInfo(
            provider=provider,
            model=settings.llm_model,
            default=True,
            note="Configured via LLM_PROVIDER / LLM_MODEL.",
        )
    ]
    if provider != "anthropic":
        configured.append(
            schemas.ModelInfo(
                provider="anthropic",
                model="claude-sonnet-4-5",
                default=False,
                note="Alternative provider; requires the anthropic extra and ANTHROPIC_API_KEY.",
            )
        )
    return configured
