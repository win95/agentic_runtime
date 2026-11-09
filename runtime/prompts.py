"""System prompts via the MLflow Prompt Registry (MLflow 3.x).

The Deep Agent's system prompt lives in the registry under the name
``fault_investigator`` so prompt versions can be reviewed, rolled back and
switched by alias without a code deploy (see
``scripts/register_prompts.py``).

Resolution order for ``get_system_prompt()``:
1. ``prompts:/fault_investigator@production`` (when MLflow tracing was
   successfully enabled in this process — i.e. the server is reachable).
2. ``prompts:/fault_investigator@latest``.
3. The hardcoded fallback passed in by the caller.

Any failure at any step falls back silently to the hardcoded default
(a debug/warning log line only, never an exception). The loaded prompt is
cached per process so the registry is not hit on every execution.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PROMPT_NAME = "fault_investigator"

_prompt_cache: dict[str, str] = {}


def get_system_prompt(fallback: str, prompt_name: str = PROMPT_NAME) -> str:
    """Return the system prompt for ``prompt_name`` from the registry.

    ``fallback`` is returned whenever MLflow is not enabled/reachable, the
    prompt is not registered yet, or the registered template is unusable.
    """
    if prompt_name in _prompt_cache:
        return _prompt_cache[prompt_name]
    from infra.mlflow_setup import is_mlflow_tracing_enabled

    if not is_mlflow_tracing_enabled():
        return fallback
    template = _load_from_registry(prompt_name)
    if template is None:
        return fallback
    _prompt_cache[prompt_name] = template
    return template


def _load_from_registry(prompt_name: str) -> str | None:
    try:
        import mlflow.genai as genai
    except Exception as exc:  # noqa: BLE001 — mlflow import problems are fallback
        logger.warning("MLflow prompt registry unavailable (%s); using fallback prompt.", exc)
        return None
    for alias in ("production", "latest"):
        try:
            version = genai.load_prompt(f"prompts:/{prompt_name}@{alias}")
        except Exception as exc:  # noqa: BLE001 — missing prompt/alias/server
            logger.debug("Could not load prompt %s@%s: %s", prompt_name, alias, exc)
            continue
        template = version.template if version is not None else None
        if isinstance(template, str) and template.strip():
            logger.info(
                "Loaded system prompt %s@%s (version %s).", prompt_name, alias, version.version
            )
            return template
        logger.warning(
            "Prompt %s@%s has no usable text template; trying next alias.", prompt_name, alias
        )
    logger.warning(
        "Prompt %s not found in the MLflow registry (@production/@latest); using fallback prompt. "
        "Register it with `python scripts/register_prompts.py`.",
        prompt_name,
    )
    return None


def clear_prompt_cache() -> None:
    """Clear the per-process prompt cache (tests only)."""
    _prompt_cache.clear()
