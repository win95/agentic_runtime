"""Agent definition pieces from MDA (``define_deep_agent`` params).

* ``AgentContext`` — the ``context_schema``: the shape of the runtime
  context every run carries (built in ``deep_agent_node``). Declared
  once, validated at run start, and passed to ``create_deep_agent``
  so the compiled agent declares it. A malformed context fails fast
  with a clear error instead of reaching the agent half-built.
* ``resolve_response_format`` — the ``response_format``: opt-in
  structured final output. ``True`` reuses the mission's
  ``output_schema``; a dict is an explicit JSON schema. Default off:
  the 3B model is unreliable at emitting a whole JSON object in one
  call, and the file-output (analytis) and guest-text (hotel) agents
  do not want a structured final message at all.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentContext(TypedDict, total=False):
    """Runtime context of one run (the ``context_schema``).

    ``mission_id`` / ``user_id`` may be None (schedule / service-key
    runs); everything else is always present.
    """

    mission_id: str | None
    execution_id: str
    tenant: str
    capabilities: list[str]
    input: dict
    execution_config: dict
    session_id: str
    user_id: str | None


def validate_agent_context(context: Any) -> list[str]:
    """Check a context dict against AgentContext. Empty list = valid."""
    problems: list[str] = []
    if not isinstance(context, dict):
        return ["agent context must be a dict"]
    if not isinstance(context.get("execution_id"), str):
        problems.append("agent context: execution_id must be a string")
    if not isinstance(context.get("tenant"), str) or not context.get("tenant"):
        problems.append("agent context: tenant must be a non-empty string")
    if not isinstance(context.get("capabilities"), list):
        problems.append("agent context: capabilities must be a list")
    if not isinstance(context.get("input"), dict):
        problems.append("agent context: input must be a dict")
    if not isinstance(context.get("execution_config"), dict):
        problems.append("agent context: execution_config must be a dict")
    if not isinstance(context.get("session_id"), str):
        problems.append("agent context: session_id must be a string")
    if context.get("mission_id") is not None and not isinstance(
        context.get("mission_id"), str
    ):
        problems.append("agent context: mission_id must be a string or None")
    if context.get("user_id") is not None and not isinstance(
        context.get("user_id"), str
    ):
        problems.append("agent context: user_id must be a string or None")
    return problems


def resolve_response_format(raw: Any, output_schema: Any) -> dict | None:
    """Normalize ``execution.response_format`` into a ``create_deep_agent`` value.

    None/False -> None (off). True -> the mission's output_schema
    (must be an object schema with properties). A dict -> used as-is
    (must look like a schema). Anything else, or a schema without
    properties, raises ValueError so a bad declaration fails fast.
    """
    if raw is None or raw is False:
        return None
    if raw is True:
        if not isinstance(output_schema, dict) or not isinstance(
            output_schema.get("properties"), dict
        ) or not output_schema["properties"]:
            raise ValueError(
                "execution.response_format: true needs a mission output_schema "
                "with non-empty properties"
            )
        return dict(output_schema)
    if isinstance(raw, dict):
        if not raw.get("properties") and raw.get("type") != "object":
            raise ValueError(
                "execution.response_format must be an object schema "
                "(a dict with properties)"
            )
        return dict(raw)
    raise ValueError(
        "execution.response_format must be true or an object schema dict, "
        f"got {raw!r}"
    )
