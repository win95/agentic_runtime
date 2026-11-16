"""Tool argument validation from registry / JSON Schema metadata."""

from __future__ import annotations

import json
from typing import Any


def _enum_for_field(schema: dict, field: str) -> list | None:
    props = schema.get("properties") or {}
    spec = props.get(field) or {}
    enum = spec.get("enum")
    if isinstance(enum, list) and enum:
        return enum
    return None


def validate_args_against_schema(
    tool_name: str,
    args: dict,
    schema: dict | None,
) -> str | None:
    """Return an error message, or None when args match the schema."""
    if not schema or not isinstance(schema, dict):
        return None
    if not isinstance(args, dict):
        return f"{tool_name}: arguments must be an object"
    required = schema.get("required") or []
    props = schema.get("properties") or {}
    for key in required:
        if key not in args or args.get(key) in (None, ""):
            return f"{tool_name}: missing required argument '{key}'"
    for key, value in args.items():
        if key not in props:
            continue
        enum = _enum_for_field(schema, key)
        if enum and value not in enum:
            return (
                f"{tool_name}: invalid value for '{key}': {value!r} "
                f"(allowed: {enum})"
            )
        spec = props[key]
        expected = spec.get("type")
        if expected == "integer" and isinstance(value, bool):
            return f"{tool_name}: '{key}' must be an integer"
        if expected == "integer" and not isinstance(value, int):
            return f"{tool_name}: '{key}' must be an integer"
    return None


def schema_from_tool(tool) -> dict | None:
    """Best-effort JSON schema for one langchain tool."""
    raw = getattr(tool, "args", None) or {}
    if isinstance(raw, dict) and raw.get("properties"):
        return raw
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        try:
            return args_schema.model_json_schema()
        except Exception:  # noqa: BLE001
            pass
    return None if not raw else raw