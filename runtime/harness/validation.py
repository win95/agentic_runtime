"""Cross-mission harness validation and deny-message template checks."""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.harness.schema import MissionHarness

_ALLOWED_DENY_FIELDS = frozenset({"tool_name"})


def validate_deny_message_template(template: str) -> None:
    """Only ``{tool_name}`` may appear — no format specs, attrs, or indices."""
    if not template.strip():
        raise ValueError("deny_message_template is empty")
    if "{{" in template or "}}" in template:
        raise ValueError("deny_message_template: escaped braces not allowed")
    if template.count("{") != 1 or template.count("}") != 1:
        raise ValueError("deny_message_template: must contain exactly {tool_name}")
    formatter = string.Formatter()
    try:
        for _literal, field_name, format_spec, conversion in formatter.parse(template):
            if field_name is None:
                continue
            base = field_name.split(".")[0].split("[")[0]
            if base not in _ALLOWED_DENY_FIELDS:
                raise ValueError(
                    f"deny_message_template: only {{tool_name}} allowed, got {{{field_name}}}"
                )
            if format_spec or conversion or field_name != base:
                raise ValueError(
                    "deny_message_template: format specs and field access not allowed"
                )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"deny_message_template: invalid: {exc}") from exc


def format_deny_message(template: str, tool_name: str) -> str:
    validate_deny_message_template(template)
    return template.replace("{tool_name}", tool_name)


def validate_cross_mission_harness(
    missions: list[tuple[str, "MissionHarness"]],
) -> None:
    """Fail seed when guest-chat or guest-output policies conflict."""
    guest_chat: list[str] = []
    guest_output_missions: list[tuple[str, str]] = []
    for mission_id, harness in missions:
        if harness.guest_channel.exposes_public_chat:
            guest_chat.append(mission_id)
        if harness.guest_output.register_reply_tools:
            cap = harness.guest_output.capability
            if not cap:
                raise ValueError(
                    f"mission {mission_id}: guest_output.capability required when "
                    "register_reply_tools is true"
                )
            guest_output_missions.append((mission_id, cap))
    if len(guest_output_missions) > 1:
        caps = {c for _, c in guest_output_missions}
        if len(caps) > 1:
            raise ValueError(
                "guest_output.register_reply_tools may not use different capabilities "
                f"across missions: {guest_output_missions}"
            )
        raise ValueError(
            "only one mission may set guest_output.register_reply_tools; "
            f"found {[m for m, _ in guest_output_missions]}"
        )
    if len(guest_chat) > 1:
        raise ValueError(
            "only one mission may set guest_channel.exposes_public_chat; "
            f"found {guest_chat}"
        )