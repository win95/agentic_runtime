"""Synthetic guest-channel tools registered for layer enforcement."""

from __future__ import annotations

from runtime.guest_output import guest_reply_tool
from runtime.tools.registry import registry
from runtime.write_pipeline_runner import confirm_guest_booking_tool

GUEST_REPLY_TOOL = guest_reply_tool()
CONFIRM_GUEST_BOOKING_TOOL = confirm_guest_booking_tool()
_REGISTERED: set[str] = set()


def ensure_guest_channel_tools_registered() -> None:
    from runtime.harness.registry import guest_output_capabilities

    for capability in guest_output_capabilities():
        if capability in _REGISTERED:
            continue
        registry.register(
            GUEST_REPLY_TOOL,
            capability=capability,
            read_only=True,
        )
        registry.register(
            CONFIRM_GUEST_BOOKING_TOOL,
            capability=capability,
            read_only=True,
        )
        _REGISTERED.add(capability)