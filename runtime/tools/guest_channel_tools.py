"""Synthetic guest-channel tools registered for layer enforcement."""

from __future__ import annotations

import logging

from runtime.guest_output import guest_reply_tool
from runtime.harness.loader import guest_output_capabilities_from_seed_files
from runtime.tools.registry import registry
from runtime.write_pipeline_runner import confirm_guest_booking_tool

logger = logging.getLogger(__name__)

GUEST_REPLY_TOOL = guest_reply_tool()
CONFIRM_GUEST_BOOKING_TOOL = confirm_guest_booking_tool()


def register_guest_channel_tools_from_seeds() -> None:
    """Register guest tools from mission seed YAML (import-safe, no DB)."""
    for capability in guest_output_capabilities_from_seed_files():
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


register_guest_channel_tools_from_seeds()