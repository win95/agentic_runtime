"""Synthetic guest-channel tools registered for layer enforcement."""

from __future__ import annotations

import logging

from runtime.guest_output import guest_reply_tool
from runtime.harness.loader import guest_output_capabilities_from_seed_files
from runtime.tools.registry import registry

logger = logging.getLogger(__name__)

GUEST_REPLY_TOOL = guest_reply_tool()


def register_guest_channel_tools_from_seeds() -> None:
    """Register guest tools from mission seed YAML (import-safe, no DB).

    The confirm tool is NOT registered here anymore: it is
    session-scoped, and the agent registers its stub lazily under the
    pipeline-id name (``confirm_<pipeline_id>``) when it builds the
    per-session tool — see ``BaseAgent.confirm_tool``. Only the reply
    tool still needs an import-time stub.
    """
    for capability in guest_output_capabilities_from_seed_files():
        registry.register(
            GUEST_REPLY_TOOL,
            capability=capability,
            read_only=True,
        )


register_guest_channel_tools_from_seeds()