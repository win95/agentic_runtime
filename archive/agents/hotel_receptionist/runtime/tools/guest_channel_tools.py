"""Synthetic guest-channel tools registered for layer enforcement."""

from __future__ import annotations

import logging

from runtime.guest_output import guest_reply_tool
from runtime.harness.registry import guest_output_capabilities
from runtime.tools.registry import registry

logger = logging.getLogger(__name__)

GUEST_REPLY_TOOL = guest_reply_tool()


def register_guest_channel_tools_from_seeds() -> None:
    """Register guest tools from the harness guest-output caps.

    Source follows ``HARNESS_SOURCE`` (db rows or seed YAML) via
    ``guest_output_capabilities`` — NOT seed-only any more. Runs at
    import time, so in db mode this does one light DB scan on import.

    The confirm tool is NOT registered here anymore: it is
    session-scoped, and the agent registers its stub lazily under the
    pipeline-id name (``confirm_<pipeline_id>``) when it builds the
    per-session tool — see ``BaseAgent.confirm_tool``. Only the reply
    tool still needs an import-time stub.
    """
    for capability in guest_output_capabilities():
        registry.register(
            GUEST_REPLY_TOOL,
            capability=capability,
            read_only=True,
        )


register_guest_channel_tools_from_seeds()