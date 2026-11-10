"""Generic MCP client adapter (Layer 5).

Loads tools from an MCP server configured ONLY by environment
(`MCP_SERVER_URL`, `MCP_TRANSPORT`). No hardcoded URL or token.
Returns an empty list when no server is configured or unreachable — MCP
tools are an optional extension, never a hard requirement.

Layer 4 (prompt-injection defense): MCP tools go through the same
registry permission check as built-in tools. Each MCP tool must be
mapped to a business capability in `MCP_TOOL_CAPABILITIES` (a JSON
object {"tool_name": "capability"}); a tool with no mapping is DENIED
and never handed to an agent, because an MCP server's tool descriptions
are untrusted prompt content (tool-description poisoning). MCP tools
are treated as writes (read_only=False) since their side effects are
unknown to this codebase.
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.config import settings
from runtime.tools.registry import registry

logger = logging.getLogger(__name__)


def _capability_map() -> dict[str, str]:
    """Parse MCP_TOOL_CAPABILITIES into {tool_name: capability}."""
    raw = (settings.mcp_tool_capabilities or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("MCP_TOOL_CAPABILITIES is not valid JSON; denying all MCP tools")
        return {}
    if not isinstance(parsed, dict):
        logger.warning("MCP_TOOL_CAPABILITIES must be a JSON object; denying all MCP tools")
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


async def aload_mcp_tools() -> list:
    """Load tools from the configured MCP server (async)."""
    url = settings.mcp_server_url
    if not url:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {"default": {"url": url, "transport": settings.mcp_transport}}
    )
    tools = await client.get_tools()
    logger.info("Loaded %d MCP tools from %s", len(tools), url)
    return tools


def load_mcp_tools() -> list:
    """Load MCP tools and register the mapped ones in the registry.

    Returns the tools that were registered. Tools with no capability
    mapping are skipped (default deny) with a warning — they are never
    handed to an agent. The caller must then use
    ``registry.get_langchain_tools(context)`` so the registry permission
    check decides which of them a mission may actually use.
    """
    if not settings.mcp_server_url:
        return []
    try:
        tools = asyncio.run(aload_mcp_tools())
    except Exception as exc:  # noqa: BLE001 — MCP is best-effort
        logger.warning("Could not load MCP tools from %s: %s", settings.mcp_server_url, exc)
        return []
    capabilities = _capability_map()
    registered = []
    for tool in tools:
        capability = capabilities.get(tool.name)
        if not capability:
            logger.warning(
                "Denying MCP tool %s: no capability mapping in "
                "MCP_TOOL_CAPABILITIES (default deny)",
                tool.name,
            )
            continue
        registry.register(
            tool, capability=capability, permission="mcp",
            timeout_seconds=60, read_only=False,
        )
        registered.append(tool)
    return registered
