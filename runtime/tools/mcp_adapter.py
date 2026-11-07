"""Generic MCP client adapter (Layer 5).

Loads tools from an MCP server configured ONLY by environment
(`MCP_SERVER_URL`, `MCP_TRANSPORT`). No hardcoded URL or token.
Returns an empty list when no server is configured or unreachable — MCP
tools are an optional extension, never a hard requirement.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import settings

logger = logging.getLogger(__name__)


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
    """Synchronous wrapper for graph nodes (Celery worker context)."""
    if not settings.mcp_server_url:
        return []
    try:
        return asyncio.run(aload_mcp_tools())
    except Exception as exc:  # noqa: BLE001 — MCP is best-effort
        logger.warning("Could not load MCP tools from %s: %s", settings.mcp_server_url, exc)
        return []
