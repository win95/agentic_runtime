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

Sync entry point: langchain-mcp-adapters tools are async-only (no sync
``func``), but this runtime invokes the agent synchronously
(``run_deep_agent`` -> ``agent.invoke``), so a bare MCP tool raises
``NotImplementedError: StructuredTool does not support sync
invocation`` on first use. ``load_mcp_tools()`` therefore wraps each
tool with a sync ``func`` (fresh event loop per call) while keeping
the async coroutine for async callers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading

from langchain_core.tools import StructuredTool

from app.config import settings
from runtime.tools.registry import registry

logger = logging.getLogger(__name__)


def _run_sync(coro):
    """Run a coroutine from sync code: a fresh event loop when none is
    running, else a worker thread with its own loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    outcome: dict = {}

    def _target() -> None:
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 — re-raised below
            outcome["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _as_sync_tool(mcp_tool):
    """Wrap an async-only MCP tool so the sync agent can call it."""
    async def _coro(**kwargs):
        return await mcp_tool.ainvoke(kwargs)

    def _sync(**kwargs):
        return _run_sync(_coro(**kwargs))

    return StructuredTool.from_function(
        func=_sync,
        coroutine=_coro,
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        args_schema=mcp_tool.args_schema,
    )


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
        sync_tool = _as_sync_tool(tool)
        registry.register(
            sync_tool, capability=capability, permission="mcp",
            timeout_seconds=60, read_only=False,
        )
        registered.append(sync_tool)
    if registered:
        # The registry gained tools after startup: the skill loader's
        # cached tool set is stale — drop it so the next load_skills()
        # re-reads the registry and skill maps keep their MCP tools
        # (otherwise the reception plan trace stays empty).
        try:
            from runtime.skills.loader import reset_known_tools_cache

            reset_known_tools_cache()
        except Exception:  # noqa: BLE001 — validation is best-effort
            logger.debug("Could not reset the skill tool cache")
    return registered
