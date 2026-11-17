"""Single source for the local mock hotel MCP port and default URL.

``MOCK_HOTEL_MCP_PORT`` overrides the listen port (server, docs, compose).
``MCP_SERVER_URL`` is the runtime client URL — when unset, manual scripts
fall back to :func:`mock_hotel_mcp_url` on localhost.
"""

from __future__ import annotations

import os

DEFAULT_MOCK_HOTEL_MCP_PORT = 8082


def mock_hotel_mcp_port() -> int:
    """Resolved listen port (default 8082; 8080 is OpenSandbox)."""
    raw = os.environ.get("MOCK_HOTEL_MCP_PORT")
    if raw is None or not str(raw).strip():
        return DEFAULT_MOCK_HOTEL_MCP_PORT
    return int(raw)


def mock_hotel_mcp_url(host: str = "127.0.0.1") -> str:
    """Default ``MCP_SERVER_URL`` for the mock hotel server on *host*."""
    return f"http://{host}:{mock_hotel_mcp_port()}/mcp"


def mock_hotel_mcp_url_for_docker_host() -> str:
    """Default MCP URL when the API/worker run inside Docker Compose."""
    return mock_hotel_mcp_url("host.docker.internal")