"""Defaults for the mock hotel MCP port / URL (single source)."""

from __future__ import annotations

from scripts.mock_hotel_mcp_defaults import (
    DEFAULT_MOCK_HOTEL_MCP_PORT,
    mock_hotel_mcp_port,
    mock_hotel_mcp_url,
    mock_hotel_mcp_url_for_docker_host,
)


def test_default_mock_port_is_8082():
    assert DEFAULT_MOCK_HOTEL_MCP_PORT == 8082


def test_mock_hotel_mcp_url_default(monkeypatch):
    monkeypatch.delenv("MOCK_HOTEL_MCP_PORT", raising=False)
    assert mock_hotel_mcp_port() == 8082
    assert mock_hotel_mcp_url() == "http://127.0.0.1:8082/mcp"
    assert mock_hotel_mcp_url_for_docker_host() == (
        "http://host.docker.internal:8082/mcp"
    )


def test_mock_hotel_mcp_port_env_override(monkeypatch):
    monkeypatch.setenv("MOCK_HOTEL_MCP_PORT", "9099")
    assert mock_hotel_mcp_port() == 9099
    assert mock_hotel_mcp_url() == "http://127.0.0.1:9099/mcp"