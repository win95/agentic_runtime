"""Tavily tools — no live API (mocked client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import runtime.tools  # noqa: F401 — registers tools
from runtime.tools.tavily_tools import tavily_extract, tavily_search


def test_tavily_search_missing_key_no_secret_leak(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    out = tavily_search.invoke({"query": "coffee prices"})
    assert out["results"] == []
    assert "TAVILY_API_KEY" in out["error"]
    assert "super-secret" not in (out.get("error") or "")


def test_tavily_search_defaults_basic_depth(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "super-secret-key")
    client = MagicMock()
    client.search.return_value = {"results": []}
    with patch("tavily.TavilyClient", return_value=client):
        tavily_search.invoke({"query": "test"})
    client.search.assert_called_once()
    assert client.search.call_args.kwargs["search_depth"] == "basic"
    assert "super-secret" not in str(client.search.call_args)


def test_tavily_search_mock_client(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    client = MagicMock()
    client.search.return_value = {
        "results": [
            {"title": "A", "url": "https://a", "content": "snippet", "score": 0.9}
        ],
        "answer": "summary",
    }
    with patch("tavily.TavilyClient", return_value=client):
        out = tavily_search.invoke(
            {"query": "robusta", "include_answer": True, "max_results": 3}
        )
    assert out["count"] == 1
    assert out["results"][0]["url"] == "https://a"
    assert out.get("answer")


def test_tavily_extract_mock_client_basic(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    client = MagicMock()
    client.extract.return_value = {
        "results": [{"url": "https://a", "raw_content": "# Title\nbody"}],
        "failed_results": [],
    }
    with patch("tavily.TavilyClient", return_value=client):
        out = tavily_extract.invoke({"urls": ["https://a"]})
    assert client.extract.call_args.kwargs["extract_depth"] == "basic"
    assert out["pages"][0]["content"].startswith("# Title")