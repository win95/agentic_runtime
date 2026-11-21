"""Tavily web-search tools (Layer 5) — for the agent-analytis agent.

Search and page extraction via the Tavily API — the search provider the
Deep Agents data-analysis / deep-research / content-builder guides use.
Both tools are read-only. The Tavily client is imported lazily and the
API key is read from ``TAVILY_API_KEY`` at call time, so importing this
module never fails when the package or the key is missing — the agent
gets a structured error dict instead, per the codebase's tool-failure
pattern.
"""

from __future__ import annotations

import os

from langchain_core.tools import tool

from runtime.tools.registry import registry

_MAX_SEARCH_RESULTS = 10
_MAX_EXTRACT_URLS = 3
_SEARCH_CONTENT_CHARS = 2000
_EXTRACT_PAGE_CHARS = 6000


def _client():
    """Build a Tavily client, or raise a clear error when unusable."""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set — add it to .env (see .env.example); "
            "web search stays unavailable until then"
        )
    try:
        from tavily import TavilyClient
    except ImportError as exc:  # pragma: no cover - missing optional dep
        raise RuntimeError(
            "tavily-python is not installed — run `pip install -r requirements.txt`"
        ) from exc
    return TavilyClient(api_key=api_key)


@tool
def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = False,
) -> dict:
    """Search the web with Tavily; returns titles, URLs and snippets.

    Use for any fact, figure or source that must come from the live web.
    Snippets are capped so the result stays small — call tavily_extract
    on the most relevant URLs for full page text.
    """
    if not query or not query.strip():
        return {"query": query, "results": [], "error": "query must not be empty",
                "stub": False, "provenance": "tavily:search"}
    count = max(1, min(int(max_results), _MAX_SEARCH_RESULTS))
    depth = search_depth if search_depth in ("basic", "advanced") else "basic"
    try:
        client = _client()
        response = client.search(
            query=query.strip(),
            search_depth=depth,
            max_results=count,
            include_answer=include_answer,
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001 — provider failure is data, not a crash
        return {"query": query, "results": [], "error": str(exc)[:500],
                "stub": False, "provenance": "tavily:search"}
    results = []
    for item in (response or {}).get("results") or []:
        if not isinstance(item, dict):
            continue
        results.append({
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "content": str(item.get("content") or "")[:_SEARCH_CONTENT_CHARS],
            "score": item.get("score"),
        })
    out: dict = {"query": query, "results": results, "count": len(results),
                 "stub": False, "provenance": "tavily:search"}
    answer = (response or {}).get("answer")
    if include_answer and answer:
        out["answer"] = str(answer)[:_SEARCH_CONTENT_CHARS]
    return out


@tool
def tavily_extract(
    urls: list[str],
    query: str = "",
    extract_depth: str = "basic",
) -> dict:
    """Fetch the full text of up to 3 web pages with Tavily.

    Use after tavily_search to read the pages behind the most relevant
    results. Pass `query` to focus the extraction on what matters for
    the analysis. Each page is capped so the result stays usable.
    """
    cleaned = [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]
    if not cleaned:
        return {"urls": [], "pages": [], "error": "urls must not be empty",
                "stub": False, "provenance": "tavily:extract"}
    picked = cleaned[:_MAX_EXTRACT_URLS]
    depth = extract_depth if extract_depth in ("basic", "advanced") else "basic"
    try:
        client = _client()
        response = client.extract(
            urls=picked,
            query=query.strip() or None,
            extract_depth=depth,
            format="markdown",
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 — provider failure is data, not a crash
        return {"urls": picked, "pages": [], "error": str(exc)[:500],
                "stub": False, "provenance": "tavily:extract"}
    pages = []
    for item in (response or {}).get("results") or []:
        if not isinstance(item, dict):
            continue
        pages.append({
            "url": item.get("url") or "",
            "content": str(item.get("raw_content") or "")[:_EXTRACT_PAGE_CHARS],
        })
    failed = (response or {}).get("failed_results") or []
    return {"urls": picked, "pages": pages, "failed": failed,
            "stub": False, "provenance": "tavily:extract"}


# Register both tools under the web-research capability — read-only (Layer 4).
registry.register(tavily_search, capability="web_research",
                  permission="web:read", timeout_seconds=60, read_only=True)
registry.register(tavily_extract, capability="web_research",
                  permission="web:read", timeout_seconds=90, read_only=True)
