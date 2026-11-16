"""Document-backed replies for policy tool flows (generic, no fixed keyword lists)."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import ToolMessage

POLICY_SEARCH_TOOL = "search_policy_documents"
POLICY_READ_TOOL = "read_policy_document"
POLICY_TOOLS = frozenset({POLICY_SEARCH_TOOL, POLICY_READ_TOOL})

_POLICY_UNAVAILABLE = (
    "Em chưa tìm thấy thông tin chính sách phù hợp trong tài liệu của khách sạn. "
    "Anh/chị vui lòng liên hệ lễ tân để được hỗ trợ chi tiết."
)

_TOKEN_RE = re.compile(r"[\w]{3,}", re.UNICODE)


def _tool_content_dict(message: Any) -> dict | None:
    if not isinstance(message, ToolMessage):
        return None
    content = getattr(message, "content", "")
    if isinstance(content, dict):
        return content
    if isinstance(content, str) and content.strip().startswith("{"):
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _messages_tool_names(messages: list) -> set[str]:
    names: set[str] = set()
    for message in messages or []:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict) and call.get("name"):
                names.add(str(call["name"]))
    for message in messages or []:
        if isinstance(message, ToolMessage):
            n = getattr(message, "name", None)
            if n:
                names.add(str(n))
    return names


def collect_policy_corpus(messages: list) -> tuple[list[str], dict[str, Any]]:
    """Text snippets from successful policy tool calls in this turn."""
    corpus: list[str] = []
    meta: dict[str, Any] = {"last_search_count": None, "read_ok": False}
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "status", None) == "error":
            continue
        data = _tool_content_dict(message)
        if not isinstance(data, dict) or data.get("error"):
            continue
        if "documents" in data and isinstance(data.get("count"), int):
            meta["last_search_count"] = data["count"]
        body = data.get("content")
        if isinstance(body, str) and body.strip():
            meta["read_ok"] = True
            corpus.append(body.strip())
        title = data.get("title")
        if isinstance(title, str) and title.strip():
            corpus.append(title.strip())
    return corpus, meta


def search_query_from_guest_message(guest_message: str, *, prior_query: str = "") -> str:
    """Generic query: substantive tokens from the guest message (no hotel keyword lists)."""
    text = (guest_message or "").strip()
    tokens = _TOKEN_RE.findall(text)
    seen: set[str] = set()
    parts: list[str] = []
    for tok in tokens:
        key = tok.casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(tok)
    if parts:
        return " ".join(parts[:16])
    pq = (prior_query or "").strip()
    return pq or text[:120]


def maybe_extend_policy_corpus(
    messages: list,
    *,
    guest_message: str,
    tools_by_name: dict | None,
) -> tuple[list[str], list[str]]:
    """One retrieval retry when search returned nothing; optionally read top hit."""
    notes: list[str] = []
    corpus, meta = collect_policy_corpus(messages)
    if corpus:
        return corpus, notes
    if meta.get("last_search_count") not in (0, None) and not meta.get("read_ok"):
        return corpus, notes
    if not tools_by_name:
        return corpus, notes
    search_tool = tools_by_name.get(POLICY_SEARCH_TOOL)
    read_tool = tools_by_name.get(POLICY_READ_TOOL)
    if search_tool is None:
        return corpus, notes
    prior_query = ""
    for message in reversed(messages or []):
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict) and call.get("name") == POLICY_SEARCH_TOOL:
                args = call.get("args") or {}
                if isinstance(args, dict) and args.get("query"):
                    prior_query = str(args["query"])
                    break
    query = search_query_from_guest_message(guest_message, prior_query=prior_query)
    if not query.strip():
        return corpus, notes
    try:
        result = search_tool.invoke({"query": query})
    except Exception as exc:  # noqa: BLE001
        notes.append(f"policy retry search failed: {exc}")
        return corpus, notes
    if not isinstance(result, dict) or result.get("error"):
        return corpus, notes
    count = int(result.get("count") or 0)
    notes.append("policy: retried search from guest message tokens")
    if count <= 0 or read_tool is None:
        return corpus, notes
    docs = result.get("documents") or []
    if not docs or not isinstance(docs[0], dict):
        return corpus, notes
    doc_id = docs[0].get("doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        return corpus, notes
    try:
        doc = read_tool.invoke({"doc_id": doc_id})
    except Exception as exc:  # noqa: BLE001
        notes.append(f"policy auto-read failed: {exc}")
        return corpus, notes
    if isinstance(doc, dict) and not doc.get("error"):
        body = doc.get("content")
        if isinstance(body, str) and body.strip():
            corpus.append(body.strip())
            notes.append("policy: read top document after retry search")
    return corpus, notes


def _sentence_support_score(sentence: str, corpus: list[str]) -> float:
    words = {w.casefold() for w in _TOKEN_RE.findall(sentence)}
    if len(words) < 2:
        return 1.0
    blob = " ".join(corpus).casefold()
    if not blob.strip():
        return 0.0
    hits = sum(1 for w in words if w in blob)
    return hits / len(words)


def _best_corpus_excerpt(corpus: list[str], *, max_chars: int = 600) -> str:
    if not corpus:
        return _POLICY_UNAVAILABLE
    text = corpus[0]
    for part in corpus:
        if len(part) > len(text):
            text = part
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def apply_policy_grounding(
    reply: str,
    messages: list,
    *,
    guest_message: str = "",
    tools_by_name: dict | None = None,
) -> tuple[str, list[str]]:
    """Ground policy answers in retrieved documents, or refuse to invent content."""
    notes: list[str] = []
    if not (_messages_tool_names(messages) & POLICY_TOOLS):
        return reply, notes
    corpus, extend_notes = maybe_extend_policy_corpus(
        messages,
        guest_message=guest_message,
        tools_by_name=tools_by_name,
    )
    notes.extend(extend_notes)
    if not corpus:
        notes.append("policy: no grounded document text — unavailable reply")
        return _POLICY_UNAVAILABLE, notes
    if not (reply or "").strip():
        notes.append("policy: empty reply — using document excerpt")
        return _best_corpus_excerpt(corpus), notes
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reply) if s.strip()]
    kept: list[str] = []
    weak = False
    for sentence in sentences:
        if _sentence_support_score(sentence, corpus) >= 0.2:
            kept.append(sentence)
        else:
            weak = True
    if weak and kept:
        notes.append("policy: dropped ungrounded sentences")
    if kept:
        return " ".join(kept), notes
    notes.append("policy: reply not supported by documents — excerpt fallback")
    return _best_corpus_excerpt(corpus), notes