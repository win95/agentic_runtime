"""Document-backed replies for policy tool flows (generic, no fixed keyword lists)."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import ToolMessage

from runtime.grounding_config import GroundingConfig
from runtime.guest_locale import localized_setting
from runtime.tool_message_parse import collect_tool_results, tool_call_id_index, tool_content_dict

_TOKEN_RE = re.compile(r"[\w]{3,}", re.UNICODE)
_DIGITS = re.compile(r"\d+")


def policy_unavailable_reply(*, locale: str = "en") -> str:
    return localized_setting("policy_unavailable_reply_template", locale)


def _messages_tool_names(messages: list) -> set[str]:
    from runtime.tool_message_parse import tool_message_name

    names: set[str] = set()
    id_index = tool_call_id_index(messages)
    for message in messages or []:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict) and call.get("name"):
                names.add(str(call["name"]))
    for message in messages or []:
        if isinstance(message, ToolMessage):
            n = tool_message_name(message, id_index)
            if n:
                names.add(n)
    for name, _ in collect_tool_results(messages):
        names.add(name)
    return names


def collect_policy_corpus(messages: list) -> tuple[list[str], dict[str, Any]]:
    """Text snippets from successful policy tool calls in this turn."""
    corpus: list[str] = []
    meta: dict[str, Any] = {"last_search_count": None, "read_ok": False}
    id_index = tool_call_id_index(messages)
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "status", None) == "error":
            continue
        data = tool_content_dict(message, id_index=id_index)
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
    config: GroundingConfig | None = None,
) -> tuple[list[str], list[str]]:
    """One retrieval retry when search returned nothing; optionally read top hit."""
    cfg = config or GroundingConfig()
    notes: list[str] = []
    corpus, meta = collect_policy_corpus(messages)
    if corpus:
        return corpus, notes
    if meta.get("last_search_count") not in (0, None) and not meta.get("read_ok"):
        return corpus, notes
    if not tools_by_name:
        return corpus, notes
    search_tool = tools_by_name.get(cfg.policy_search_tool) if cfg.policy_search_tool else None
    read_tool = tools_by_name.get(cfg.policy_read_tool) if cfg.policy_read_tool else None
    if search_tool is None:
        return corpus, notes
    prior_query = ""
    for message in reversed(messages or []):
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict) and call.get("name") == cfg.policy_search_tool:
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
    blob = " ".join(corpus).casefold()
    if not blob.strip():
        return 0.0
    if len(words) < 2:
        return 1.0
    hits = sum(1 for w in words if w in blob)
    ratio = hits / len(words)
    if ratio >= 0.12:
        return ratio
    sent_digits = set(_DIGITS.findall(sentence))
    if sent_digits and sent_digits.issubset(set(_DIGITS.findall(blob))):
        return 0.5
    return ratio


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _compose_from_corpus(corpus: list[str], guest_message: str, *, max_chars: int = 700) -> str:
    """Select document sentences relevant to the guest question (claim-level)."""
    sentences: list[str] = []
    for part in corpus:
        sentences.extend(_split_sentences(part))
    if not sentences:
        return _best_corpus_excerpt(corpus, max_chars=max_chars)
    qtokens = {t.casefold() for t in _TOKEN_RE.findall(guest_message)}
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        stoks = {t.casefold() for t in _TOKEN_RE.findall(sentence)}
        overlap = len(stoks & qtokens) / max(len(qtokens), 1) if qtokens else 0.0
        scored.append((overlap, sentence))
    scored.sort(
        key=lambda pair: (
            -pair[0],
            -len(_DIGITS.findall(pair[1])),
            -len(pair[1]),
        )
    )
    chosen: list[str] = []
    for sc, sentence in scored:
        if sc > 0 and sentence not in chosen:
            chosen.append(sentence)
        if len(chosen) >= 5:
            break
    for _, sentence in scored:
        if _DIGITS.search(sentence) and sentence not in chosen:
            chosen.append(sentence)
        if len(chosen) >= 6:
            break
    if not chosen:
        chosen = [s for _, s in scored[:4]]
    text = " ".join(chosen).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return text or _best_corpus_excerpt(corpus, max_chars=max_chars)


def _best_corpus_excerpt(corpus: list[str], *, max_chars: int = 600) -> str:
    if not corpus:
        return ""
    text = max(corpus, key=len)
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
    guest_locale: str = "en",
    config: GroundingConfig | None = None,
) -> tuple[str, list[str]]:
    """Ground policy answers in retrieved documents, or refuse to invent content."""
    cfg = config or GroundingConfig()
    notes: list[str] = []
    if not (_messages_tool_names(messages) & cfg.policy_tools):
        return reply, notes
    corpus, extend_notes = maybe_extend_policy_corpus(
        messages,
        guest_message=guest_message,
        tools_by_name=tools_by_name,
        config=cfg,
    )
    notes.extend(extend_notes)
    if not corpus:
        notes.append("policy: no grounded document text — unavailable reply")
        return policy_unavailable_reply(locale=guest_locale), notes
    if not (reply or "").strip():
        notes.append("policy: empty reply — composed from document")
        return _compose_from_corpus(corpus, guest_message), notes
    sentences = _split_sentences(reply)
    kept: list[str] = []
    weak = False
    for sentence in sentences:
        if _sentence_support_score(sentence, corpus) >= 0.12:
            kept.append(sentence)
        else:
            weak = True
    if weak and kept:
        notes.append("policy: dropped ungrounded sentences")
    if kept:
        return " ".join(kept), notes
    notes.append("policy: reply not supported by documents — composed from document")
    return _compose_from_corpus(corpus, guest_message), notes