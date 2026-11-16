"""Guest-facing output channel — separate from model reasoning/context."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from runtime.reply_grounding import apply_reply_grounding
from runtime.scope import get_bound_guest_id

GUEST_REPLY_TOOL_NAME = "reply_to_guest"
_NGRAM_WORDS = 8
_OVERLAP_RATIO = 0.35
_GUEST_ID_IN_TEXT = re.compile(
    r'"guest_id"\s*:\s*"([^"]+)"|\'guest_id\'\s*:\s*\'([^\']+)\''
)
_DATA_BLOCK_RE = re.compile(r"<data\b[^>]*>.*?</data>", re.IGNORECASE | re.DOTALL)


def guest_reply_tool() -> StructuredTool:
    """Synthetic tool: the only supported guest-answer channel in flow agents."""

    def _record(message: str) -> str:
        return json.dumps({"recorded": True}, ensure_ascii=False)

    return StructuredTool.from_function(
        name=GUEST_REPLY_TOOL_NAME,
        description=(
            "Deliver your final message to the guest. Call exactly once "
            "when you are ready to answer — do not put the guest reply "
            "only in free-form assistant text."
        ),
        func=_record,
    )


def reply_from_tool_calls(messages: list) -> str | None:
    """Last reply_to_guest tool argument in the message list."""
    for message in reversed(messages or []):
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict):
                name = call.get("name")
                args = call.get("args") or {}
            else:
                name = getattr(call, "name", None)
                args = getattr(call, "args", None) or {}
            if name != GUEST_REPLY_TOOL_NAME or not isinstance(args, dict):
                continue
            msg = args.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
    return None


def _word_ngrams(text: str, n: int) -> set[str]:
    words = re.findall(r"\S+", (text or "").lower())
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _dedupe_paragraphs(text: str) -> str:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    return "\n\n".join(out)


def _strip_data_blocks(text: str) -> str:
    return _DATA_BLOCK_RE.sub("", text or "").strip()


def _remove_overlap_sentences(text: str, context_blobs: list[str]) -> str:
    context_ngrams: set[str] = set()
    for blob in context_blobs or []:
        context_ngrams |= _word_ngrams(blob, _NGRAM_WORDS)
    if not context_ngrams:
        return text
    reply_ngrams = _word_ngrams(text, _NGRAM_WORDS)
    if not reply_ngrams:
        return text
    overlap = len(reply_ngrams & context_ngrams) / max(len(reply_ngrams), 1)
    if overlap < _OVERLAP_RATIO:
        return text
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        s = sentence.strip()
        if not s:
            continue
        if _word_ngrams(s, _NGRAM_WORDS) & context_ngrams:
            continue
        kept.append(s)
    return " ".join(kept).strip() if kept else text[:400].strip()


def _allowed_guest_ids(session_id: str | None, input_data: dict | None) -> set[str]:
    allowed: set[str] = set()
    bound = get_bound_guest_id(session_id or "")
    if isinstance(bound, str) and bound:
        allowed.add(bound)
    data = input_data or {}
    for key in ("guest_id",):
        gid = data.get(key)
        if isinstance(gid, str) and gid:
            allowed.add(gid)
    return allowed


def _guest_ids_in_text(text: str) -> set[str]:
    ids: set[str] = set()
    for m in _GUEST_ID_IN_TEXT.finditer(text or ""):
        ids.add(m.group(1) or m.group(2))
    return {g for g in ids if g}


def guard_guest_reply(
    text: str,
    *,
    context_blobs: list[str],
    session_id: str | None = None,
    input_data: dict | None = None,
) -> tuple[str, list[str]]:
    """Return (sanitized_reply, guard_notes)."""
    notes: list[str] = []
    if not text or not str(text).strip():
        return "", notes
    cleaned = _strip_data_blocks(str(text))
    cleaned = _dedupe_paragraphs(cleaned)
    before = cleaned
    cleaned = _remove_overlap_sentences(cleaned, context_blobs)
    if cleaned != before:
        notes.append("Guest reply trimmed: removed prompt/context echo")
    allowed_guests = _allowed_guest_ids(session_id, input_data)
    if allowed_guests:
        mentioned = _guest_ids_in_text(cleaned)
        out_of_scope = mentioned - allowed_guests
        if out_of_scope:
            for gid in sorted(out_of_scope):
                cleaned = cleaned.replace(gid, "[guest]")
            notes.append(
                "Guest reply redacted out-of-scope guest identifiers"
            )
    return cleaned.strip(), notes


def extract_flow_guest_reply(
    messages: list,
    *,
    context_blobs: list[str],
    session_id: str | None = None,
    input_data: dict | None = None,
    fallback_text: str | None = None,
) -> tuple[str, list[str]]:
    """Prefer reply_to_guest tool; fall back to sanitized assistant text."""
    raw = reply_from_tool_calls(messages)
    if raw is None:
        raw = fallback_text or ""
    cleaned, notes = guard_guest_reply(
        raw,
        context_blobs=context_blobs,
        session_id=session_id,
        input_data=input_data,
    )
    cleaned, gnotes = apply_reply_grounding(cleaned, messages)
    notes.extend(gnotes)
    return cleaned, notes