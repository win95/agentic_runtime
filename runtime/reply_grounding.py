"""Ground guest replies in tool results from the same turn."""

from __future__ import annotations

import json
import re
from langchain_core.messages import AIMessage, ToolMessage

_CURRENCY_AMOUNT = re.compile(
    r"(?P<amount>\d{1,3}(?:[,\s]\d{3})+|\d+(?:\.\d+)?)\s*"
    r"(?P<currency>USD|VND|EUR|usd|vnd|€|\$)",
    re.IGNORECASE,
)
_PLAIN_LARGE_NUMBER = re.compile(r"\b\d{1,3}(?:[,\s]\d{3}){1,}\b|\b\d{5,}\b")


def _collect_tool_numbers(messages: list) -> set[str]:
    """Numeric strings present in successful tool outputs this turn."""
    nums: set[str] = set()
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "status", None) == "error":
            continue
        content = getattr(message, "content", "")
        text = ""
        if isinstance(content, dict):
            text = json.dumps(content, ensure_ascii=False, default=str)
        elif isinstance(content, str):
            text = content
        for m in _CURRENCY_AMOUNT.finditer(text):
            nums.add(m.group("amount").replace(" ", "").replace(",", ""))
        for m in _PLAIN_LARGE_NUMBER.finditer(text):
            nums.add(m.group(0).replace(" ", "").replace(",", ""))
        for m in re.finditer(r'"total_price"\s*:\s*(\d+)', text):
            nums.add(m.group(1))
        for m in re.finditer(r'"price_per_night"\s*:\s*(\d+)', text):
            nums.add(m.group(1))
    return nums


def _reply_amounts(text: str) -> list[str]:
    amounts: list[str] = []
    for m in _CURRENCY_AMOUNT.finditer(text or ""):
        amounts.append(m.group("amount").replace(" ", "").replace(",", ""))
    for m in _PLAIN_LARGE_NUMBER.finditer(text or ""):
        amounts.append(m.group(0).replace(" ", "").replace(",", ""))
    return amounts


def grounded_reply_from_tools(messages: list) -> str:
    """One-line summary built only from tool JSON (fallback reply)."""
    fragments: list[str] = []
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "status", None) == "error":
            continue
        data = getattr(message, "content", None)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (TypeError, ValueError):
                data = None
        if not isinstance(data, dict) or data.get("error"):
            continue
        rates = data.get("rates")
        if isinstance(rates, list) and rates:
            r0 = rates[0]
            if isinstance(r0, dict):
                total = r0.get("total_price")
                cur = r0.get("currency") or data.get("currency") or ""
                room = r0.get("room_type") or r0.get("name") or ""
                if total is not None:
                    fragments.append(
                        f"{room}: {total} {cur}".strip()
                    )
        if "booking_id" in data:
            fragments.append(f"booking {data['booking_id']}")
    return "; ".join(fragments) if fragments else ""


def apply_reply_grounding(
    reply: str,
    messages: list,
    *,
    regenerate_once: bool = True,  # noqa: ARG001 — reserved for future LLM regen
) -> tuple[str, list[str]]:
    """Return grounded reply and notes. Falls back to tool-built text."""
    notes: list[str] = []
    if not reply or not reply.strip():
        built = grounded_reply_from_tools(messages)
        if built:
            notes.append("Guest reply rebuilt from tool results (empty reply)")
        return built, notes
    allowed = _collect_tool_numbers(messages)
    if not allowed:
        return reply, notes
    suspicious = [
        amt for amt in _reply_amounts(reply) if amt not in allowed
    ]
    if not suspicious:
        return reply, notes
    built = grounded_reply_from_tools(messages)
    if built:
        notes.append(
            "Guest reply replaced: stated amounts not supported by tool data"
        )
        return built, notes
    notes.append("Guest reply trimmed ungrounded currency amounts")
    cleaned = reply
    for amt in suspicious:
        cleaned = re.sub(
            rf"\b{re.escape(amt)}\b\s*(USD|VND|EUR|\$)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    return cleaned.strip() or reply, notes