"""Ground guest replies in tool results from the same turn.

Generic engine, parameterized by the agent's grounding config
(``runtime/grounding_config.py``): the module names NO tool — the
agent declares which tools are its quote sources and which JSON
fields are its prices. With an empty config the quote branches do
nothing and the reply passes through unchanged.

The quote *format* (availability + per-night + total, in the guest's
language) is the shape of the configured quote payload — it only runs
when an agent configured quote sources. The amount validation
(``_collect_tool_numbers`` / ``_reply_covers_quote``) is fully
generic: any number the reply states must appear in a tool result.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import ToolMessage

from runtime.grounding_config import GroundingConfig
from runtime.guest_locale import localized_setting
from runtime.tool_message_parse import collect_tool_results

_CURRENCY_AMOUNT = re.compile(
    r"(?P<amount>\d{1,3}(?:[,\s]\d{3})+|\d+(?:\.\d+)?)\s*"
    r"(?P<currency>USD|VND|EUR|usd|vnd|€|\$)",
    re.IGNORECASE,
)
_PLAIN_LARGE_NUMBER = re.compile(r"\b\d{1,3}(?:[,\s]\d{3}){1,}\b|\b\d{5,}\b")


def _collect_tool_numbers(messages: list, price_fields: tuple[str, ...]) -> set[str]:
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
        for field_name in price_fields:
            if not field_name:
                continue
            pattern = rf'"{re.escape(field_name)}"\s*:\s*(\d+)'
            for m in re.finditer(pattern, text):
                nums.add(m.group(1))
    return nums


def _reply_amounts(text: str) -> list[str]:
    amounts: list[str] = []
    for m in _CURRENCY_AMOUNT.finditer(text or ""):
        amounts.append(m.group("amount").replace(" ", "").replace(",", ""))
    for m in _PLAIN_LARGE_NUMBER.finditer(text or ""):
        amounts.append(m.group(0).replace(" ", "").replace(",", ""))
    return amounts


def _quote_tool_data(
    messages: list, config: GroundingConfig
) -> tuple[dict | None, dict | None]:
    avail: dict | None = None
    rates: dict | None = None
    for name, data in collect_tool_results(messages):
        if config.availability_tool and name == config.availability_tool:
            avail = data
        elif config.rates_tool and name == config.rates_tool:
            rates = data
    return avail, rates


def quote_reply_from_tools(
    messages: list,
    *,
    guest_locale: str = "en",
    config: GroundingConfig | None = None,
) -> str:
    """Availability + per-night + total from quote tools (no flow-specific prose)."""
    cfg = config or GroundingConfig()
    if not cfg.quote_tools:
        return ""
    avail, rates = _quote_tool_data(messages, cfg)
    if avail is None and rates is None:
        return ""

    room = None
    available = None
    if isinstance(avail, dict):
        items = avail.get("room_types") or []
        if isinstance(items, list) and items:
            pick = items[0] if isinstance(items[0], dict) else {}
            room = pick.get("name") or (
                pick.get(cfg.room_field) if cfg.room_field else None
            )
            available = pick.get("available_rooms")
            if available is not None and avail.get("available") is False:
                available = 0

    per_night = None
    total = None
    currency = ""
    nights = None
    if isinstance(rates, dict):
        nights = rates.get("nights")
        rate_rows = rates.get(cfg.rates_key) if cfg.rates_key else None
        if isinstance(rate_rows, list) and rate_rows and isinstance(rate_rows[0], dict):
            row = rate_rows[0]
            room = room or row.get("name") or (
                row.get(cfg.room_field) if cfg.room_field else None
            )
            per_night = row.get(cfg.per_night_field) if cfg.per_night_field else None
            total = row.get(cfg.total_price_field) if cfg.total_price_field else None
            currency = str(
                (row.get(cfg.currency_field) if cfg.currency_field else None)
                or currency
                or ""
            )
        elif isinstance(avail, dict) and isinstance(avail.get("room_types"), list):
            row = avail["room_types"][0]
            if isinstance(row, dict):
                per_night = (
                    row.get(cfg.per_night_field) if cfg.per_night_field else None
                )
                currency = str(
                    (row.get(cfg.currency_field) if cfg.currency_field else None)
                    or currency
                    or ""
                )

    if available is None and isinstance(avail, dict):
        available = 1 if avail.get("available") else 0

    parts: list[str] = []
    if guest_locale == "vi":
        if room and available is not None:
            status = "còn phòng" if int(available) > 0 else "hết phòng"
            parts.append(f"{room}: {status} ({available} phòng).")
        if per_night is not None:
            parts.append(f"Giá mỗi đêm: {per_night} {currency}".strip())
        if total is not None and nights is not None:
            parts.append(f"Tổng {nights} đêm: {total} {currency}".strip())
    else:
        if room and available is not None:
            status = "available" if int(available) > 0 else "not available"
            parts.append(f"{room} is {status} ({available} room(s)).")
        if per_night is not None:
            parts.append(f"Price per night: {per_night} {currency}".strip())
        if total is not None and nights is not None:
            parts.append(f"Total for {nights} night(s): {total} {currency}".strip())

    offer = localized_setting("guest_quote_booking_offer", guest_locale)
    body = " ".join(p for p in parts if p)
    if body and offer:
        return f"{body} {offer}"
    return body


def grounded_reply_from_tools(
    messages: list, *, config: GroundingConfig | None = None
) -> str:
    """One-line summary built only from tool JSON (fallback reply)."""
    cfg = config or GroundingConfig()
    quote = quote_reply_from_tools(messages, config=cfg)
    if quote:
        return quote
    fragments: list[str] = []
    for name, data in collect_tool_results(messages):
        if cfg.rates_key:
            rates = data.get(cfg.rates_key)
            if isinstance(rates, list) and rates:
                r0 = rates[0]
                if isinstance(r0, dict):
                    total = (
                        r0.get(cfg.total_price_field)
                        if cfg.total_price_field
                        else None
                    )
                    cur = (
                        (r0.get(cfg.currency_field) if cfg.currency_field else None)
                        or (
                            data.get(cfg.currency_field)
                            if cfg.currency_field
                            else None
                        )
                        or ""
                    )
                    room = (
                        (r0.get(cfg.room_field) if cfg.room_field else None)
                        or r0.get("name")
                        or ""
                    )
                    if total is not None:
                        fragments.append(f"{room}: {total} {cur}".strip())
        if cfg.booking_id_field and cfg.booking_id_field in data:
            fragments.append(f"booking {data[cfg.booking_id_field]}")
    return "; ".join(fragments) if fragments else ""


def _reply_covers_quote(
    reply: str, messages: list, config: GroundingConfig
) -> bool:
    price_fields = tuple(
        f for f in (config.total_price_field, config.per_night_field) if f
    )
    nums = _collect_tool_numbers(messages, price_fields)
    if not nums:
        return True
    mentioned = set(_reply_amounts(reply))
    if not mentioned:
        return False
    return mentioned.issubset(nums) and len(mentioned) >= min(2, len(nums))


def apply_reply_grounding(
    reply: str,
    messages: list,
    *,
    guest_locale: str = "en",
    regenerate_once: bool = True,  # noqa: ARG001
    trust_channel: bool = False,
    config: GroundingConfig | None = None,
) -> tuple[str, list[str]]:
    """Return grounded reply and notes. Falls back to tool-built text."""
    cfg = config or GroundingConfig()
    notes: list[str] = []
    if trust_channel and (reply or "").strip():
        return reply.strip(), notes
    used_quote_tools = bool(
        {name for name, _ in collect_tool_results(messages)} & cfg.quote_tools
    )
    if used_quote_tools:
        quote = quote_reply_from_tools(
            messages, guest_locale=guest_locale, config=cfg
        )
        if quote and (not reply or not _reply_covers_quote(reply, messages, cfg)):
            notes.append("Guest reply composed from quote tool results")
            return quote, notes

    if not reply or not reply.strip():
        built = grounded_reply_from_tools(messages, config=cfg)
        if built:
            notes.append("Guest reply rebuilt from tool results (empty reply)")
        return built, notes
    price_fields = tuple(
        f for f in (cfg.total_price_field, cfg.per_night_field) if f
    )
    allowed = _collect_tool_numbers(messages, price_fields)
    if not allowed:
        return reply, notes
    suspicious = [amt for amt in _reply_amounts(reply) if amt not in allowed]
    if not suspicious:
        return reply, notes
    built = grounded_reply_from_tools(messages, config=cfg)
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
