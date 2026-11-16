"""Guest message locale detection and localized templates (no keyword lists)."""

from __future__ import annotations

import re
import unicodedata


def detect_guest_locale(text: str, *, default: str | None = None) -> str:
    """Infer a BCP-47-ish locale tag from the guest message script usage."""
    from app.config import settings

    fallback = default or settings.guest_default_locale or "en"
    sample = (text or "").strip()[:800]
    if not sample:
        return fallback
    letters = [c for c in sample if unicodedata.category(c).startswith("L")]
    if not letters:
        return fallback
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    if non_ascii / len(letters) >= 0.12:
        return "vi"
    if re.search(r"[\u4e00-\u9fff]", sample):
        return "zh"
    return "en"


def localized_setting(field_base: str, locale: str) -> str:
    """``field_base`` or ``field_base_{locale}`` from app settings."""
    from app.config import settings

    loc = (locale or "").strip().lower() or "en"
    specific = getattr(settings, f"{field_base}_{loc}", None)
    if isinstance(specific, str) and specific.strip():
        return specific.strip()
    generic = getattr(settings, field_base, "")
    return (generic or "").strip()


def guest_language_instruction(locale: str) -> str:
    loc = (locale or "en").strip()
    return (
        f"Write reply_to_guest in the guest's language (locale {loc}), "
        "matching how they wrote their last message."
    )


def guest_reply_reprompt(locale: str) -> str:
    from runtime.guest_output import GUEST_REPLY_TOOL_NAME

    return (
        f"Call {GUEST_REPLY_TOOL_NAME} exactly once with the final message "
        f"for the guest. {guest_language_instruction(locale)}"
    )