"""Prompt-injection defense helpers (docs/prompt-injection-defense.md).

Shared by the defense layers so each one is a thin call site:

- Layer 1 — execution-input validation (size, field length, ``asset_id``
  allowlist, date shape, control characters).
- Layer 2 — ``<data>`` delimiters that separate untrusted data from
  instructions in the agent prompt.
- Layer 3 — truncation + whitespace normalization of tool-output text.
- Layer 6 — heuristic detection of instruction-like text (signals only,
  never a block — see ``detect_injection_signals``).

Nothing here is a guarantee: delimiter + system-prompt defenses lower
the success rate of injected instructions, they do not make it zero.
Layers 5–7 (output validation, detection, damage limits) exist because
of that.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date

# --- Layer 1 limits ---
MAX_INPUT_JSON_BYTES = 8192
MAX_STRING_FIELD_CHARS = 500
ASSET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Session ids (docs/sessions.md) use the same allowlist shape as
# asset ids: short, URL-safe, no separators that could break the
# session-history rendering or a path parameter.
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_valid_session_id(value: str | None) -> bool:
    """True when ``value`` is a usable session id (None is not valid —
    callers that accept an omitted id generate one instead)."""
    return isinstance(value, str) and bool(SESSION_ID_RE.match(value))

# --- Layer 3 limits ---
MAX_TEXT_FIELD_CHARS = 2000
MAX_DOCUMENT_CONTENT_CHARS = 4000
# Free-text fields that get the tighter cap (anything else long falls
# back to MAX_TEXT_FIELD_CHARS; document bodies get the larger cap).
_LONG_TEXT_FIELDS = {"content": MAX_DOCUMENT_CONTENT_CHARS}


def strip_control_chars(value: str) -> str:
    """Remove Unicode control characters (keeps tab/newline)."""
    return "".join(
        ch
        for ch in value
        if ch in ("\t", "\n") or unicodedata.category(ch) != "Cc"
    )


def sanitize_input_value(value):
    """Recursively strip control characters from an input value."""
    if isinstance(value, str):
        return strip_control_chars(value)
    if isinstance(value, dict):
        return {k: sanitize_input_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_input_value(v) for v in value]
    return value


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        pass
    try:
        return date.fromisoformat(value[:10])
    except (ValueError, TypeError, IndexError):
        return None


def validate_execution_input_data(mission: dict, input_data: dict) -> list[str]:
    """Validate one execution input. Returns a list of problems (empty = ok).

    Checks, in order: required fields present, total JSON size, per-field
    string length, ``asset_id`` allowlist, ``start_date``/``end_date``
    shape and ordering. Callers sanitize control characters first
    (see ``sanitize_input_value``).
    """
    problems: list[str] = []
    input_schema = (mission or {}).get("input_schema") or {}
    required = input_schema.get("required") or []
    missing = [f for f in required if f not in input_data]
    if missing:
        problems.append(f"Missing required input fields: {missing}")
    try:
        size = len(json.dumps(input_data, ensure_ascii=False))
    except (TypeError, ValueError):
        size = MAX_INPUT_JSON_BYTES + 1
    if size > MAX_INPUT_JSON_BYTES:
        problems.append(
            f"input too large: {size} bytes (max {MAX_INPUT_JSON_BYTES})"
        )
    for key, value in input_data.items():
        if not isinstance(value, str):
            continue
        if len(value) > MAX_STRING_FIELD_CHARS:
            problems.append(
                f"field '{key}' too long: {len(value)} chars "
                f"(max {MAX_STRING_FIELD_CHARS})"
            )
        if key == "asset_id" and not ASSET_ID_RE.match(value):
            problems.append(
                "field 'asset_id' must match ^[A-Za-z0-9_-]{1,64}$"
            )
    start = input_data.get("start_date")
    end = input_data.get("end_date")
    start_date = _parse_date(start) if isinstance(start, str) else None
    end_date = _parse_date(end) if isinstance(end, str) else None
    if isinstance(start, str) and start_date is None:
        problems.append(f"field 'start_date' is not a valid date: {start!r}")
    if isinstance(end, str) and end_date is None:
        problems.append(f"field 'end_date' is not a valid date: {end!r}")
    if start_date is not None and end_date is not None and end_date < start_date:
        problems.append(
            f"field 'end_date' ({end}) is before 'start_date' ({start})"
        )
    return problems


# --- Layer 2: data delimiters ---
def wrap_data(source: str, text: str, item_id: str | None = None) -> str:
    """Wrap untrusted text in a <data> block so the model can tell data
    from instructions. The system prompt (runtime/deep_agent.py) tells
    the agent to cite, never obey, content inside these blocks.

    The text itself is escaped first: a literal ``</data>`` inside
    hostile content would otherwise close the block early and let the
    trailing text pose as instructions outside the delimiter.
    """
    safe_text = text.replace("</data>", "<\\/data>")
    label = f' source="{source}"' + (f' id="{item_id}"' if item_id else "")
    return f"<data{label}>\n{safe_text}\n</data>"


# --- Layer 3: tool-output hygiene ---
def truncate_text(value: str, max_chars: int = MAX_TEXT_FIELD_CHARS) -> str:
    """Collapse whitespace and cut a long free-text field, marking the cut."""
    cleaned = re.sub(r"\s+", " ", value).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + " … (truncated)"


def clean_text_fields(row: dict) -> dict:
    """Normalize + truncate every string field of one store row/doc.

    ``content`` (document bodies) gets the larger cap, everything else
    the standard cap. Non-string values pass through unchanged.
    """
    out = {}
    for key, value in row.items():
        if isinstance(value, str):
            out[key] = truncate_text(value, _LONG_TEXT_FIELDS.get(key, MAX_TEXT_FIELD_CHARS))
        else:
            out[key] = value
    return out


# --- Layer 6: detection signals (warning only, never a block) ---
# Instruction-like phrases that belong in a review flag, not in a
# blocklist: the same words appear in legitimate maintenance text, so
# blocking on them would false-positive while attackers rephrase around
# them. Detection runs in validate_results (runtime/mission_graph.py).
INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "ignore-previous-instructions"),
    (r"disregard\s+(all\s+)?(previous|prior|above)", "disregard-previous"),
    (r"you\s+must\b", "you-must"),
    (r"you\s+are\s+now\b", "you-are-now"),
    (r"forget\s+(all\s+)?(previous|prior|above)", "forget-previous"),
    (r"bỏ\s+qua\s+(hướng\s+dẫn|chỉ\s+dẫn|lệnh)", "bo-qua-huong-dan"),
    (r"phớt\s+lờ\b", "phot-lo"),
    (r"quên\s+(hướng\s+dẫn|chỉ\s+dẫn)", "quen-huong-dan"),
]
_COMPILED_PATTERNS = [
    (re.compile(p, re.IGNORECASE), label) for p, label in INJECTION_PATTERNS
]


def detect_injection_signals(text: str) -> list[str]:
    """Return the labels of instruction-like patterns found in ``text``.

    Empty list = no signal. A signal is a review flag, never a reason to
    block or drop content.
    """
    if not text:
        return []
    found: list[str] = []
    for pattern, label in _COMPILED_PATTERNS:
        if pattern.search(text) and label not in found:
            found.append(label)
    return found
