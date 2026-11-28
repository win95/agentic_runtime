"""Lark Calendar tools (Layer 5) — read-only free/busy for agent_lark_me.

Capability: ``lark_calendar``. Only ``lark_calendar_freebusy`` is
exposed; it queries the Lark Calendar freebusy API via
``runtime/integrations/lark_client.get_freebusy``. The model never
computes calendar windows — ``range`` / optional ``start``+``end`` are
interpreted deterministically in ``timezone`` (default Asia/Saigon).

The queried user is **not** a tool parameter: ``LarkAgent.scope_tool``
injects the run's sender ``open_id`` (primary calendar of the user
chatting with the bot). Free/busy returns busy intervals only — no
event titles or descriptions.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from runtime.integrations import lark_client
from runtime.tools.registry import registry

_ALLOWED_RANGES = frozenset(
    {"today", "tomorrow", "this_week", "next_7_days", "custom"}
)
_MAX_WINDOW = timedelta(days=90)


def _err(name: str, result: dict) -> dict:
    out = dict(result)
    out.setdefault("tool", name)
    out.setdefault("stub", False)
    out.setdefault("provenance", f"lark:{name.removeprefix('lark_')}")
    return out


def _load_tz(name: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 — bad tz name is user/model input
        return None


def _local_midnight(day: date, tz: ZoneInfo) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=tz)


def _to_rfc3339(dt: datetime) -> str:
    return dt.isoformat()


def _format_local(iso: str | None, tz: ZoneInfo) -> str | None:
    if not iso:
        return None
    text = iso.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return iso
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(tz).isoformat()


def compute_freebusy_window(
    *,
    range_name: str,
    start: str | None,
    end: str | None,
    timezone: str = "Asia/Saigon",
    now: datetime | None = None,
) -> dict:
    """Resolve ``range`` into RFC3339 ``time_min`` / ``time_max`` in ``timezone``.

    Returns ``{"time_min", "time_max", "window": {...}}`` or ``{"error": ...}``.
    """
    if range_name not in _ALLOWED_RANGES:
        return {"error": f"unknown range '{range_name}'"}
    tz = _load_tz(timezone)
    if tz is None:
        return {"error": f"unknown timezone '{timezone}'"}
    if now is not None and now.tzinfo is not None:
        anchor = now.astimezone(tz)
    elif now is not None:
        anchor = now.replace(tzinfo=tz)
    else:
        anchor = datetime.now(tz)

    today = anchor.date()

    if range_name == "custom":
        if not start or not end:
            return {"error": "custom range requires start and end (RFC3339)"}
        time_min = start.strip()
        time_max = end.strip()
        win_start = _parse_rfc3339(time_min)
        win_end = _parse_rfc3339(time_max)
        if win_start is None or win_end is None:
            return {"error": "start and end must be valid RFC3339 datetimes"}
        if win_end <= win_start:
            return {"error": "end must be after start"}
        if win_end - win_start > _MAX_WINDOW:
            return {"error": "custom range must not exceed 90 days"}
        label = f"custom ({time_min} → {time_max})"
    elif range_name == "today":
        lo = _local_midnight(today, tz)
        hi = lo + timedelta(days=1)
        time_min, time_max = _to_rfc3339(lo), _to_rfc3339(hi)
        label = "today"
    elif range_name == "tomorrow":
        d = today + timedelta(days=1)
        lo = _local_midnight(d, tz)
        hi = lo + timedelta(days=1)
        time_min, time_max = _to_rfc3339(lo), _to_rfc3339(hi)
        label = "tomorrow"
    elif range_name == "this_week":
        monday = today - timedelta(days=today.weekday())
        lo = _local_midnight(monday, tz)
        hi = lo + timedelta(days=7)
        time_min, time_max = _to_rfc3339(lo), _to_rfc3339(hi)
        label = "this_week"
    else:  # next_7_days
        lo = _local_midnight(today, tz)
        hi = lo + timedelta(days=7)
        time_min, time_max = _to_rfc3339(lo), _to_rfc3339(hi)
        label = "next_7_days"

    return {
        "time_min": time_min,
        "time_max": time_max,
        "window": {
            "range": range_name,
            "timezone": timezone,
            "label": label,
            "time_min_local": _format_local(time_min, tz),
            "time_max_local": _format_local(time_max, tz),
        },
    }


def _parse_rfc3339(value: str) -> datetime | None:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def run_lark_calendar_freebusy(
    *,
    user_open_id: str,
    range_name: str = "today",
    start: str | None = None,
    end: str | None = None,
    timezone: str = "Asia/Saigon",
    now: datetime | None = None,
) -> dict:
    """Core freebusy implementation (``user_open_id`` supplied by scope_tool)."""
    if not user_open_id or not str(user_open_id).strip():
        return {
            "error": "lark_calendar_freebusy: no sender calendar — refused",
            "tool": "lark_calendar_freebusy",
        }
    window = compute_freebusy_window(
        range_name=range_name,
        start=start,
        end=end,
        timezone=timezone,
        now=now,
    )
    if "error" in window:
        return _err("lark_calendar_freebusy", window)
    result = lark_client.get_freebusy(
        user_open_id=str(user_open_id).strip(),
        time_min=window["time_min"],
        time_max=window["time_max"],
    )
    if "error" in result:
        return _err("lark_calendar_freebusy", result)
    tz = _load_tz(timezone) or ZoneInfo("Asia/Saigon")
    busy = []
    for item in result.get("busy_intervals") or []:
        busy.append(
            {
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "start_time_local": _format_local(item.get("start_time"), tz),
                "end_time_local": _format_local(item.get("end_time"), tz),
            }
        )
    return {
        "busy_intervals": busy,
        "count": len(busy),
        "window": window["window"],
        "time_min": window["time_min"],
        "time_max": window["time_max"],
        "timezone": timezone,
        "stub": False,
        "provenance": "lark:calendar_freebusy",
    }


@tool
def lark_calendar_freebusy(
    range: str = "today",
    start: str | None = None,
    end: str | None = None,
    timezone: str = "Asia/Saigon",
) -> dict:
    """Read the sender's Lark calendar busy/free intervals for a time window.

    Read-only. ``range`` is one of today|tomorrow|this_week|next_7_days|custom
    (custom requires ``start`` and ``end`` as RFC3339). ``timezone`` sets how
    the window and busy times are formatted (default Asia/Saigon). Returns
    busy intervals only — no event titles or details — plus the resolved
    window. Only the chatting user's own calendar is queried (scope_tool).
    """
    return run_lark_calendar_freebusy(
        user_open_id="",
        range_name=range,
        start=start,
        end=end,
        timezone=timezone,
    )


registry.register(
    lark_calendar_freebusy,
    capability="lark_calendar",
    permission="lark:read",
    timeout_seconds=30,
    read_only=True,
)