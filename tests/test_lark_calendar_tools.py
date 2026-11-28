"""Lark calendar tool — window computation and registration."""

from datetime import datetime
from zoneinfo import ZoneInfo

import runtime.tools.lark_calendar_tools as calendar_tools
from runtime.integrations import lark_client
from runtime.tools.registry import registry


def test_calendar_tool_registered():
    entry = registry.get("lark_calendar_freebusy")
    assert entry is not None
    assert entry.capability == "lark_calendar"
    assert entry.read_only is True


def test_compute_today_window_saigon():
    tz = ZoneInfo("Asia/Saigon")
    anchor = datetime(2026, 11, 28, 15, 30, tzinfo=tz)
    out = calendar_tools.compute_freebusy_window(
        range_name="today", start=None, end=None, timezone="Asia/Saigon", now=anchor
    )
    assert "error" not in out
    assert out["window"]["label"] == "today"
    assert "+07:00" in out["time_min"] or "+07:00" in out["window"]["time_min_local"]


def test_compute_this_week_starts_monday():
    tz = ZoneInfo("Asia/Saigon")
    # 2026-11-28 is Saturday
    anchor = datetime(2026, 11, 28, 9, 0, tzinfo=tz)
    out = calendar_tools.compute_freebusy_window(
        range_name="this_week",
        start=None,
        end=None,
        timezone="Asia/Saigon",
        now=anchor,
    )
    assert out["window"]["time_min_local"].startswith("2026-11-23")


def test_run_freebusy_formats_busy_local(monkeypatch):
    monkeypatch.setattr(
        lark_client,
        "get_freebusy",
        lambda **kw: {
            "busy_intervals": [
                {
                    "start_time": "2026-11-28T03:00:00+00:00",
                    "end_time": "2026-11-28T04:00:00+00:00",
                }
            ],
            "count": 1,
        },
    )
    tz = ZoneInfo("Asia/Saigon")
    out = calendar_tools.run_lark_calendar_freebusy(
        user_open_id="ou_1",
        range_name="today",
        timezone="Asia/Saigon",
        now=datetime(2026, 11, 28, 8, 0, tzinfo=tz),
    )
    assert out["count"] == 1
    local_start = out["busy_intervals"][0]["start_time_local"]
    assert local_start is not None and "+07:00" in local_start


def test_tool_without_user_errors():
    out = calendar_tools.lark_calendar_freebusy.invoke({"range": "today"})
    assert "error" in out