"""Lark client get_freebusy — mocked SDK, no network."""

from types import SimpleNamespace

from runtime.integrations import lark_client


class _Resp:
    def __init__(self, code=0, msg="", data=None):
        self.code = code
        self.msg = msg
        self.data = data


def test_get_freebusy_rejects_over_90_days():
    out = lark_client.get_freebusy(
        user_open_id="ou_1",
        time_min="2026-01-01T00:00:00+07:00",
        time_max="2026-05-01T00:00:00+07:00",
    )
    assert "error" in out
    assert "90 days" in out["error"]


def test_get_freebusy_success(monkeypatch):
    fb = SimpleNamespace(
        start_time="2026-11-28T10:00:00+07:00",
        end_time="2026-11-28T11:00:00+07:00",
    )
    data = SimpleNamespace(freebusy_list=[fb])
    captured = {}

    def fake_list(req):
        captured["req"] = req
        return _Resp(data=data)

    fake_client = SimpleNamespace(
        calendar=SimpleNamespace(
            v4=SimpleNamespace(freebusy=SimpleNamespace(list=fake_list))
        )
    )

    from lark_oapi.api.calendar.v4 import model as cal_model

    monkeypatch.setattr(
        lark_client, "_calendar_model", lambda: (fake_client, cal_model)
    )
    out = lark_client.get_freebusy(
        user_open_id="ou_x",
        time_min="2026-11-28T00:00:00+07:00",
        time_max="2026-11-29T00:00:00+07:00",
    )
    assert out["count"] == 1
    assert out["busy_intervals"][0]["start_time"].startswith("2026-11-28")
    assert captured["req"] is not None


def test_get_freebusy_missing_scope_99991672(monkeypatch):
    from lark_oapi.api.calendar.v4 import model as cal_model

    monkeypatch.setattr(
        lark_client,
        "_calendar_model",
        lambda: (
            SimpleNamespace(
                calendar=SimpleNamespace(
                    v4=SimpleNamespace(
                        freebusy=SimpleNamespace(
                            list=lambda req: _Resp(
                                code=99991672, msg="scope required"
                            )
                        )
                    )
                )
            ),
            cal_model,
        ),
    )
    out = lark_client.get_freebusy(
        user_open_id="ou_x",
        time_min="2026-11-28T00:00:00+07:00",
        time_max="2026-11-29T00:00:00+07:00",
    )
    assert "99991672" in out["error"]
    assert "calendar:calendar.free_busy:read" in out["error"]