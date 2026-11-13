"""Outcome graders for the hotel manual suite (Phase 1).

Covers the main features of ``scripts/manual/verifiers.py`` only:
semantic checks, state verifiers, scenario verdicts, trajectory
diagnostics — plus the two negative controls every grader must satisfy:
a do-nothing run fails every verdict, and a gold run passes all of them.
"""

import scripts.manual.verifiers as v

# --- Semantic checks -----------------------------------------------------------


def test_unavailability_passes_on_vietnamese_paraphrases():
    for answer in (
        "Phòng Suite đã hết, không có phòng Suite trống từ 24 đến 26/12.",
        "Xin lỗi, Suite không còn phòng nào trong khoảng thời gian này.",
        "Suite fully booked cho 24-26/12, không còn trống.",
        "Suite da het phong, khong con phong trong.",  # no diacritics
    ):
        assert v.check_unavailability(answer).passed, answer


def test_unavailability_passes_on_english_paraphrases():
    for answer in (
        "No rooms available for the Suite on those dates.",
        "The Suite is sold out for Dec 24-26.",
    ):
        assert v.check_unavailability(answer).passed, answer


def test_unavailability_fails_when_rooms_exist_or_answer_is_empty():
    for answer in (
        "",
        "Suite còn 2 phòng trống từ 24 đến 26/12, giá 3.500.000/đêm.",
        "Deluxe còn 5 phòng, Suite thì chưa kiểm tra.",
        "Hook 'tool_error_feedback': tool 'read_file' failed.",
    ):
        assert not v.check_unavailability(answer).passed, answer


def test_policy_facts_require_both_window_and_fee():
    assert v.check_policy_facts(
        "Hủy trước 3 ngày được miễn phí; hủy trong vòng 3 ngày thì phí 1 đêm."
    ).passed
    assert v.check_policy_facts(
        "Free cancellation up to three days before check-in; "
        "late cancellation costs one night."
    ).passed
    assert not v.check_policy_facts("Hủy trước 3 ngày được miễn phí.").passed
    assert not v.check_policy_facts("Hủy sát ngày thì phí 1 đêm.").passed
    assert not v.check_policy_facts("").passed


# --- State verifiers -------------------------------------------------------------


def _fake_store(rows):
    """get_booking callable over a dict of code -> booking row."""

    def get_booking(code):
        return rows.get(code, {"error": "not found"})

    return get_booking


def test_find_bookings_by_window_scans_only_new_ids():
    rows = {
        "BK-1002": {  # seeded sold-out booking: below the scan range
            "room_type": "suite",
            "check_in": "2026-12-24",
            "check_out": "2026-12-26",
        },
        "BK-1004": {  # a booking the agent created in this window
            "booking_id": "BK-1004",
            "room_type": "Suite",
            "check_in": "2026-12-24",
            "check_out": "2026-12-26",
        },
        "BK-1005": {  # another window: must not match
            "booking_id": "BK-1005",
            "room_type": "suite",
            "check_in": "2027-01-05",
            "check_out": "2027-01-07",
        },
    }
    found = v.find_bookings_by_window(
        _fake_store(rows), "suite", "2026-12-24", "2026-12-26"
    )
    assert [b["booking_id"] for b in found] == ["BK-1004"]


def test_find_bookings_by_window_stops_at_first_missing_id():
    rows = {
        "BK-1004": {
            "booking_id": "BK-1004",
            "room_type": "suite",
            "check_in": "2026-12-24",
            "check_out": "2026-12-26",
        },
        # BK-1005 missing on purpose; BK-1006 must never be reached.
        "BK-1006": {
            "booking_id": "BK-1006",
            "room_type": "suite",
            "check_in": "2026-12-24",
            "check_out": "2026-12-26",
        },
    }
    found = v.find_bookings_by_window(
        _fake_store(rows), "suite", "2026-12-24", "2026-12-26"
    )
    assert [b["booking_id"] for b in found] == ["BK-1004"]


def test_verify_no_new_booking():
    assert v.verify_no_new_booking([]).passed
    assert not v.verify_no_new_booking([{"booking_id": "BK-1004"}]).passed


def test_verify_task_row():
    good = {"status": "OPEN", "room": "201", "team": "technicians"}
    assert v.verify_task_row(good).passed
    assert not v.verify_task_row(None).passed
    assert not v.verify_task_row({**good, "status": "ASSIGNED"}).passed
    assert not v.verify_task_row({**good, "room": "202"}).passed
    assert not v.verify_task_row({**good, "team": "housekeeping"}).passed


# --- Trajectory diagnostics ------------------------------------------------------


def test_trajectory_findings():
    calls = [
        {"tool": "check_room_availability", "status": "success"},
        {"tool": "create_booking", "status": "error"},
    ]
    assert v.trajectory_findings(calls) == []
    assert v.trajectory_findings(
        calls, required=("list_staff",)
    ) == ["required tool 'list_staff' was not called successfully"]
    assert v.trajectory_findings(
        calls, forbidden=("create_booking",)
    ) == ["forbidden tool 'create_booking' was attempted"]
    ordered = [
        {"tool": "create_task", "status": "success"},
        {"tool": "list_staff", "status": "success"},
    ]
    findings = v.trajectory_findings(
        ordered, order=(("list_staff", "create_task"),)
    )
    assert len(findings) == 1 and "before" in findings[0]


# --- Negative controls: do-nothing fails, gold passes ------------------------------


def test_soldout_verdict_do_nothing_fails_and_gold_passes():
    # Do-nothing: empty answer + untouched store. The state half alone
    # would pass (nothing was created) — the semantic half must fail it.
    assert not v.soldout_verdict("", []).passed
    # Gold: right state + right meaning, in different words.
    assert v.soldout_verdict("Không có phòng Suite trống.", []).passed
    # Right words but the agent secretly booked anyway.
    assert not v.soldout_verdict(
        "Không có phòng Suite trống.", [{"booking_id": "BK-1004"}]
    ).passed
    # Right state but the answer never says sold-out.
    assert not v.soldout_verdict("Để tôi kiểm tra lại giúp bạn.", []).passed


def test_policy_verdict_do_nothing_fails_and_gold_passes():
    assert not v.policy_verdict("", read_done=False).passed
    assert not v.policy_verdict(
        "Hủy trước 3 ngày miễn phí, sát ngày phí 1 đêm.", read_done=False
    ).passed  # facts right, document never read
    assert not v.policy_verdict(
        "Chính sách hủy phòng như sau.", read_done=True
    ).passed  # read done, facts missing
    assert v.policy_verdict(
        "Hủy trước 3 ngày miễn phí, sát ngày phí 1 đêm.", read_done=True
    ).passed
