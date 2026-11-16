"""Guest output guard — tool channel + echo/scope checks."""

from langchain_core.messages import AIMessage

from runtime.guest_output import (
    GUEST_REPLY_TOOL_NAME,
    _dedupe_paragraphs,
    guard_guest_reply,
    reply_from_tool_calls,
    strip_internal_prose,
)


def test_reply_from_tool_calls_prefers_tool():
    msg = AIMessage(
        content="prose should not win",
        tool_calls=[
            {
                "id": "1",
                "name": GUEST_REPLY_TOOL_NAME,
                "args": {"message": "Xin chào, em có thể giúp gì?"},
            }
        ],
    )
    assert reply_from_tool_calls([msg]) == "Xin chào, em có thể giúp gì?"


def test_guard_strips_data_blocks_and_overlap():
    context = (
        "Session history: guest asked about Deluxe room for March dates "
        "with phone listed in the registration block."
    )
    reply = (
        "<data source=\"execution_input\">\n"
        '{"message":"book deluxe"}\n</data>\n\n'
        "Session history: guest asked about Deluxe room for March dates "
        "with phone listed in the registration block."
    )
    cleaned, notes = guard_guest_reply(reply, context_blobs=[context])
    assert "<data" not in cleaned
    assert "book deluxe" in cleaned or len(cleaned) < len(reply)


def test_strip_internal_prose_extracts_quoted_reply():
    raw = (
        "Booking summary:\n- Room Type: Deluxe\n\n"
        'Reply to guest: "Phòng Deluxe còn trống."'
    )
    assert "Deluxe" in strip_internal_prose(raw)
    assert "Booking summary" not in strip_internal_prose(raw)


def test_dedupe_near_duplicate_paragraphs():
    p1 = "Flow 'policy-lookup': answered policy question for the guest."
    p2 = "Flow 'policy-lookup': answered policy question for the guest today."
    text = _dedupe_paragraphs(f"{p1}\n\n{p2}\n\nKhách nhận chính sách hủy.")
    assert text.count("Flow 'policy-lookup'") <= 1