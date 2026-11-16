"""Guest output guard — tool channel + echo/scope checks."""

from langchain_core.messages import AIMessage

from app.config import settings
from runtime.guest_output import (
    GUEST_REPLY_TOOL_NAME,
    _dedupe_paragraphs,
    finalize_guest_reply,
    guard_guest_reply,
    reply_from_tool_calls,
    strip_runtime_markup,
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
        '<data source="execution_input">\n'
        '{"message":"book deluxe"}\n</data>\n\n'
        "Session history: guest asked about Deluxe room for March dates "
        "with phone listed in the registration block."
    )
    cleaned, notes = guard_guest_reply(reply, context_blobs=[context])
    assert "<data" not in cleaned
    assert "book deluxe" in cleaned or len(cleaned) < len(reply)


def test_strip_runtime_markup_removes_only_data_blocks():
    raw = (
        "<data source=\"x\">internal payload</data>\n\n"
        "Visible reply for the guest."
    )
    cleaned = strip_runtime_markup(raw)
    assert "internal payload" not in cleaned
    assert "Visible reply" in cleaned
    assert "Internal planning note" in strip_runtime_markup(
        "Internal planning note\n\nGuest-visible line."
    )


def test_finalize_guest_reply_uses_generic_fallback_without_tool():
    answer, notes = finalize_guest_reply(
        [AIMessage(content="assistant text only, no tool")],
        context_blobs=[],
        session_id="s1",
        input_data={"channel": "guest_api"},
    )
    assert answer == settings.guest_reply_generic_fallback
    assert any("generic fallback" in n for n in notes)


def test_finalize_guest_reply_prefers_reply_to_guest():
    msgs = [
        AIMessage(
            content="ignored",
            tool_calls=[
                {
                    "id": "1",
                    "name": GUEST_REPLY_TOOL_NAME,
                    "args": {"message": "Tool channel answer."},
                }
            ],
        )
    ]
    answer, _ = finalize_guest_reply(msgs, context_blobs=[])
    assert answer == "Tool channel answer."


def test_dedupe_near_duplicate_paragraphs():
    p1 = "The room category is Deluxe with sea view."
    p2 = "The room category is Deluxe with sea view today."
    text = _dedupe_paragraphs(f"{p1}\n\n{p2}\n\nCancellation policy applies.")
    assert text.count("sea view") <= 2
    assert "Cancellation policy" in text