"""Guest output guard — tool channel + echo/scope checks."""

from langchain_core.messages import AIMessage

from runtime.guest_output import (
    GUEST_REPLY_TOOL_NAME,
    guard_guest_reply,
    reply_from_tool_calls,
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