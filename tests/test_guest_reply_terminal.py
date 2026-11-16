"""Guest reply termination and extraction (flow agent)."""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from runtime.guest_output import (
    GUEST_REPLY_TOOL_NAME,
    extract_flow_guest_reply,
    finalize_guest_reply,
    reply_from_tool_calls,
)
from runtime.subgraph.common import GuestReplyCompleteError, ToolCallCap


def _req(name: str, args: dict, call_id: str = "c1"):
    from types import SimpleNamespace

    return SimpleNamespace(
        tool_call={"name": name, "args": args, "id": call_id},
    )


def test_cap_terminates_agent_after_successful_reply_to_guest():
    cap = ToolCallCap(max_calls=10)

    def handler(_request):
        return ToolMessage(
            content='{"recorded": true}',
            tool_call_id="c3",
        )

    cap.wrap_tool_call(
        _req("check_room_availability", {"check_in": "2026-01-01"}),
        handler,
    )
    with pytest.raises(GuestReplyCompleteError) as exc:
        cap.wrap_tool_call(
            _req(
                GUEST_REPLY_TOOL_NAME,
                {"message": "Phòng Deluxe còn trống ngày 20–22/12."},
                "c-guest",
            ),
            handler,
        )
    assert cap.guest_reply_message == "Phòng Deluxe còn trống ngày 20–22/12."
    assert exc.value.guest_message == cap.guest_reply_message


def test_reply_from_tool_calls_with_tool_message_sequence():
    """Realistic turn: availability + rates + reply_to_guest AIMessage + ToolMessage."""
    guest_text = "Deluxe 20–22/12: còn phòng, 2.500.000 VND/đêm."
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "t-avail",
                    "name": "check_room_availability",
                    "args": {
                        "check_in": "2026-12-20",
                        "check_out": "2026-12-22",
                        "room_type": "deluxe",
                    },
                }
            ],
        ),
        ToolMessage(
            content=json.dumps({"available": True}),
            tool_call_id="t-avail",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "t-rates",
                    "name": "get_room_rates",
                    "args": {
                        "check_in": "2026-12-20",
                        "check_out": "2026-12-22",
                    },
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "rates": [
                        {
                            "room_type": "deluxe",
                            "total_price": 5000000,
                            "currency": "VND",
                        }
                    ]
                }
            ),
            tool_call_id="t-rates",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "t-reply",
                    "name": GUEST_REPLY_TOOL_NAME,
                    "args": {"message": guest_text},
                }
            ],
        ),
        ToolMessage(
            content='{"recorded": true}',
            tool_call_id="t-reply",
        ),
    ]
    assert reply_from_tool_calls(messages) == guest_text
    answer, notes = finalize_guest_reply(
        messages,
        context_blobs=[],
        input_data={"message": "giá phòng deluxe", "channel": "guest_api"},
    )
    assert answer == guest_text
    assert "generic fallback" not in " ".join(notes).lower()
    answer2, _ = extract_flow_guest_reply(
        messages,
        context_blobs=[],
        input_data={"message": "giá phòng deluxe"},
    )
    assert answer2 == guest_text