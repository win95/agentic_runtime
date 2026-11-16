"""Phase 3 — write pipelines, preconditions, idempotency, grounding."""

from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from runtime.reply_grounding import apply_reply_grounding
from runtime.session_write_pending import get_pending, merge_tool_result, reset_pending_store_for_tests
from runtime.tool_preconditions import check_write_preconditions
from runtime.tools.registry import ToolEntry
from runtime.write_idempotency import invoke_idempotent_write, reset_idempotency_cache_for_tests
from runtime.write_pipeline_runner import (
    WritePipeline,
    build_write_args,
    parse_pipelines,
    try_automated_write,
)


def test_parse_pipelines_from_map_shape():
    raw = [
        {
            "id": "booking_create",
            "quote_tools": ["get_room_rates"],
            "write_tool": "create_booking",
            "write_node": "create-booking",
            "write_arg_fields": ["guest_id", "room_type"],
            "arg_paths": {"room_type": ["get_room_rates", "rates", 0, "room_type"]},
        }
    ]
    pipes = parse_pipelines(raw, "reception")
    assert len(pipes) == 1
    assert pipes[0].write_tool == "create_booking"


def test_pending_and_automated_write():
    reset_pending_store_for_tests()
    reset_idempotency_cache_for_tests()
    merge_tool_result(
        "sess-1",
        "booking_create",
        "get_room_rates",
        {
            "check_in": "2025-01-01",
            "check_out": "2025-01-03",
            "rates": [{"room_type": "deluxe", "total_price": 4_000_000}],
        },
    )
    merge_tool_result(
        "sess-1",
        "booking_create",
        "check_room_availability",
        {"guests": 2, "available": True},
    )
    assert get_pending("sess-1", "booking_create") is not None

    class StubTool:
        name = "create_booking"
        calls = 0

        def invoke(self, args):
            self.calls += 1
            return {"booking_id": "BK-1", **args}

    tool = StubTool()
    pipe = WritePipeline(
        id="booking_create",
        write_tool="create_booking",
        write_arg_fields=["guest_id", "room_type", "check_in", "check_out", "guests"],
        arg_paths={
            "room_type": ["get_room_rates", "rates", 0, "room_type"],
            "check_in": ["get_room_rates", "check_in"],
            "check_out": ["get_room_rates", "check_out"],
            "guests": ["check_room_availability", "guests"],
        },
    )
    with patch("runtime.write_pipeline_runner.get_bound_guest_id", return_value="guest-001"):
        args = build_write_args(
            pipe, get_pending("sess-1", "booking_create")["payload"], session_id="sess-1"
        )
        assert args["room_type"] == "deluxe"
        assert args["guest_id"] == "guest-001"

        ans, errs, _ = try_automated_write(
            pipe,
            session_id="sess-1",
            execution_id="exec-1",
            tools_by_name={"create_booking": tool},
            session_turns=1,
        )
        assert not errs
        assert "BK-1" in ans
        assert tool.calls == 1
        ans2, errs2, _ = try_automated_write(
            pipe,
            session_id="sess-1",
            execution_id="exec-2",
            tools_by_name={"create_booking": tool},
            session_turns=1,
        )
        assert not errs2
        assert tool.calls == 1


def test_idempotent_replay():
    reset_idempotency_cache_for_tests()

    class Stub:
        name = "create_booking"
        calls = 0

        def invoke(self, args):
            self.calls += 1
            return {"booking_id": "BK-9", "ok": True}

    t = Stub()
    invoke_idempotent_write(t, {"guest_id": "g1"}, session_id="s1", execution_id="e1")
    invoke_idempotent_write(t, {"guest_id": "g1"}, session_id="s1", execution_id="e2")
    assert t.calls == 1


def test_record_payment_precondition_missing_booking():
    entry = ToolEntry(name="record_payment", description="", capability="hotel", read_only=False)
    with patch("runtime.tool_preconditions.registry.get", return_value=entry), patch(
        "runtime.tool_preconditions._booking_exists_for_session",
        return_value="booking does not exist",
    ):
        msg = check_write_preconditions(
            "record_payment",
            {"booking_id": "BK-404", "amount": 1, "method": "cash"},
            session_id="sess-x",
        )
    assert msg and "refused" in msg


def test_reply_grounding_replaces_invented_usd():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"id": "1", "name": "get_room_rates", "args": {}}],
        ),
        ToolMessage(
            content='{"rates":[{"total_price":2000000,"currency":"VND","room_type":"deluxe"}]}',
            tool_call_id="1",
        ),
    ]
    reply, notes = apply_reply_grounding(
        "The total is 1200 USD for deluxe.",
        messages,
    )
    assert "1200" not in reply or "VND" in reply
    assert notes