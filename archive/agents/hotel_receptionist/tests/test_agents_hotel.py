"""HotelReceptionistAgent tests — Phase 2 (guest scope) + Phase 3 (write pipeline).

The scope behaviour itself is covered end-to-end by the layer tests
(tests/test_layer_enforcement.py, fake guest tools + the agent in the
context). This file covers what is specific to the agent class: it
resolves, it does not scope when no agent is present, the confirm
tool follows the pipeline id, and the booking pipeline keeps its
hotel flavour (requested room-type row + guest-facing answer).
"""

import json
from unittest.mock import patch

from langchain_core.tools import tool as langchain_tool

import runtime.tools  # noqa: F401 — importing registers the built-in tools
from runtime.agents.base import BaseAgent
from runtime.agents.hotel import HotelReceptionistAgent
from runtime.agents.registry import resolve_agent
from runtime.session_write_pending import (
    merge_tool_result,
    reset_pending_store_for_tests,
    save_pending,
)
from runtime.tool_preconditions import check_write_preconditions
from runtime.tools.registry import registry
from runtime.write_idempotency import reset_idempotency_cache_for_tests
from runtime.write_pipeline_runner import WritePipeline


def _hotel_agent():
    return HotelReceptionistAgent(
        "hotel_receptionist", {"mission_id": "hotel_reception"}
    )


def test_resolve_agent_returns_hotel_agent():
    agent = resolve_agent("hotel_receptionist", {"mission_id": "hotel_reception"})
    assert isinstance(agent, HotelReceptionistAgent)
    assert agent.name == "hotel_receptionist"


def test_base_agent_scope_tool_is_a_noop():
    agent = BaseAgent("fault_investigator", {"mission_id": "fault_analysis"})

    @langchain_tool
    def find_guest(query: str) -> dict:
        """Find a guest."""
        return {"guests": [], "count": 0}

    assert agent.scope_tool(find_guest, "sess-x") is find_guest


def test_registry_does_not_scope_without_an_agent():
    """Regression (Phase 2): the registry only wraps when the context
    carries an agent — a caller without one gets the raw tools."""

    @langchain_tool
    def find_guest(query: str) -> dict:
        """Find a guest."""
        return {"guests": [{"guest_id": "guest-001"}], "count": 1}

    registry.register(find_guest, capability="hotel_reception", read_only=True)
    entry = registry.get("find_guest")
    context = {
        "mission_id": "hotel_reception",
        "tenant": "default",
        "execution_id": "exec_test",
        "capabilities": ["hotel_reception"],
        "session_id": "sess-no-agent",
    }
    tools = {t.name: t for t in registry.get_langchain_tools(context)}
    assert tools["find_guest"].func is entry.tool.func


def test_confirm_tool_name_follows_pipeline_id():
    assert _hotel_agent().confirm_tool_name("booking_create") == "confirm_booking_create"
    assert BaseAgent("", {}).confirm_tool_name("booking_create") == "confirm_booking_create"


def test_confirm_tool_is_session_scoped_and_registers_stub():
    reset_pending_store_for_tests()
    merge_tool_result(
        "sess-hotel-1",
        "booking_create",
        "get_room_rates",
        {"rates": [{"room_type": "deluxe"}]},
    )
    agent = _hotel_agent()
    tool = agent.confirm_tool("sess-hotel-1", "booking_create")
    assert tool.name == "confirm_booking_create"
    assert "confirm_guest_booking" not in tool.name
    # The old hotel name is gone from the registry entirely.
    assert registry.get("confirm_guest_booking") is None
    # The stub is registered lazily under the new name, so the L2
    # call-time gate knows the tool.
    assert registry.get("confirm_booking_create") is not None
    payload = json.loads(tool.invoke({}))
    assert payload == {"confirmed": True, "pipeline_id": "booking_create"}


def _setup_booking_pipeline(session_id: str, requested_room_type: str):
    """Pending quote with two rate rows; the guest asked for one of them."""
    reset_pending_store_for_tests()
    reset_idempotency_cache_for_tests()
    merge_tool_result(
        session_id,
        "booking_create",
        "get_room_rates",
        {
            "check_in": "2025-01-01",
            "check_out": "2025-01-03",
            "rates": [
                {"room_type": "standard", "total_price": 2_000_000},
                {"room_type": "deluxe", "total_price": 4_000_000},
            ],
        },
    )
    merge_tool_result(
        session_id,
        "booking_create",
        "check_room_availability",
        {"guests": 2, "requested_room_type": requested_room_type},
    )


def _booking_pipe():
    return WritePipeline(
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


class _StubCreateBooking:
    name = "create_booking"
    seen_args: dict | None = None

    def invoke(self, args):
        _StubCreateBooking.seen_args = dict(args)
        return {"booking_id": "BK-1", **args}


def test_write_pipeline_picks_requested_room_type_row():
    """The hotel resolver picks the rate row the guest asked for —
    not rates[0] (the old special-case behaviour, now in the agent)."""
    from runtime.session_write_pending import get_pending

    session_id = "sess-hotel-2"
    _setup_booking_pipeline(session_id, "deluxe")
    save_pending(
        session_id,
        "booking_create",
        status="confirmed",
        payload=get_pending(session_id, "booking_create")["payload"],
    )
    tool = _StubCreateBooking()
    with patch(
        "runtime.write_pipeline_runner.get_bound_guest_id", return_value="guest-001"
    ), patch(
        "runtime.tool_preconditions.get_bound_guest_id", return_value="guest-001"
    ):
        answer, errs, _ = _hotel_agent().write_pipeline(
            _booking_pipe(),
            session_id=session_id,
            execution_id="exec-1",
            tools_by_name={"create_booking": tool},
            session_turns=1,
        )
    assert not errs
    assert answer == "Your booking has been created (reference BK-1)."
    assert _StubCreateBooking.seen_args["room_type"] == "deluxe"


def test_write_pipeline_names_confirm_tool_when_unconfirmed():
    session_id = "sess-hotel-3"
    _setup_booking_pipeline(session_id, "standard")
    answer, errs, _ = _hotel_agent().write_pipeline(
        _booking_pipe(),
        session_id=session_id,
        execution_id="exec-1",
        tools_by_name={"create_booking": _StubCreateBooking()},
        session_turns=1,
    )
    assert not answer
    assert errs and "confirm_booking_create" in errs[0]
    assert "confirm_guest_booking" not in errs[0]


def test_booking_exists_for_session_via_agent(monkeypatch):
    """The precondition check runs through the agent's policy:
    another guest's booking is refused."""
    from runtime.tools.registry import ToolEntry

    class _FakeGetBooking:
        name = "get_booking"

        def invoke(self, args):
            return {"booking_id": "BK-1002", "guest_id": "guest-002"}

    fake_entry = ToolEntry(
        name="get_booking", description="", capability="hotel", read_only=True
    )
    fake_entry.tool = _FakeGetBooking()
    record_entry = ToolEntry(
        name="record_payment", description="", capability="hotel", read_only=False
    )

    def _fake_get(name):
        if name == "get_booking":
            return fake_entry
        if name == "record_payment":
            return record_entry
        return None

    monkeypatch.setattr(registry, "get", _fake_get)
    monkeypatch.setattr(
        "runtime.scope.get_bound_guest_id", lambda sid: "guest-001"
    )
    msg = check_write_preconditions(
        "record_payment",
        {"booking_id": "BK-1002", "amount": 1, "method": "cash"},
        session_id="sess-x",
        agent=_hotel_agent(),
    )
    assert msg and "refused" in msg and "another guest" in msg


def test_booking_exists_for_session_allows_own_booking(monkeypatch):
    """Same path, own booking: no refusal from the ownership check."""
    from runtime.tools.registry import ToolEntry

    class _FakeGetBooking:
        name = "get_booking"

        def invoke(self, args):
            return {"booking_id": "BK-1001", "guest_id": "guest-001"}

    fake_entry = ToolEntry(
        name="get_booking", description="", capability="hotel", read_only=True
    )
    fake_entry.tool = _FakeGetBooking()
    record_entry = ToolEntry(
        name="record_payment", description="", capability="hotel", read_only=False
    )

    def _fake_get(name):
        if name == "get_booking":
            return fake_entry
        if name == "record_payment":
            return record_entry
        return None

    monkeypatch.setattr(registry, "get", _fake_get)
    monkeypatch.setattr(
        "runtime.scope.get_bound_guest_id", lambda sid: "guest-001"
    )
    # record_payment's schema is empty here, so the only gate that can
    # fire is the booking-ownership check — it must not fire.
    msg = check_write_preconditions(
        "record_payment",
        {"booking_id": "BK-1001", "amount": 1, "method": "cash"},
        session_id="sess-x",
        agent=_hotel_agent(),
    )
    assert msg is None


# --- Phase 4: guest channel + grounding config -----------------------


def _hotel_mission_with_guest_flag():
    # Real shape: the flag lives in the parsed harness
    # (execution.harness.guest_output), as load_missions stores it.
    return {
        "mission_id": "hotel_reception",
        "execution": {
            "harness": {
                "guest_output": {
                    "register_reply_tools": True,
                    "capability": "hotel_reception",
                }
            }
        },
    }


def test_hotel_agent_gets_reply_tool_when_flag_set():
    agent = HotelReceptionistAgent(
        "hotel_receptionist", _hotel_mission_with_guest_flag()
    )
    tool = agent.guest_reply_tool()
    assert tool is not None
    assert tool.name == "reply_to_guest"


def test_base_agent_has_no_reply_tool_without_flag():
    assert (
        BaseAgent(
            "agent_analytis",
            {"mission_id": "agent_analytis_mission", "execution": {}},
        ).guest_reply_tool()
        is None
    )
    # Hotel agent without the flag: same — the flag, not the class,
    # decides.
    assert (
        HotelReceptionistAgent(
            "hotel_receptionist", {"mission_id": "hotel_reception"}
        ).guest_reply_tool()
        is None
    )


def test_hotel_grounding_config_declares_hotel_tools():
    cfg = HotelReceptionistAgent.grounding_config
    assert cfg.quote_tools == {"check_room_availability", "get_room_rates"}
    assert cfg.availability_tool == "check_room_availability"
    assert cfg.rates_tool == "get_room_rates"
    assert cfg.total_price_field == "total_price"
    assert cfg.per_night_field == "price_per_night"
    assert cfg.policy_search_tool == "search_policy_documents"
    assert cfg.policy_read_tool == "read_policy_document"


def test_base_agent_grounding_config_is_empty():
    assert BaseAgent("agent_analytis", {}).grounding_config.quote_tools == set()
    assert (
        BaseAgent("agent_analytis", {}).grounding_config.policy_tools == set()
    )
