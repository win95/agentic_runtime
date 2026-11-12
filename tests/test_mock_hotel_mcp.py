"""Mock hotel MCP server tests — tool logic, contract, registry wiring,
and one end-to-end hotel run through the real Deep Agent.

The mock server is the stand-in for the real PMS / staff / docs MCP
servers (not connected yet). The wiring + end-to-end tests prove the
swap-in path: mock server -> MCP client -> registry (capability +
allowed_tools gates) -> Deep Agent -> tool execution over HTTP.
"""

import asyncio
import json
import pathlib
import socket
import threading
import time

import pytest
import uvicorn
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import scripts.mock_hotel_mcp_server as mock_server
from app.config import settings
from app.services import mission_service
from runtime import deep_agent as deep_agent_module
from runtime.mission_graph import run_mission
from runtime.tools.registry import registry
from seeds import load_missions

BASE = pathlib.Path(__file__).resolve().parents[1] / "seeds"

# The contract source: the agent YAML's allowed_tools (docs/hotel-reception.md).
HOTEL_TOOLS = set(
    load_missions.load_agent_file(
        BASE / "agents" / "hotel_receptionist.yaml"
    )["allowed_tools"]
)

HOTEL_CONTEXT = {
    "mission_id": "hotel_reception",
    "tenant": "default",
    "execution_id": "exec_hotel_test",
    "capabilities": ["hotel_reception"],
    "allowed_tools": sorted(HOTEL_TOOLS),
}

FAULT_CONTEXT = {
    "mission_id": "fault_analysis",
    "tenant": "default",
    "execution_id": "exec_test",
    "capabilities": ["fault_history", "maintenance_analysis", "knowledge_retrieval"],
}


@pytest.fixture(autouse=True)
def _reset_store():
    mock_server.reset_store()
    yield
    mock_server.reset_store()


@pytest.fixture(autouse=True)
def _clean_proxy_env(monkeypatch):
    """The VM's default no_proxy carries bracketed IPv6 entries
    ([::1], ...) that break httpx (the MCP client's HTTP layer) with
    InvalidURL. The mock server is localhost: override with a clean
    list for every test in this module."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture
def _clean_registry():
    """load_mcp_tools() registers into the global registry; restore it
    so tests that assert the exact built-in set keep passing."""
    snapshot = dict(registry._tools)
    yield
    registry._tools.clear()
    registry._tools.update(snapshot)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def mock_server_url():
    """Run the mock MCP server in a thread; yield its streamable-HTTP URL."""
    port = _free_port()
    # Fresh FastMCP instance per test: the SDK's session manager can
    # only run once per instance.
    config = uvicorn.Config(
        mock_server.create_server().streamable_http_app(),
        host="127.0.0.1", port=port, log_level="warning",
        timeout_graceful_shutdown=0,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError("mock MCP server did not start")
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=10)


def _patch_mcp_settings(monkeypatch, url: str) -> None:
    monkeypatch.setattr(settings, "mcp_server_url", url)
    monkeypatch.setattr(
        settings, "mcp_tool_capabilities",
        json.dumps({name: "hotel_reception" for name in sorted(HOTEL_TOOLS)}),
    )


# --- The mock exposes exactly the contract ---
def test_mock_exposes_contract_tools():
    tools = asyncio.run(mock_server.mcp.list_tools())
    assert {t.name for t in tools} == HOTEL_TOOLS


# --- Tool logic (direct calls, no HTTP) ---
def test_availability_sold_out_and_free():
    sold_out = mock_server.check_room_availability(
        "2026-12-24", "2026-12-26", room_type="suite")
    assert sold_out["available"] is False
    assert sold_out["room_types"][0]["available_rooms"] == 0
    free = mock_server.check_room_availability(
        "2026-12-20", "2026-12-22", room_type="deluxe")
    assert free["available"] is True
    # 6 Deluxe rooms minus the one seeded BK-1001 booking.
    assert free["room_types"][0]["available_rooms"] == 5
    assert free["nights"] == 2
    all_types = mock_server.check_room_availability("2026-12-20", "2026-12-22")
    assert {item["room_type"] for item in all_types["room_types"]} == {
        "standard", "deluxe", "suite",
    }


def test_availability_invalid_input():
    assert "error" in mock_server.check_room_availability(
        "2026-12-22", "2026-12-20", room_type="deluxe")
    assert "error" in mock_server.check_room_availability(
        "20/12/2026", "2026-12-22")
    assert "error" in mock_server.check_room_availability(
        "2026-12-20", "2026-12-22", room_type="penthouse")


def test_rates_total():
    rates = mock_server.get_room_rates("2026-12-20", "2026-12-22",
                                       room_type="deluxe")
    assert rates["nights"] == 2
    assert rates["rates"][0]["total_price"] == 2_000_000 * 2
    assert rates["rates"][0]["currency"] == "VND"
    assert "error" in mock_server.get_room_rates(
        "2026-12-20", "2026-12-22", room_type="penthouse")


def test_guest_find_and_register():
    found = mock_server.find_guest("0901234567")
    assert found["count"] == 1
    assert found["guests"][0]["guest_id"] == "guest-001"
    assert mock_server.find_guest("không có ai")["count"] == 0
    new = mock_server.register_guest("Lê Văn Cường", "0909999999")
    assert new["guest_id"] == "guest-003"
    assert mock_server.find_guest("Cường")["count"] == 1
    assert "error" in mock_server.register_guest("", "0901111111")


def test_booking_lifecycle():
    created = mock_server.create_booking(
        "guest-001", "standard", "2026-12-20", "2026-12-22", guests=2)
    assert created["booking_id"] == "BK-1004"
    assert created["status"] == "CONFIRMED"
    assert created["total_price"] == 1_200_000 * 2
    got = mock_server.get_booking("BK-1004")
    assert got["guest_id"] == "guest-001"
    moved = mock_server.change_booking_dates("BK-1004", "2026-12-21", "2026-12-23")
    assert moved["check_in"] == "2026-12-21"
    assert moved["total_price"] == 1_200_000 * 2
    cancelled = mock_server.cancel_booking("BK-1004", reason="đổi kế hoạch")
    assert cancelled["status"] == "CANCELLED"
    # Check-in is far ahead: free cancellation.
    assert cancelled["fee_vnd"] == 0
    assert mock_server.get_booking("BK-1004")["status"] == "CANCELLED"
    assert "error" in mock_server.cancel_booking("BK-1004")
    assert "error" in mock_server.change_booking_dates(
        "BK-1004", "2026-12-25", "2026-12-27")
    assert "error" in mock_server.get_booking("BK-9999")


def test_booking_create_sold_out_errors():
    # Both suites are taken 2026-12-24 -> 2026-12-26 (seeded BK-1002/1003).
    assert "error" in mock_server.create_booking(
        "guest-001", "suite", "2026-12-24", "2026-12-26")
    assert "error" in mock_server.create_booking(
        "guest-999", "standard", "2026-12-20", "2026-12-22")


def test_task_flow():
    staff = mock_server.list_staff()
    assert staff["count"] == 6
    housekeeping = mock_server.list_staff(team="housekeeping")
    assert housekeeping["count"] == 3
    assert "error" in mock_server.list_staff(team="concierge")
    task = mock_server.create_task(
        "Điều hòa không mát", "Điều hòa phòng 201 thổi hơi nóng", room="201",
        team="technicians")
    assert task["task_id"] == "TK-2001"
    assert task["status"] == "OPEN"
    assigned = mock_server.assign_task("TK-2001", "NV-201")
    assert assigned["status"] == "ASSIGNED"
    assert assigned["assignee"] == "NV-201"
    assert mock_server.get_task("TK-2001")["assignee"] == "NV-201"
    assert "error" in mock_server.assign_task("TK-2001", "NV-999")
    assert "error" in mock_server.assign_task("TK-9999", "NV-201")
    assert "error" in mock_server.get_task("TK-9999")
    assert "error" in mock_server.create_task("", "no title")


def test_policy_search_and_read():
    found = mock_server.search_policy_documents("hủy")
    assert found["count"] == 1
    assert found["documents"][0]["doc_id"] == "POL-CANCEL"
    assert mock_server.search_policy_documents("không có gì")["count"] == 0
    doc = mock_server.read_policy_document("POL-SURCHARGE")
    assert "300.000" in doc["content"]
    assert "error" in mock_server.read_policy_document("POL-NOPE")


# --- Registry wiring over real HTTP ---
def test_load_mcp_tools_registers_hotel_tools(
    mock_server_url, monkeypatch, _clean_registry,
):
    _patch_mcp_settings(monkeypatch, mock_server_url)
    from runtime.tools.mcp_adapter import load_mcp_tools

    registered = load_mcp_tools()
    assert {t.name for t in registered} == HOTEL_TOOLS
    for name in HOTEL_TOOLS:
        entry = registry.get(name)
        assert entry is not None
        assert entry.capability == "hotel_reception"
        assert entry.permission == "mcp"
        assert entry.read_only is False
    # Capability gate: the hotel mission sees all 14; fault_analysis none.
    assert {t.name for t in registry.get_langchain_tools(HOTEL_CONTEXT)} == HOTEL_TOOLS
    assert not ({t.name for t in registry.get_langchain_tools(FAULT_CONTEXT)}
                & HOTEL_TOOLS)
    # Sync entry point: the runtime invokes the agent synchronously, so
    # a registered MCP tool must answer a plain .invoke() over HTTP.
    result = registry.get("check_room_availability").tool.invoke({
        "check_in": "2026-12-20", "check_out": "2026-12-22",
        "room_type": "deluxe",
    })
    assert "available_rooms" in json.dumps(result)


# --- End-to-end: real Deep Agent + fake model + mock over HTTP ---
class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_hotel_run_end_to_end(db_session, mock_server_url, monkeypatch,
                              _clean_registry):
    load_missions.main()
    mission = mission_service.get_mission(db_session, "hotel_reception")
    mission_dict = mission_service.mission_to_dict(mission)
    _patch_mcp_settings(monkeypatch, mock_server_url)
    # The run gate (require_llm_config) needs base + key present; the
    # scripted fake model below is what actually answers, so dummy
    # values are never contacted. Patch build_chat_model (not only
    # get_chat_model): a versioned run builds the model from the pinned
    # version via build_chat_model(model_config).
    monkeypatch.setattr(settings, "llm_api_base", "http://127.0.0.1:1")
    monkeypatch.setattr(settings, "llm_api_key", "test-key")

    model = _ScriptedModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "check_room_availability",
            "args": {"check_in": "2026-12-20", "check_out": "2026-12-22",
                     "room_type": "deluxe"},
            "id": "call-1", "type": "tool_call",
        }]),
        AIMessage(content="Phòng Deluxe còn trống, giá 2.000.000 VND/đêm."),
        AIMessage(content="Phòng Deluxe còn trống, giá 2.000.000 VND/đêm."),
    ]))
    monkeypatch.setattr(deep_agent_module, "get_chat_model", lambda: model)
    monkeypatch.setattr(deep_agent_module, "build_chat_model",
                        lambda cfg=None: model)

    final = run_mission(
        mission_dict,
        {"message": "Tôi muốn đặt phòng Deluxe từ 20 đến 22/12 cho 2 người"},
        "exec-hotel-e2e",
    )
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert final["result"]["actions_taken"] == ["check_room_availability"]
    assert "Deluxe" in final["result"]["answer"]
    # The real tool call was attributed to its skill-map node: the
    # mock's data reached the run, not just the scripted reply.
    assert "check-availability" in (
        final["result"]["plan_trace"]["reception"]["visited"])
    assert final["result"]["evidence"]
