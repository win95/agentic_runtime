"""Layer enforcement tests — L1 read-open + guest scope, L2 call-time
gate, L3 validate_results FAIL (tran 2026-11-13).

Principle under test: reads stay open so the agent can build context;
what is enforced is the *data* scope (asset/guest wrappers) and the
*effect* layer (call-time hook). Fake guest/booking tools stand in
for the MCP contract — the wrapper logic is identical, the PMS is not
involved.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import tool as langchain_tool

import runtime.tools  # noqa: F401 — importing registers the built-in tools
from app.services import session_scope_service
from runtime.hooks import builtin
from runtime.hooks.events import HookContext
from runtime.mission_graph import validate_results
from runtime.scope import guest_id_from_result, scope_violations_in_evidence
from runtime.tools.registry import registry

HOTEL_CONTEXT = {
    "mission_id": "hotel_reception",
    "tenant": "default",
    "execution_id": "exec_test",
    "capabilities": ["hotel_reception"],
    "allowed_tools": [
        "find_guest",
        "register_guest",
        "get_booking",
        "create_booking",
        "cancel_booking",
        "change_booking_dates",
    ],
    "session_id": "sess-layer-test",
}

_GUESTS = {
    "guest-001": {"guest_id": "guest-001", "name": "Nguyễn Văn An", "phone": "0901234567"},
    "guest-002": {"guest_id": "guest-002", "name": "Trần Thị Bình", "phone": "0907654321"},
}
_BOOKINGS = {
    "BK-1001": {"booking_id": "BK-1001", "guest_id": "guest-001", "status": "CONFIRMED"},
    "BK-1002": {"booking_id": "BK-1002", "guest_id": "guest-002", "status": "CONFIRMED"},
}


@pytest.fixture()
def _guest_tools(monkeypatch):
    """Fake guest/booking tools registered like the MCP contract."""
    calls: list[tuple[str, dict]] = []
    _store: dict[str, str] = {}
    monkeypatch.setattr("runtime.scope.get_bound_guest_id", lambda sid: _store.get(sid))
    monkeypatch.setattr(
        "runtime.scope.bind_guest",
        lambda sid, gid, phone=None, mid=None: _store.setdefault(sid, gid),
    )

    @langchain_tool
    def find_guest(query: str) -> dict:
        """Find a guest by name or phone."""
        calls.append(("find_guest", {"query": query}))
        found = [
            g for g in _GUESTS.values()
            if query.lower() in g["name"].lower() or query in g["phone"]
        ]
        return {"guests": found, "count": len(found)}

    @langchain_tool
    def register_guest(name: str, phone: str) -> dict:
        """Register a new guest."""
        calls.append(("register_guest", {"name": name, "phone": phone}))
        return {"guest_id": "guest-003", "name": name, "phone": phone}

    @langchain_tool
    def get_booking(booking_id: str) -> dict:
        """Look up a booking by id."""
        calls.append(("get_booking", {"booking_id": booking_id}))
        booking = _BOOKINGS.get(booking_id)
        if booking is None:
            return {"error": f"Booking not found: {booking_id}"}
        return dict(booking)

    @langchain_tool
    def create_booking(guest_id: str, room_type: str) -> dict:
        """Create a booking for a guest."""
        calls.append(("create_booking", {"guest_id": guest_id, "room_type": room_type}))
        return {"booking_id": "BK-1004", "guest_id": guest_id, "status": "CONFIRMED"}

    @langchain_tool
    def cancel_booking(booking_id: str) -> dict:
        """Cancel a booking."""
        calls.append(("cancel_booking", {"booking_id": booking_id}))
        return {"booking_id": booking_id, "status": "CANCELLED"}

    @langchain_tool
    def change_booking_dates(booking_id: str, check_in: str, check_out: str) -> dict:
        """Change a booking's dates."""
        calls.append((
            "change_booking_dates",
            {"booking_id": booking_id, "check_in": check_in, "check_out": check_out},
        ))
        return {"booking_id": booking_id, "status": "CONFIRMED"}

    for fn, read_only in (
        (find_guest, True),
        (register_guest, False),
        (get_booking, True),
        (create_booking, False),
        (cancel_booking, False),
        (change_booking_dates, False),
    ):
        registry.register(fn, capability="hotel_reception", permission="mcp",
                          read_only=read_only)
    tools = {t.name: t for t in registry.get_langchain_tools(dict(HOTEL_CONTEXT))}
    return {"tools": tools, "calls": calls, "store": _store}


def _invoke(tools, tool_name, **kwargs):
    return tools[tool_name].invoke(kwargs)


# --- L1: read-open -------------------------------------------------------

def test_read_tools_listed_without_allowed_tools():
    """L1: reads need capability only — dropping a read tool from
    allowed_tools must not starve the agent of context."""
    context = {
        "mission_id": "fault_analysis",
        "tenant": "default",
        "execution_id": "exec_test",
        "capabilities": ["fault_history", "maintenance_analysis", "knowledge_retrieval"],
        "allowed_tools": ["search_fault_history"],
    }
    names = {t.name for t in registry.get_langchain_tools(context)}
    assert {
        "search_fault_history",
        "get_maintenance_records",
        "search_technical_documents",
        "calculate_failure_rate",
    } <= names
    admin_context = {
        "mission_id": "system_admin",
        "tenant": "default",
        "execution_id": "exec_test",
        "capabilities": ["system_admin"],
        "allowed_tools": ["get_missions"],
    }
    admin_names = {t.name for t in registry.get_langchain_tools(admin_context)}
    assert {"get_missions", "get_executions", "list_agents", "system_health"} <= admin_names
    # Writes still need the pin.
    assert "save_mission" not in admin_names
    assert "delete_mission" not in admin_names


# --- L1: guest scope -----------------------------------------------------

def test_find_guest_binds_first_sight(_guest_tools):
    tools, store = _guest_tools["tools"], _guest_tools["store"]
    result = _invoke(tools, "find_guest", query="An")
    assert result["count"] == 1
    assert result["guests"][0]["guest_id"] == "guest-001"
    assert store.get("sess-layer-test") == "guest-001"


def test_find_guest_filters_to_bound_guest(_guest_tools, monkeypatch):
    tools, store = _guest_tools["tools"], _guest_tools["store"]
    store["sess-layer-test"] = "guest-001"
    result = _invoke(tools, "find_guest", query="Bình")
    assert result == {"guests": [], "count": 0}
    result = _invoke(tools, "find_guest", query="An")
    assert result["count"] == 1


def test_get_booking_withholds_other_guest(_guest_tools):
    tools, calls = _guest_tools["tools"], _guest_tools["calls"]
    _guest_tools["store"]["sess-layer-test"] = "guest-001"
    result = _invoke(tools, "get_booking", booking_id="BK-1002")
    assert "error" in result
    assert "guest-002" not in str(result)
    # The read ran (L1 reads stay open); only the data was withheld.
    assert ("get_booking", {"booking_id": "BK-1002"}) in calls


def test_get_booking_binds_when_unbound(_guest_tools):
    tools, store = _guest_tools["tools"], _guest_tools["store"]
    result = _invoke(tools, "get_booking", booking_id="BK-1001")
    assert result["guest_id"] == "guest-001"
    assert store.get("sess-layer-test") == "guest-001"


def test_create_booking_refuses_other_guest(_guest_tools):
    tools, calls = _guest_tools["tools"], _guest_tools["calls"]
    _guest_tools["store"]["sess-layer-test"] = "guest-001"
    result = _invoke(tools, "create_booking", guest_id="guest-002", room_type="deluxe")
    assert "error" in result
    assert not [c for c in calls if c[0] == "create_booking"]


def test_create_booking_binds_first_sight(_guest_tools):
    tools, store = _guest_tools["tools"], _guest_tools["store"]
    result = _invoke(tools, "create_booking", guest_id="guest-001", room_type="deluxe")
    assert result["status"] == "CONFIRMED"
    assert store.get("sess-layer-test") == "guest-001"


def test_cancel_refuses_other_guest_booking(_guest_tools):
    tools, calls = _guest_tools["tools"], _guest_tools["calls"]
    _guest_tools["store"]["sess-layer-test"] = "guest-001"
    result = _invoke(tools, "cancel_booking", booking_id="BK-1002")
    assert "error" in result
    assert not [c for c in calls if c[0] == "cancel_booking"]


def test_cancel_allows_own_booking(_guest_tools):
    tools, calls = _guest_tools["tools"], _guest_tools["calls"]
    _guest_tools["store"]["sess-layer-test"] = "guest-001"
    result = _invoke(tools, "cancel_booking", booking_id="BK-1001")
    assert result["status"] == "CANCELLED"
    assert ("cancel_booking", {"booking_id": "BK-1001"}) in calls


def test_register_guest_refuses_when_bound(_guest_tools):
    tools, calls = _guest_tools["tools"], _guest_tools["calls"]
    _guest_tools["store"]["sess-layer-test"] = "guest-001"
    result = _invoke(tools, "register_guest", name="Người Lạ", phone="0900000000")
    assert "error" in result
    assert not [c for c in calls if c[0] == "register_guest"]


# --- L2: enforce_layer hook ----------------------------------------------

def _hook_ctx(tool_name, tool_input=None, **overrides):
    ctx = HookContext(
        execution_id="e1",
        mission_id="system_admin",
        tool_name=tool_name,
        tool_input=tool_input or {},
        capabilities=["system_admin"],
        allowed_tools=["get_missions"],
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def test_enforce_layer_denies_unknown_tool():
    decision = builtin.enforce_layer(_hook_ctx("not_a_tool"))
    assert decision is not None and decision.action == "deny"


def test_enforce_layer_denies_capability_mismatch():
    decision = builtin.enforce_layer(
        _hook_ctx("search_fault_history", capabilities=["system_admin"])
    )
    assert decision is not None and decision.action == "deny"


def test_enforce_layer_denies_write_outside_allowed_tools():
    decision = builtin.enforce_layer(_hook_ctx("save_mission", {"mission_id": "x"}))
    assert decision is not None and decision.action == "deny"


def test_enforce_layer_allows_read_outside_allowed_tools():
    assert builtin.enforce_layer(_hook_ctx("get_missions")) is None
    assert builtin.enforce_layer(
        _hook_ctx("get_executions", capabilities=["system_admin"])
    ) is None


def test_enforce_layer_allows_write_in_allowed_tools():
    assert builtin.enforce_layer(
        _hook_ctx("get_missions", capabilities=["system_admin"],
                  allowed_tools=["get_missions"])
    ) is None


def test_enforce_layer_skips_builtin_tools():
    assert builtin.enforce_layer(_hook_ctx("read_file", {"path": "/x"})) is None


# --- L3: validate_results FAILs on out-of-scope evidence -----------------

def _valid_state(**overrides):
    evidence = overrides.pop("evidence", [])
    state = {
        "mission": {
            "output_schema": {"properties": {
                "answer": {}, "actions_taken": {}, "evidence": {},
            }},
        },
        "evidence": evidence,
        "result": {
            "answer": "done",
            "actions_taken": [],
            "evidence": evidence,
        },
        "errors": [],
        "input": {},
    }
    state.update(overrides)
    return state


def test_validate_fails_evidence_outside_asset_scope():
    evidence = [{
        "source": "fault_history", "summary": "fault on O99",
        "data": {"asset_id": "O99", "id": "fh-1"},
    }]
    out = validate_results(_valid_state(
        evidence=evidence, input={"asset_id": "O21"},
    ))
    assert out["status"] == "FAILED"
    assert any("outside execution scope" in e for e in out["errors"])


def test_validate_fails_evidence_outside_guest_scope(monkeypatch):
    monkeypatch.setattr(
        "runtime.scope.get_bound_guest_id", lambda sid: "guest-001"
    )
    evidence = [{
        "source": "get_booking", "summary": "booking BK-1002",
        "data": {"booking_id": "BK-1002", "guest_id": "guest-002"},
    }]
    out = validate_results(_valid_state(
        evidence=evidence, session_id="sess-layer-test",
    ))
    assert out["status"] == "FAILED"
    assert any("outside execution scope" in e for e in out["errors"])


def test_validate_passes_in_scope_evidence(monkeypatch):
    monkeypatch.setattr(
        "runtime.scope.get_bound_guest_id", lambda sid: "guest-001"
    )
    evidence = [{
        "source": "get_booking", "summary": "booking BK-1001",
        "data": {"booking_id": "BK-1001", "guest_id": "guest-001"},
    }]
    out = validate_results(_valid_state(
        evidence=evidence, session_id="sess-layer-test",
        input={"asset_id": "O21"},
    ))
    assert out["status"] == "SUCCEEDED"


def test_scope_violations_pure_helpers():
    assert guest_id_from_result({"guest_id": "guest-001"}) == "guest-001"
    assert guest_id_from_result({"guests": [{"guest_id": "guest-002"}]}) == "guest-002"
    assert guest_id_from_result({"guests": []}) is None
    assert scope_violations_in_evidence(
        [{"source": "x", "data": {"asset_id": "O21"}}], asset_id="O21"
    ) == []
    assert len(scope_violations_in_evidence(
        [{"source": "x", "data": {"asset_id": "O99"}}], asset_id="O21"
    )) == 1


# --- Session scope service ------------------------------------------------

def test_session_scope_first_sight_wins(db_session):
    assert session_scope_service.get_session_guest(db_session, "s1") is None
    row = session_scope_service.bind_session_guest(db_session, "s1", "guest-001")
    assert row is not None and row.guest_id == "guest-001"
    again = session_scope_service.bind_session_guest(db_session, "s1", "guest-002")
    assert again.guest_id == "guest-001"
    assert session_scope_service.get_session_guest(db_session, "s1").guest_id == "guest-001"
