"""Probe the hotel payment tools directly over MCP — no agent, no router.

The payment scenario in the manual suite is blocked by the router
(v5 review): the agent never reaches the booking-payment flow, so the
payment tools have never been exercised through the live chain. This
probe bypasses the agent entirely and drives the mock PMS server over
the real MCP protocol instead:

1. load the tools the server exposes, and apply the same rule the
   runtime's MCP adapter applies — a tool is usable only when
   ``MCP_TOOL_CAPABILITIES`` maps it to a capability (default deny).
   A missing payment mapping is reported as a FAIL, not a crash, so
   this is also the check for the .env mapping (the .env.example fix
   in b8424141 only helps fresh copies — your own .env needs the 3
   payment tools mapped too).
2. create a booking directly via MCP, then run the payment flow
   against it: record (PENDING) -> confirm (CONFIRMED) -> PAID, plus
   the rejection cases (overpay, double confirm, bad method, zero
   amount, pay after cancel) and the cancel refund
   (paid - POL-CANCEL fee, as a PENDING refund record).

Preconditions: the mock server is running
(``python -m scripts.mock_hotel_mcp_server`` — 127.0.0.1:8082/mcp)
and ``MCP_SERVER_URL`` points at it (``--mcp-url`` overrides the env).
``.env`` is loaded from the repo root at import, so the documented
invocation works without exporting the capability map by hand.

Exit codes: 0 = every check passed, 1 = at least one check failed,
2 = preflight failed (no URL, server unreachable, or a tool the probe
needs is not exposed at all). The probe creates fresh bookings on
random future windows, so repeated runs never collide and no reseed
is needed.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import random
import sys
from pathlib import Path

# The MCP client (httpx) chokes on this VM's default no_proxy, which
# carries bracketed IPv6 entries — same quirk as docker-compose.yml.
# The probe only talks to the mock on localhost, so pin it here.
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Preflight reads the capability map and the server URL straight from
# os.environ, so the repo-root `.env` must be in it — without this the
# documented invocation denies every tool (empty MCP_TOOL_CAPABILITIES).
from _api import load_dotenv  # noqa: E402
from scripts.mock_hotel_mcp_defaults import mock_hotel_mcp_url  # noqa: E402

load_dotenv()

# 17 tools in the hotel contract; the probe needs these 7.
CONTRACT_TOOLS = {
    "check_room_availability",
    "get_room_rates",
    "find_guest",
    "register_guest",
    "create_booking",
    "get_booking",
    "record_payment",
    "confirm_payment",
    "list_payments",
    "cancel_booking",
    "change_booking_dates",
    "list_staff",
    "create_task",
    "assign_task",
    "get_task",
    "search_policy_documents",
    "read_policy_document",
}
NEEDED_TOOLS = {
    "register_guest",
    "create_booking",
    "get_booking",
    "record_payment",
    "confirm_payment",
    "list_payments",
    "cancel_booking",
}
PAYMENT_TOOLS = {"record_payment", "confirm_payment", "list_payments"}

DELUXE_PER_NIGHT = 2_000_000
NIGHTS = 2


class Probe:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"PASS — {name}: {detail}")
        else:
            self.failed += 1
            print(f"FAIL — {name}: {detail}")
        return ok

    def summary(self) -> int:
        total = self.passed + self.failed
        print(f"\n{self.passed}/{total} checks passed")
        return 0 if self.failed == 0 else 1


def _future_window() -> tuple[str, str]:
    start = datetime.date.today() + datetime.timedelta(days=random.randint(30, 60))
    end = start + datetime.timedelta(days=NIGHTS)
    return start.isoformat(), end.isoformat()


async def _call(tools: dict, tool_name: str, **kwargs) -> dict:
    """Invoke one MCP tool and normalise its result to a dict.

    The mock returns errors as data (``{"error": ...}``), not
    exceptions — a returned error dict is a legitimate outcome the
    caller checks for, never a crash."""
    result = await tools[tool_name].ainvoke(kwargs)
    if isinstance(result, list):
        # langchain-mcp-adapters returns the raw MCP content blocks:
        # [{"type": "text", "text": "{...json...}"}].
        texts = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        result = "\n".join(texts)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return {"_raw": result}
    if isinstance(result, dict):
        return result
    return {"_raw": result}


def _pay_state(state: dict) -> str:
    """One-line payment state for check details."""
    return (
        f"{state.get('status')}/{state.get('payment_status')}"
        f" paid={state.get('paid_amount')} pending={state.get('pending_amount')}"
        f" balance={state.get('balance')}"
    )


def _capability_map() -> dict[str, str]:
    raw = (os.environ.get("MCP_TOOL_CAPABILITIES") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return {str(k): str(v) for k, v in parsed.items()} if isinstance(parsed, dict) else {}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mcp-url",
        default=os.environ.get("MCP_SERVER_URL") or mock_hotel_mcp_url(),
        help="mock MCP server URL (default: $MCP_SERVER_URL or mock_hotel_mcp_url())",
    )
    args = parser.parse_args()
    probe = Probe()

    from langchain_mcp_adapters.client import MultiServerMCPClient

    try:
        client = MultiServerMCPClient(
            {"default": {"url": args.mcp_url, "transport": "streamable_http"}}
        )
        loaded = await client.get_tools()
    except Exception as exc:  # noqa: BLE001 — preflight, any failure is fatal
        print(f"PREFLIGHT FAIL — cannot reach the MCP server at {args.mcp_url}: {exc}")
        return 2
    tools = {t.name: t for t in loaded}
    exposed = set(tools)
    missing_exposed = sorted(CONTRACT_TOOLS - exposed)
    probe.check(
        "server exposes the 17-tool contract",
        not missing_exposed,
        f"{len(exposed)} tools loaded"
        + (f"; missing: {', '.join(missing_exposed)}" if missing_exposed else ""),
    )
    if NEEDED_TOOLS - exposed:
        print(
            "PREFLIGHT FAIL — the probe needs "
            + ", ".join(sorted(NEEDED_TOOLS - exposed))
            + " but the server does not expose them"
        )
        return 2

    # The same default-deny rule the runtime adapter applies: a tool
    # with no capability mapping never reaches an agent.
    mapping = _capability_map()
    denied = sorted(t for t in CONTRACT_TOOLS if t not in mapping)
    probe.check(
        "MCP_TOOL_CAPABILITIES maps all 17 tools",
        not denied,
        "all mapped"
        if not denied
        else f"denied: {', '.join(denied)} — add to MCP_TOOL_CAPABILITIES in .env",
    )
    usable = {t for t in exposed if t in mapping}
    if PAYMENT_TOOLS - usable:
        print(
            "PREFLIGHT FAIL — payment tools denied by the capability map: "
            + ", ".join(sorted(PAYMENT_TOOLS - usable))
        )
        return 1

    # --- Booking A: full payment, bank transfer ----------------------
    guest = await _call(
        tools,
        "register_guest",
        name="Probe Guest",
        phone=f"090999{random.randint(1000, 9999)}",
    )
    if not probe.check(
        "register probe guest",
        "guest_id" in guest and "error" not in guest,
        f"guest_id={guest.get('guest_id', guest)}",
    ):
        return probe.summary()
    guest_id = guest["guest_id"]

    check_in, check_out = _future_window()
    booking_a = await _call(
        tools,
        "create_booking",
        guest_id=guest_id,
        room_type="deluxe",
        check_in=check_in,
        check_out=check_out,
        guests=2,
    )
    if not probe.check(
        "create booking A (direct MCP)",
        "booking_id" in booking_a and "error" not in booking_a,
        f"{booking_a.get('booking_id', booking_a)} deluxe {check_in} -> {check_out}",
    ):
        return probe.summary()
    code_a = booking_a["booking_id"]
    total_a = booking_a["total_price"]
    probe.check(
        "booking A total is 2 nights x 2,000,000",
        total_a == DELUXE_PER_NIGHT * NIGHTS,
        f"total_price={total_a}",
    )

    state = await _call(tools, "get_booking", booking_id=code_a)
    probe.check(
        "booking A starts UNPAID",
        state.get("status") == "CONFIRMED"
        and state.get("payment_status") == "UNPAID"
        and state.get("balance") == total_a,
        _pay_state(state),
    )

    payment = await _call(
        tools, "record_payment", booking_id=code_a, amount=total_a, method="bank_transfer"
    )
    probe.check(
        "record full payment -> PENDING",
        payment.get("status") == "PENDING" and "error" not in payment,
        f"payment_id={payment.get('payment_id', payment)}",
    )
    payment_id = payment.get("payment_id", "")

    state = await _call(tools, "get_booking", booking_id=code_a)
    probe.check(
        "booking A PENDING after record",
        state.get("payment_status") == "PENDING"
        and state.get("pending_amount") == total_a
        and state.get("balance") == 0,
        _pay_state(state),
    )

    payments = await _call(tools, "list_payments", booking_id=code_a)
    listed = payments.get("payments", [])
    probe.check(
        "list_payments shows the PENDING payment",
        len(listed) == 1
        and listed[0].get("status") == "PENDING"
        and listed[0].get("amount") == total_a
        and listed[0].get("method") == "bank_transfer",
        f"{len(listed)} payment(s)",
    )

    confirmed = await _call(tools, "confirm_payment", payment_id=payment_id)
    probe.check(
        "confirm payment -> CONFIRMED",
        confirmed.get("status") == "CONFIRMED" and "error" not in confirmed,
        f"payment_id={confirmed.get('payment_id', confirmed)}",
    )

    state = await _call(tools, "get_booking", booking_id=code_a)
    probe.check(
        "booking A PAID, balance 0",
        state.get("payment_status") == "PAID"
        and state.get("paid_amount") == total_a
        and state.get("balance") == 0,
        _pay_state(state),
    )

    overpay = await _call(tools, "record_payment", booking_id=code_a, amount=1, method="cash")
    probe.check(
        "overpay on a PAID booking is rejected",
        "error" in overpay and "balance" in str(overpay.get("error", "")),
        str(overpay.get("error", overpay)),
    )

    double = await _call(tools, "confirm_payment", payment_id=payment_id)
    probe.check(
        "double confirm is rejected",
        "error" in double,
        str(double.get("error", double)),
    )

    # --- Booking B: bad inputs, partial payment, cancel refund -------
    check_in_b, check_out_b = _future_window()
    booking_b = await _call(
        tools,
        "create_booking",
        guest_id=guest_id,
        room_type="deluxe",
        check_in=check_in_b,
        check_out=check_out_b,
        guests=2,
    )
    if not probe.check(
        "create booking B (direct MCP)",
        "booking_id" in booking_b and "error" not in booking_b,
        f"{booking_b.get('booking_id', booking_b)} deluxe {check_in_b} -> {check_out_b}",
    ):
        return probe.summary()
    code_b = booking_b["booking_id"]
    total_b = booking_b["total_price"]

    bad_method = await _call(
        tools, "record_payment", booking_id=code_b, amount=1_000_000, method="crypto"
    )
    probe.check(
        "unknown method is rejected",
        "error" in bad_method and "method" in str(bad_method.get("error", "")).lower(),
        str(bad_method.get("error", bad_method)),
    )

    zero = await _call(tools, "record_payment", booking_id=code_b, amount=0, method="cash")
    probe.check(
        "zero amount is rejected",
        "error" in zero,
        str(zero.get("error", zero)),
    )

    partial = await _call(
        tools, "record_payment", booking_id=code_b, amount=1_000_000, method="cash"
    )
    probe.check(
        "record partial payment -> PENDING",
        partial.get("status") == "PENDING" and "error" not in partial,
        f"payment_id={partial.get('payment_id', partial)}",
    )
    partial_id = partial.get("payment_id", "")
    confirmed_b = await _call(tools, "confirm_payment", payment_id=partial_id)
    probe.check(
        "confirm partial payment -> CONFIRMED",
        confirmed_b.get("status") == "CONFIRMED" and "error" not in confirmed_b,
        f"payment_id={confirmed_b.get('payment_id', confirmed_b)}",
    )
    state = await _call(tools, "get_booking", booking_id=code_b)
    probe.check(
        "booking B PARTIAL after 1,000,000 of 4,000,000",
        state.get("payment_status") == "PARTIAL"
        and state.get("paid_amount") == 1_000_000
        and state.get("balance") == total_b - 1_000_000,
        _pay_state(state),
    )

    cancelled = await _call(tools, "cancel_booking", booking_id=code_b)
    probe.check(
        "cancel booking B refunds paid minus fee (free >= 3 days out)",
        cancelled.get("status") == "CANCELLED"
        and cancelled.get("fee_vnd") == 0
        and cancelled.get("refund_amount") == 1_000_000
        and cancelled.get("refund_status") == "PENDING",
        f"fee={cancelled.get('fee_vnd')} refund={cancelled.get('refund_amount')}"
        f" {cancelled.get('refund_status')}",
    )
    state = await _call(tools, "get_booking", booking_id=code_b)
    probe.check(
        "booking B CANCELLED with refund recorded",
        state.get("status") == "CANCELLED" and state.get("refund_amount") == 1_000_000,
        f"status={state.get('status')} refund={state.get('refund_amount')}",
    )

    after_cancel = await _call(tools, "record_payment", booking_id=code_b, amount=1, method="cash")
    probe.check(
        "pay after cancel is rejected",
        "error" in after_cancel and "cancell" in str(after_cancel.get("error", "")).lower(),
        str(after_cancel.get("error", after_cancel)),
    )

    return probe.summary()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
