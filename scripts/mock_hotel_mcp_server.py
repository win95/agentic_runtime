"""Mock hotel MCP server — fake PMS + staff/tasks + policy docs.

Implements exactly the 14 tools the ``hotel_receptionist`` agent
expects (see ``docs/hotel-reception.md``). In-memory fake data with a
deterministic seed; state resets on restart (and via ``reset_store()``
for tests). Run it, then point the runtime at it::

    python -m scripts.mock_hotel_mcp_server   # http://127.0.0.1:8081/mcp

    MCP_SERVER_URL=http://127.0.0.1:8081/mcp
    MCP_TRANSPORT=streamable_http
    MCP_TOOL_CAPABILITIES='{"check_room_availability": "hotel_reception", ...}'

Swap in the real server by changing those env vars — the agent,
mission, skill and prompts do not change. The server is standalone:
it imports only the MCP SDK + stdlib, never the app.

Port: 8081 by default (8080 is the OpenSandbox server),
``MOCK_HOTEL_MCP_PORT`` or ``--port`` overrides it.
"""

from __future__ import annotations

import argparse
import datetime
import os

from mcp.server.fastmcp import FastMCP

# The tool functions below are plain functions (tests call them
# directly). `create_server()` registers them on a fresh FastMCP
# instance — a new instance per server start, because the MCP SDK's
# StreamableHTTPSessionManager can only run once per instance.

DEFAULT_PORT = 8081
CURRENCY = "VND"

# Free cancellation up to this many days before check-in; inside the
# window the fee is one night. Mirrors POL-CANCEL below.
FREE_CANCEL_DAYS = 3

ROOM_TYPES = {
    "standard": {
        "room_type": "standard", "name": "Standard",
        "price_per_night": 1_200_000, "total_rooms": 10,
    },
    "deluxe": {
        "room_type": "deluxe", "name": "Deluxe",
        "price_per_night": 2_000_000, "total_rooms": 6,
    },
    "suite": {
        "room_type": "suite", "name": "Suite",
        "price_per_night": 3_500_000, "total_rooms": 2,
    },
}

STAFF = [
    {"staff_id": "NV-101", "name": "Lan", "team": "housekeeping", "status": "available"},
    {"staff_id": "NV-102", "name": "Hùng", "team": "housekeeping", "status": "available"},
    {"staff_id": "NV-103", "name": "Mai", "team": "housekeeping", "status": "available"},
    {"staff_id": "NV-201", "name": "Tuấn", "team": "technicians", "status": "available"},
    {"staff_id": "NV-202", "name": "Dũng", "team": "technicians", "status": "available"},
    {"staff_id": "NV-203", "name": "Linh", "team": "technicians", "status": "available"},
]

POLICIES = [
    {
        "doc_id": "POL-CANCEL",
        "title": "Chính sách hủy phòng",
        "content": (
            "Hủy phòng miễn phí khi hủy trước ngày check-in ít nhất 3 ngày. "
            "Hủy trong vòng 3 ngày trước check-in: phí 1 đêm đầu tiên. "
            "Không đến nhận phòng (no-show): phí 1 đêm đầu tiên. "
            "Tiền đã thanh toán được hoàn lại theo đúng mức phí trên."
        ),
    },
    {
        "doc_id": "POL-CHANGE",
        "title": "Chính sách đổi ngày",
        "content": (
            "Đổi ngày miễn phí 1 lần khi đổi trước ngày check-in ít nhất 3 ngày. "
            "Đổi trong vòng 3 ngày trước check-in: phí 200.000 VND. "
            "Giá phòng mới tính theo giá tại thời điểm đổi; khách trả thêm phần "
            "chênh lệch nếu giá mới cao hơn, không hoàn lại nếu giá mới thấp hơn."
        ),
    },
    {
        "doc_id": "POL-SURCHARGE",
        "title": "Phụ thu",
        "content": (
            "Khách thêm (từ khách thứ 3 trong phòng Standard/Deluxe): 300.000 VND "
            "một khách một đêm. Check-in sớm trước 12:00 hoặc check-out muộn sau "
            "12:00: phụ thu 50% giá 1 đêm. Giờ check-in chuẩn 14:00, check-out "
            "chuẩn 12:00."
        ),
    },
]


def _seed_store() -> dict:
    return {
        "guests": [
            {"guest_id": "guest-001", "name": "Nguyễn Văn An",
             "phone": "0901234567", "email": "an@example.com"},
            {"guest_id": "guest-002", "name": "Trần Thị Bình",
             "phone": "0907654321", "email": "binh@example.com"},
        ],
        "bookings": [
            {"booking_id": "BK-1001", "guest_id": "guest-001",
             "room_type": "deluxe", "check_in": "2026-12-20",
             "check_out": "2026-12-22", "guests": 2,
             "total_price": 4_000_000, "currency": CURRENCY, "status": "CONFIRMED"},
            # Both suites are taken 2026-12-24 -> 2026-12-26 (sold out).
            {"booking_id": "BK-1002", "guest_id": "guest-001",
             "room_type": "suite", "check_in": "2026-12-24",
             "check_out": "2026-12-26", "guests": 2,
             "total_price": 7_000_000, "currency": CURRENCY, "status": "CONFIRMED"},
            {"booking_id": "BK-1003", "guest_id": "guest-002",
             "room_type": "suite", "check_in": "2026-12-24",
             "check_out": "2026-12-26", "guests": 2,
             "total_price": 7_000_000, "currency": CURRENCY, "status": "CONFIRMED"},
        ],
        "tasks": [],
        "counters": {"guest": 3, "booking": 1004, "task": 2001},
    }


_STORE = _seed_store()


def reset_store() -> None:
    """Restore the deterministic seed (tests call this between cases)."""
    global _STORE
    _STORE = _seed_store()


def _parse_date(value: str) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(str(value))
    except ValueError:
        return None


def _nights(check_in: datetime.date, check_out: datetime.date) -> int:
    return (check_out - check_in).days


def _overlapping(room_type: str, check_in: datetime.date,
                 check_out: datetime.date, exclude_id: str | None = None) -> int:
    count = 0
    for booking in _STORE["bookings"]:
        if booking["room_type"] != room_type or booking["status"] != "CONFIRMED":
            continue
        if exclude_id is not None and booking["booking_id"] == exclude_id:
            continue
        start = datetime.date.fromisoformat(booking["check_in"])
        end = datetime.date.fromisoformat(booking["check_out"])
        if start < check_out and check_in < end:
            count += 1
    return count


def _valid_range(check_in: str, check_out: str) -> tuple | dict:
    start = _parse_date(check_in)
    end = _parse_date(check_out)
    if start is None or end is None:
        return {"error": "Dates must be YYYY-MM-DD", "check_in": check_in,
                "check_out": check_out}
    if _nights(start, end) <= 0:
        return {"error": "check_out must be after check_in",
                "check_in": check_in, "check_out": check_out}
    return (start, end)


# --- PMS: rooms, prices, guests, bookings ---
def check_room_availability(check_in: str, check_out: str,
                            room_type: str | None = None,
                            guests: int = 1) -> dict:
    """Kiểm tra phòng trống theo ngày. Trả về số phòng còn trống và giá
    mỗi loại phòng; room_type để trống thì trả tất cả các loại."""
    parsed = _valid_range(check_in, check_out)
    if isinstance(parsed, dict):
        return parsed
    start, end = parsed
    wanted = [room_type] if room_type else sorted(ROOM_TYPES)
    if room_type and room_type not in ROOM_TYPES:
        return {"error": f"Unknown room type: {room_type}",
                "room_types": sorted(ROOM_TYPES)}
    items = []
    for key in wanted:
        info = ROOM_TYPES[key]
        taken = _overlapping(key, start, end)
        items.append({
            "room_type": key, "name": info["name"],
            "available_rooms": info["total_rooms"] - taken,
            "price_per_night": info["price_per_night"], "currency": CURRENCY,
        })
    return {
        "available": any(item["available_rooms"] > 0 for item in items),
        "room_types": items, "check_in": check_in, "check_out": check_out,
        "nights": _nights(start, end), "guests": guests,
    }


def get_room_rates(check_in: str, check_out: str,
                   room_type: str | None = None) -> dict:
    """Báo giá phòng theo ngày: giá mỗi đêm và tổng tiền theo số đêm."""
    parsed = _valid_range(check_in, check_out)
    if isinstance(parsed, dict):
        return parsed
    start, end = parsed
    nights = _nights(start, end)
    wanted = [room_type] if room_type else sorted(ROOM_TYPES)
    if room_type and room_type not in ROOM_TYPES:
        return {"error": f"Unknown room type: {room_type}",
                "room_types": sorted(ROOM_TYPES)}
    rates = []
    for key in wanted:
        info = ROOM_TYPES[key]
        rates.append({
            "room_type": key, "name": info["name"],
            "price_per_night": info["price_per_night"],
            "nights": nights,
            "total_price": info["price_per_night"] * nights,
            "currency": CURRENCY,
        })
    return {"rates": rates, "check_in": check_in, "check_out": check_out,
            "nights": nights}


def find_guest(query: str) -> dict:
    """Tìm khách theo tên hoặc số điện thoại (khớp một phần)."""
    needle = (query or "").strip().casefold()
    found = [g for g in _STORE["guests"]
             if needle and (needle in g["name"].casefold()
                            or needle in g["phone"])]
    return {"guests": found, "count": len(found)}


def register_guest(name: str, phone: str, email: str | None = None) -> dict:
    """Đăng ký khách mới."""
    if not (name or "").strip() or not (phone or "").strip():
        return {"error": "name and phone are required"}
    guest_id = f"guest-{_STORE['counters']['guest']:03d}"
    _STORE["counters"]["guest"] += 1
    guest = {"guest_id": guest_id, "name": name.strip(),
             "phone": phone.strip(), "email": email}
    _STORE["guests"].append(guest)
    return guest


def create_booking(guest_id: str, room_type: str, check_in: str,
                   check_out: str, guests: int = 1) -> dict:
    """Tạo booking cho khách. Chỉ gọi sau khi khách đã xác nhận trong chat."""
    if not any(g["guest_id"] == guest_id for g in _STORE["guests"]):
        return {"error": f"Guest not found: {guest_id}"}
    if room_type not in ROOM_TYPES:
        return {"error": f"Unknown room type: {room_type}",
                "room_types": sorted(ROOM_TYPES)}
    parsed = _valid_range(check_in, check_out)
    if isinstance(parsed, dict):
        return parsed
    start, end = parsed
    info = ROOM_TYPES[room_type]
    if _overlapping(room_type, start, end) >= info["total_rooms"]:
        return {"error": f"No {room_type} room available for these dates",
                "room_type": room_type, "check_in": check_in,
                "check_out": check_out}
    booking_id = f"BK-{_STORE['counters']['booking']}"
    _STORE["counters"]["booking"] += 1
    booking = {
        "booking_id": booking_id, "guest_id": guest_id, "room_type": room_type,
        "check_in": check_in, "check_out": check_out, "guests": guests,
        "total_price": info["price_per_night"] * _nights(start, end),
        "currency": CURRENCY, "status": "CONFIRMED",
    }
    _STORE["bookings"].append(booking)
    return booking


def get_booking(booking_id: str) -> dict:
    """Tra cứu một booking theo mã."""
    for booking in _STORE["bookings"]:
        if booking["booking_id"] == booking_id:
            return booking
    return {"error": f"Booking not found: {booking_id}"}


def cancel_booking(booking_id: str, reason: str | None = None) -> dict:
    """Hủy booking. Chỉ gọi sau khi khách đã xác nhận trong chat.
    Phí hủy: miễn phí khi hủy trước check-in ít nhất 3 ngày, còn lại
    bằng giá 1 đêm đầu tiên."""
    for booking in _STORE["bookings"]:
        if booking["booking_id"] != booking_id:
            continue
        if booking["status"] == "CANCELLED":
            return {"error": f"Booking {booking_id} is already cancelled"}
        check_in = datetime.date.fromisoformat(booking["check_in"])
        days_left = (check_in - datetime.date.today()).days
        fee = 0 if days_left >= FREE_CANCEL_DAYS else ROOM_TYPES[
            booking["room_type"]]["price_per_night"]
        booking["status"] = "CANCELLED"
        return {"booking_id": booking_id, "status": "CANCELLED",
                "fee_vnd": fee, "currency": CURRENCY,
                "reason": reason}
    return {"error": f"Booking not found: {booking_id}"}


def change_booking_dates(booking_id: str, check_in: str,
                         check_out: str) -> dict:
    """Đổi ngày của booking. Chỉ gọi sau khi khách đã xác nhận trong chat."""
    parsed = _valid_range(check_in, check_out)
    if isinstance(parsed, dict):
        return parsed
    start, end = parsed
    for booking in _STORE["bookings"]:
        if booking["booking_id"] != booking_id:
            continue
        if booking["status"] == "CANCELLED":
            return {"error": f"Booking {booking_id} is cancelled"}
        info = ROOM_TYPES[booking["room_type"]]
        if _overlapping(booking["room_type"], start, end,
                        exclude_id=booking_id) >= info["total_rooms"]:
            return {"error": "No room available for the new dates",
                    "room_type": booking["room_type"],
                    "check_in": check_in, "check_out": check_out}
        booking["check_in"] = check_in
        booking["check_out"] = check_out
        booking["total_price"] = info["price_per_night"] * _nights(start, end)
        return booking
    return {"error": f"Booking not found: {booking_id}"}


# --- Staff / tasks ---
def list_staff(team: str | None = None) -> dict:
    """Lấy danh sách nhân viên; team là 'housekeeping' (buồng phòng)
    hoặc 'technicians' (kỹ thuật), để trống thì trả tất cả."""
    if team is not None and team not in ("housekeeping", "technicians"):
        return {"error": f"Unknown team: {team}",
                "teams": ["housekeeping", "technicians"]}
    staff = [s for s in STAFF if team is None or s["team"] == team]
    return {"staff": staff, "count": len(staff)}


def create_task(title: str, description: str, room: str | None = None,
                team: str | None = None) -> dict:
    """Tạo task dịch vụ (hỏng hóc, dọn phòng, ...) cho một phòng."""
    if not (title or "").strip() or not (description or "").strip():
        return {"error": "title and description are required"}
    task_id = f"TK-{_STORE['counters']['task']}"
    _STORE["counters"]["task"] += 1
    task = {"task_id": task_id, "title": title.strip(),
            "description": description.strip(), "room": room,
            "team": team, "status": "OPEN", "assignee": None}
    _STORE["tasks"].append(task)
    return task


def assign_task(task_id: str, staff_id: str) -> dict:
    """Giao task cho một nhân viên."""
    task = next((t for t in _STORE["tasks"] if t["task_id"] == task_id), None)
    if task is None:
        return {"error": f"Task not found: {task_id}"}
    staff = next((s for s in STAFF if s["staff_id"] == staff_id), None)
    if staff is None:
        return {"error": f"Staff not found: {staff_id}"}
    task["assignee"] = staff_id
    task["status"] = "ASSIGNED"
    return task


def get_task(task_id: str) -> dict:
    """Xem trạng thái một task theo mã."""
    for task in _STORE["tasks"]:
        if task["task_id"] == task_id:
            return task
    return {"error": f"Task not found: {task_id}"}


# --- Policy documents ---
def search_policy_documents(query: str) -> dict:
    """Tìm tài liệu chính sách (hủy, đổi ngày, phụ thu) theo từ khóa."""
    needle = (query or "").strip().casefold()
    found = []
    for doc in POLICIES:
        if needle and (needle in doc["title"].casefold()
                       or needle in doc["content"].casefold()):
            found.append({"doc_id": doc["doc_id"], "title": doc["title"],
                          "snippet": doc["content"][:200]})
    return {"documents": found, "count": len(found)}


def read_policy_document(doc_id: str) -> dict:
    """Đọc toàn văn một tài liệu chính sách theo mã."""
    for doc in POLICIES:
        if doc["doc_id"] == doc_id:
            return doc
    return {"error": f"Document not found: {doc_id}",
            "documents": [d["doc_id"] for d in POLICIES]}


HOTEL_TOOL_FUNCTIONS = [
    check_room_availability,
    get_room_rates,
    find_guest,
    register_guest,
    create_booking,
    get_booking,
    cancel_booking,
    change_booking_dates,
    list_staff,
    create_task,
    assign_task,
    get_task,
    search_policy_documents,
    read_policy_document,
]


def create_server(name: str = "hotel-mock") -> FastMCP:
    """A fresh FastMCP instance exposing the 14 contract tools."""
    server = FastMCP(name)
    for fn in HOTEL_TOOL_FUNCTIONS:
        server.tool()(fn)
    return server


# Module-level instance for `python -m scripts.mock_hotel_mcp_server`
# and for `await mcp.list_tools()` in tests.
mcp = create_server()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock hotel MCP server")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("MOCK_HOTEL_MCP_PORT", DEFAULT_PORT)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    import uvicorn

    print(f"Mock hotel MCP server on http://{args.host}:{args.port}/mcp",
          flush=True)
    uvicorn.run(mcp.streamable_http_app(), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
