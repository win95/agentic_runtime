"""Mock hotel MCP server — fake PMS + staff/tasks + policy docs.

Implements exactly the 14 tools the ``hotel_receptionist`` agent
expects (see ``docs/hotel-reception.md``). The mutable data (guests,
bookings, tasks, id counters) lives in SQLite, so a create / update /
delete survives a server restart and can be checked afterwards — e.g.::

    sqlite3 mock_hotel.db "SELECT booking_id, status FROM bookings;"
    python -m scripts.mock_hotel_mcp_server --dump

Run it, then point the runtime at it::

    python -m scripts.mock_hotel_mcp_server   # http://127.0.0.1:8081/mcp

    MCP_SERVER_URL=http://127.0.0.1:8081/mcp
    MCP_TRANSPORT=streamable_http
    MCP_TOOL_CAPABILITIES='{"check_room_availability": "hotel_reception", ...}'

Swap in the real server by changing those env vars — the agent,
mission, skill and prompts do not change. The server is standalone:
it imports only the MCP SDK + stdlib, never the app.

Database: ``mock_hotel.db`` in the working directory by default
(``MOCK_HOTEL_MCP_DB`` or ``--db`` overrides it; ``*.db`` is
gitignored). The first open seeds the deterministic data below; later
opens keep what is there. ``--reseed`` wipes and re-seeds (the clean
state for a manual run), ``--dump`` prints the whole store as JSON
and exits. Tests point the store at a temp file instead.

Port: 8081 by default (8080 is the OpenSandbox server),
``MOCK_HOTEL_MCP_PORT`` or ``--port`` overrides it.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sqlite3
import threading

import anyio
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


# --- SQLite store --------------------------------------------------------
# One connection + one lock: FastMCP may call the (sync) tools from
# different threads, so every statement runs serialized and every
# write commits before the tool returns. The store opens lazily on
# first use so importing this module never touches the filesystem
# (tests point it at a temp file first).

_SCHEMA = """
CREATE TABLE IF NOT EXISTS guests (
    guest_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    email TEXT
);
CREATE TABLE IF NOT EXISTS bookings (
    booking_id TEXT PRIMARY KEY,
    guest_id TEXT NOT NULL,
    room_type TEXT NOT NULL,
    check_in TEXT NOT NULL,
    check_out TEXT NOT NULL,
    guests INTEGER NOT NULL,
    total_price INTEGER NOT NULL,
    currency TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    room TEXT,
    team TEXT,
    status TEXT NOT NULL,
    assignee TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_SEED_GUESTS = [
    {"guest_id": "guest-001", "name": "Nguyễn Văn An",
     "phone": "0901234567", "email": "an@example.com"},
    {"guest_id": "guest-002", "name": "Trần Thị Bình",
     "phone": "0907654321", "email": "binh@example.com"},
]

_SEED_BOOKINGS = [
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
]

_SEED_COUNTERS = {"guest": 3, "booking": 1004, "task": 2001}

_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None
_DB_PATH: str | None = None


def default_db_path() -> str:
    """DB file: ``MOCK_HOTEL_MCP_DB`` or ``mock_hotel.db`` in the CWD."""
    return os.environ.get("MOCK_HOTEL_MCP_DB", "mock_hotel.db")


def open_store(db_path: str | None = None) -> None:
    """(Re)open the store at ``db_path`` (default: :func:`default_db_path`).

    Creates the schema and seeds the deterministic data when the
    database is empty; an existing database keeps its rows. Tests
    point this at a temp file for isolation.
    """
    global _CONN, _DB_PATH
    path = db_path or default_db_path()
    with _LOCK:
        if _CONN is not None:
            _CONN.close()
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        empty = conn.execute("SELECT COUNT(*) FROM guests").fetchone()[0] == 0
        if empty:
            _seed_locked(conn)
        conn.commit()
        _CONN = conn
        _DB_PATH = path


def close_store() -> None:
    """Close the store connection (tests call this after each case)."""
    global _CONN, _DB_PATH
    with _LOCK:
        if _CONN is not None:
            _CONN.close()
            _CONN = None
            _DB_PATH = None


def store_path() -> str:
    """Path of the currently open database (opens the default first)."""
    _db()
    return _DB_PATH or default_db_path()


def _db() -> sqlite3.Connection:
    if _CONN is None:
        open_store()
    assert _CONN is not None
    return _CONN


def _seed_locked(conn: sqlite3.Connection) -> None:
    for guest in _SEED_GUESTS:
        conn.execute(
            "INSERT INTO guests (guest_id, name, phone, email) VALUES (?, ?, ?, ?)",
            (guest["guest_id"], guest["name"], guest["phone"], guest["email"]),
        )
    for booking in _SEED_BOOKINGS:
        conn.execute(
            "INSERT INTO bookings (booking_id, guest_id, room_type, check_in,"
            " check_out, guests, total_price, currency, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (booking["booking_id"], booking["guest_id"], booking["room_type"],
             booking["check_in"], booking["check_out"], booking["guests"],
             booking["total_price"], booking["currency"], booking["status"]),
        )
    for key, value in _SEED_COUNTERS.items():
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, str(value)),
        )


def reset_store() -> None:
    """Restore the deterministic seed (tests call this between cases)."""
    conn = _db()
    with _LOCK:
        conn.execute("DELETE FROM tasks")
        conn.execute("DELETE FROM bookings")
        conn.execute("DELETE FROM guests")
        conn.execute("DELETE FROM meta")
        _seed_locked(conn)
        conn.commit()


def dump_store() -> dict:
    """The whole store as plain dicts (guests/bookings/tasks/counters)."""
    conn = _db()
    with _LOCK:
        guests = [dict(r) for r in conn.execute(
            "SELECT * FROM guests ORDER BY guest_id")]
        bookings = [dict(r) for r in conn.execute(
            "SELECT * FROM bookings ORDER BY booking_id")]
        tasks = [dict(r) for r in conn.execute(
            "SELECT * FROM tasks ORDER BY task_id")]
        counters = {r["key"]: int(r["value"]) for r in conn.execute(
            "SELECT * FROM meta")}
    return {"guests": guests, "bookings": bookings, "tasks": tasks,
            "counters": counters}


def _next_counter_locked(conn: sqlite3.Connection, key: str) -> int:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    value = int(row["value"]) if row else 1
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        (key, str(value + 1)),
    )
    return value


def _room_key(room_type: str | None) -> str | None:
    """Room-type names are case-insensitive ('Standard' -> 'standard')."""
    if room_type is None:
        return None
    return str(room_type).strip().lower()


def _parse_date(value: str) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(str(value))
    except ValueError:
        return None


def _nights(check_in: datetime.date, check_out: datetime.date) -> int:
    return (check_out - check_in).days


def _overlapping_locked(conn: sqlite3.Connection, room_type: str,
                        check_in: datetime.date, check_out: datetime.date,
                        exclude_id: str | None = None) -> int:
    # ISO dates compare lexicographically; overlap is start < new_end
    # and new_start < end. Only CONFIRMED bookings occupy a room.
    # Caller must hold _LOCK.
    sql = ("SELECT COUNT(*) FROM bookings WHERE room_type = ?"
           " AND status = 'CONFIRMED' AND check_in < ? AND ? < check_out")
    params: list = [room_type, check_out.isoformat(), check_in.isoformat()]
    if exclude_id is not None:
        sql += " AND booking_id != ?"
        params.append(exclude_id)
    return conn.execute(sql, params).fetchone()[0]


def _overlapping(room_type: str, check_in: datetime.date,
                 check_out: datetime.date, exclude_id: str | None = None) -> int:
    conn = _db()
    with _LOCK:
        return _overlapping_locked(conn, room_type, check_in, check_out,
                                   exclude_id)


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
    room_type = _room_key(room_type)
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
    room_type = _room_key(room_type)
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
    conn = _db()
    with _LOCK:
        rows = conn.execute("SELECT * FROM guests").fetchall()
    found = [dict(r) for r in rows
             if needle and (needle in r["name"].casefold()
                            or needle in r["phone"])]
    return {"guests": found, "count": len(found)}


def register_guest(name: str, phone: str, email: str | None = None) -> dict:
    """Đăng ký khách mới."""
    if not (name or "").strip() or not (phone or "").strip():
        return {"error": "name and phone are required"}
    conn = _db()
    with _LOCK:
        number = _next_counter_locked(conn, "guest")
        guest_id = f"guest-{number:03d}"
        conn.execute(
            "INSERT INTO guests (guest_id, name, phone, email)"
            " VALUES (?, ?, ?, ?)",
            (guest_id, name.strip(), phone.strip(), email),
        )
        conn.commit()
    return {"guest_id": guest_id, "name": name.strip(),
            "phone": phone.strip(), "email": email}


def create_booking(guest_id: str, room_type: str, check_in: str,
                   check_out: str, guests: int = 1) -> dict:
    """Tạo booking cho khách. Chỉ gọi sau khi khách đã xác nhận trong chat."""
    room_type = _room_key(room_type)
    if room_type not in ROOM_TYPES:
        return {"error": f"Unknown room type: {room_type}",
                "room_types": sorted(ROOM_TYPES)}
    parsed = _valid_range(check_in, check_out)
    if isinstance(parsed, dict):
        return parsed
    start, end = parsed
    info = ROOM_TYPES[room_type]
    conn = _db()
    # One locked section for check + insert: two concurrent creates
    # cannot both see the last free room and overbook it.
    with _LOCK:
        exists = conn.execute(
            "SELECT 1 FROM guests WHERE guest_id = ?", (guest_id,)).fetchone()
        if not exists:
            return {"error": f"Guest not found: {guest_id}"}
        if _overlapping_locked(conn, room_type, start, end) >= info["total_rooms"]:
            return {"error": f"No {room_type} room available for these dates",
                    "room_type": room_type, "check_in": check_in,
                    "check_out": check_out}
        number = _next_counter_locked(conn, "booking")
        booking_id = f"BK-{number}"
        booking = {
            "booking_id": booking_id, "guest_id": guest_id, "room_type": room_type,
            "check_in": check_in, "check_out": check_out, "guests": guests,
            "total_price": info["price_per_night"] * _nights(start, end),
            "currency": CURRENCY, "status": "CONFIRMED",
        }
        conn.execute(
            "INSERT INTO bookings (booking_id, guest_id, room_type, check_in,"
            " check_out, guests, total_price, currency, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (booking["booking_id"], booking["guest_id"], booking["room_type"],
             booking["check_in"], booking["check_out"], booking["guests"],
             booking["total_price"], booking["currency"], booking["status"]),
        )
        conn.commit()
    return booking


def get_booking(booking_id: str) -> dict:
    """Tra cứu một booking theo mã."""
    conn = _db()
    with _LOCK:
        row = conn.execute(
            "SELECT * FROM bookings WHERE booking_id = ?",
            (booking_id,)).fetchone()
    if row is None:
        return {"error": f"Booking not found: {booking_id}"}
    return dict(row)


def cancel_booking(booking_id: str, reason: str | None = None) -> dict:
    """Hủy booking. Chỉ gọi sau khi khách đã xác nhận trong chat.
    Phí hủy: miễn phí khi hủy trước check-in ít nhất 3 ngày, còn lại
    bằng giá 1 đêm đầu tiên."""
    conn = _db()
    with _LOCK:
        row = conn.execute(
            "SELECT * FROM bookings WHERE booking_id = ?",
            (booking_id,)).fetchone()
        if row is None:
            return {"error": f"Booking not found: {booking_id}"}
        booking = dict(row)
        if booking["status"] == "CANCELLED":
            return {"error": f"Booking {booking_id} is already cancelled"}
        check_in = datetime.date.fromisoformat(booking["check_in"])
        days_left = (check_in - datetime.date.today()).days
        fee = 0 if days_left >= FREE_CANCEL_DAYS else ROOM_TYPES[
            booking["room_type"]]["price_per_night"]
        conn.execute(
            "UPDATE bookings SET status = 'CANCELLED' WHERE booking_id = ?",
            (booking_id,),
        )
        conn.commit()
    return {"booking_id": booking_id, "status": "CANCELLED",
            "fee_vnd": fee, "currency": CURRENCY,
            "reason": reason}


def change_booking_dates(booking_id: str, check_in: str,
                         check_out: str) -> dict:
    """Đổi ngày của booking. Chỉ gọi sau khi khách đã xác nhận trong chat."""
    parsed = _valid_range(check_in, check_out)
    if isinstance(parsed, dict):
        return parsed
    start, end = parsed
    conn = _db()
    with _LOCK:
        row = conn.execute(
            "SELECT * FROM bookings WHERE booking_id = ?",
            (booking_id,)).fetchone()
        if row is None:
            return {"error": f"Booking not found: {booking_id}"}
        booking = dict(row)
        if booking["status"] == "CANCELLED":
            return {"error": f"Booking {booking_id} is cancelled"}
        info = ROOM_TYPES[booking["room_type"]]
        if _overlapping_locked(conn, booking["room_type"], start, end,
                               exclude_id=booking_id) >= info["total_rooms"]:
            return {"error": "No room available for the new dates",
                    "room_type": booking["room_type"],
                    "check_in": check_in, "check_out": check_out}
        total = info["price_per_night"] * _nights(start, end)
        conn.execute(
            "UPDATE bookings SET check_in = ?, check_out = ?,"
            " total_price = ? WHERE booking_id = ?",
            (check_in, check_out, total, booking_id),
        )
        conn.commit()
        booking["check_in"] = check_in
        booking["check_out"] = check_out
        booking["total_price"] = total
    return booking


# --- Staff / tasks ---
def list_staff(team: str | None = None) -> dict:
    """Lấy danh sách nhân viên; team là 'housekeeping' (buồng phòng)
    hoặc 'technicians' (kỹ thuật), để trống thì trả tất cả."""
    team = str(team).strip().lower() if team is not None else None
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
    room = str(room).strip() if room is not None else None
    team = str(team).strip().lower() if team is not None else None
    conn = _db()
    with _LOCK:
        number = _next_counter_locked(conn, "task")
        task_id = f"TK-{number}"
        task = {"task_id": task_id, "title": title.strip(),
                "description": description.strip(), "room": room,
                "team": team, "status": "OPEN", "assignee": None}
        conn.execute(
            "INSERT INTO tasks (task_id, title, description, room, team,"
            " status, assignee) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task["task_id"], task["title"], task["description"],
             task["room"], task["team"], task["status"], task["assignee"]),
        )
        conn.commit()
    return task


def assign_task(task_id: str, staff_id: str) -> dict:
    """Giao task cho một nhân viên."""
    staff = next((s for s in STAFF if s["staff_id"] == staff_id), None)
    if staff is None:
        return {"error": f"Staff not found: {staff_id}"}
    conn = _db()
    with _LOCK:
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            return {"error": f"Task not found: {task_id}"}
        task = dict(row)
        conn.execute(
            "UPDATE tasks SET assignee = ?, status = 'ASSIGNED'"
            " WHERE task_id = ?",
            (staff_id, task_id),
        )
        conn.commit()
        task["assignee"] = staff_id
        task["status"] = "ASSIGNED"
    return task


def get_task(task_id: str) -> dict:
    """Xem trạng thái một task theo mã."""
    conn = _db()
    with _LOCK:
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if row is None:
        return {"error": f"Task not found: {task_id}"}
    return dict(row)


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


class _SuppressClosedSseWriter(logging.Filter):
    """Drop the benign session-teardown ERROR from mcp==1.30.0.

    When a client terminates its session, the SDK closes the standalone
    SSE stream while the writer task is blocked in ``receive()``, so the
    writer raises ``anyio.ClosedResourceError`` and the SDK logs it as
    "Error in standalone SSE writer" — on EVERY clean disconnect, even
    though the server survives and keeps serving (verified: a second
    client connects fine right after). Upstream fixed exactly this in
    mcp 2.3.0 by ignoring ``ClosedResourceError`` in the writer; until
    the SDK is upgraded (a major version, 1.30.0 -> 2.3.0, needs its own
    evaluation), drop only that one record and keep every other error
    from the writer visible.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            record.getMessage() == "Error in standalone SSE writer"
            and record.exc_info
            and issubclass(record.exc_info[0], anyio.ClosedResourceError)
        ):
            return False
        return True


def _install_sse_teardown_filter() -> None:
    """Attach the teardown filter to the SDK's streamable-HTTP logger.

    Installed at import time (not in ``main()``) so it is active both
    for the CLI server and for servers embedded by tests.
    """
    logger = logging.getLogger("mcp.server.streamable_http")
    if not any(isinstance(f, _SuppressClosedSseWriter)
               for f in logger.filters):
        logger.addFilter(_SuppressClosedSseWriter())


_install_sse_teardown_filter()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock hotel MCP server")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("MOCK_HOTEL_MCP_PORT", DEFAULT_PORT)))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--allow-host", action="append", default=[],
                        help="Extra Host header value allowed by the MCP transport"
                        " security (repeatable, e.g. host.docker.internal:*) when the"
                        " server is reached from a container instead of localhost")
    parser.add_argument("--db", default=None,
                        help="SQLite file (default: MOCK_HOTEL_MCP_DB"
                        " or mock_hotel.db in the working directory)")
    parser.add_argument("--reseed", action="store_true",
                        help="wipe the database and re-seed it, then start")
    parser.add_argument("--dump", action="store_true",
                        help="print the whole store as JSON and exit")
    args = parser.parse_args()

    open_store(args.db)
    if args.reseed:
        reset_store()
    if args.dump:
        print(json.dumps(dump_store(), ensure_ascii=False, indent=2))
        return
    import uvicorn

    if args.allow_host:
        # FastMCP enables DNS-rebinding protection for loopback servers and
        # only allows 127.0.0.1/localhost/[::1] Host headers. A container
        # reaches the host mock as host.docker.internal, so widen the list
        # explicitly instead of disabling the check entirely.
        from mcp.server.transport_security import TransportSecuritySettings

        security = mcp.settings.transport_security or TransportSecuritySettings()
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=security.enable_dns_rebinding_protection,
            allowed_hosts=[*security.allowed_hosts, *args.allow_host],
            allowed_origins=list(security.allowed_origins),
        )

    print(f"Mock hotel MCP server on http://{args.host}:{args.port}/mcp"
          f" (db: {store_path()})",
          flush=True)
    uvicorn.run(mcp.streamable_http_app(), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
