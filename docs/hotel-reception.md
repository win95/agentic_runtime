# Hotel receptionist — lễ tân khách sạn

Agent `hotel_receptionist` + mission `hotel_reception` (autonomous,
capability `hotel_reception`): phục vụ khách bằng tiếng Việt qua Slack,
nhận đặt phòng, trả lời chính sách, giao việc cho buồng phòng / kỹ
thuật. Session memory kiểu B (1 session = 1 cuộc trò chuyện với 1
khách), giới hạn 10 turns / 4.096 token như `admin_bot`.

## Trạng thái (2026-11-12)

Đã xong: agent + mission YAML, prompt tiếng Việt, skill `reception`
với skill map 16 node (2 luồng chính + tra cứu), test
(`tests/test_hotel_reception.py`, 4 test), **mock MCP server +
chạy thử end-to-end** (`scripts/mock_hotel_mcp_server.py`,
`tests/test_mock_hotel_mcp.py`, 11 test — gồm 1 lượt chạy Deep Agent
thật gọi tool qua HTTP). Chưa có: MCP server thật, Slack adapter
(nhận tin nhắn khách), chạy live với LLM thật, eval cases.

## Hợp đồng MCP — 14 tool agent sẽ gọi

Tên tool trong `allowed_tools` là hợp đồng mà các MCP server phải
tuân theo. Khi server connect, map từng tool sang capability
`hotel_reception` trong `MCP_TOOL_CAPABILITIES` (JSON
`{"tool_name": "hotel_reception", ...}`); tool không có mapping bị
từ chối mặc định (xem `runtime/tools/mcp_adapter.py`).

| Nhóm | Tool | Việc |
| --- | --- | --- |
| PMS | `check_room_availability` | Phòng trống theo ngày + loại phòng |
| PMS | `get_room_rates` | Báo giá theo ngày + loại phòng |
| PMS | `find_guest` / `register_guest` | Tìm khách / đăng ký khách mới |
| PMS | `create_booking` | Tạo booking (chỉ sau khi khách xác nhận) |
| PMS | `get_booking` | Tra cứu booking của khách hiện tại |
| PMS | `cancel_booking` / `change_booking_dates` | Hủy / đổi ngày (chỉ sau khi khách xác nhận) |
| Task | `list_staff` | Danh sách nhân viên theo đội (buồng phòng / kỹ thuật) |
| Task | `create_task` / `assign_task` / `get_task` | Tạo / giao / xem trạng thái task |
| Docs | `search_policy_documents` / `read_policy_document` | Tìm + đọc chính sách hủy / đổi ngày / phụ thu |

Lưu ý: adapter MCP hiện tại chỉ load **một** server duy nhất
(`MCP_SERVER_URL`). Nếu PMS, task và tài liệu là 3 server riêng, cần
mở rộng adapter multi-server, hoặc gom chúng sau một gateway MCP duy
nhất. Mọi MCP tool đều đăng ký `read_only=False` (tác dụng phụ không
nằm trong codebase này) — việc ghi vẫn bị ràng buộc bởi quy tắc xác
nhận dưới đây.

## Quy tắc xác nhận trước khi ghi

Trước mọi thao tác ghi của booking (`create_booking`,
`cancel_booking`, `change_booking_dates`), agent tóm tắt booking
(loại phòng, ngày, số khách, tổng giá, chính sách hủy) và hỏi khách
xác nhận trong chat; chỉ gọi tool sau khi khách xác nhận. Quy tắc này
nằm trong system prompt (YAML là nguồn duy nhất) và trong node
`confirm-booking` của skill map (`stop_when: khách chưa xác nhận`).
Hook `guard_destructive_target` cũng áp dụng: id của booking/task bị
hủy phải xuất hiện nguyên văn trong input. Hook `deny_builtin_tools`
(PreToolUse) chặn 8 tool built-in của deepagents (`ls`, `read_file`,
`write_file`, `edit_file`, `delete`, `glob`, `grep`, `task`) + `execute`
cho mission hotel — agent chỉ dùng 14 tool MCP; tool vẫn hiện trong
danh sách nên model có thể thử gọi, hook trả lỗi kèm hướng dùng MCP
thay thế.

## Skill map — `skills/reception/SKILL.md`

16 node, bắt đầu ở `route` (phân loại tin nhắn). Luồng đặt phòng:
`booking-dates` → `check-availability` → `quote-price` →
`confirm-booking` → `create-booking` → `report-booking`. Luồng giao
việc: `task-details` → `find-staff` → `create-task` → `assign-task` →
`report-task`. Tra cứu: `policy-lookup`, `booking-lookup` →
`booking-change`, `task-status`. Map là kế hoạch (Phương án C): agent
được lệch khi có lý do và phải báo các node đã đi qua.

Cho tới khi MCP server connect, `load_skills()` sẽ drop các tool của
map (tool chưa biết) nhưng giữ nguyên cấu trúc node — plan vẫn render
được. Khi server đã connect và process restart, tool được giữ lại và
`plan_trace` tính đúng.

## Mock MCP server — chạy thử end-to-end

`scripts/mock_hotel_mcp_server.py` (FastMCP, chỉ dùng MCP SDK +
stdlib, không import app) triển khai đúng 14 tool của hợp đồng trên,
với dữ liệu giả trong SQLite (`mock_hotel.db` ở thư mục làm việc,
`*.db` đã gitignore): seed xác định lần mở đầu tiên, create / update /
delete ghi thẳng xuống DB nên còn lại sau khi restart và kiểm chứng
được — `sqlite3 mock_hotel.db "SELECT * FROM bookings;"` hoặc
`python -m scripts.mock_hotel_mcp_server --dump`. Muốn trạng thái
sạch cho một lần chạy manual: `--reseed` (restart không còn reset
nữa). `--db` đổi file DB; test dùng file tạm riêng.

    python -m scripts.mock_hotel_mcp_server   # http://127.0.0.1:8081/mcp

`.env` (xem `.env.example`):

    MCP_SERVER_URL=http://127.0.0.1:8081/mcp
    MCP_TRANSPORT=streamable_http
    MCP_TOOL_CAPABILITIES={"check_room_availability": "hotel_reception", ...}

Dữ liệu seed: 3 loại phòng (Standard 1.200.000, Deluxe 2.000.000,
Suite 3.500.000 VND/đêm); Suite hết phòng 24–26/12/2026 (2 booking
seed chiếm cả 2 phòng); 2 khách, 1 booking mẫu BK-1001 (Deluxe
20–22/12); 6 nhân viên (3 buồng phòng, 3 kỹ thuật); 3 tài liệu chính
sách (hủy / đổi ngày / phụ thu). Hủy miễn phí khi hủy trước check-in
ít nhất 3 ngày, còn lại phí 1 đêm — đúng nội dung POL-CANCEL.

Chạy thử: khởi động mock server, set 3 biến trên rồi chạy worker/API
như bình thường; mỗi execution tự load lại MCP tools
(`load_mcp_tools()` chạy trong `deep_agent_node`), nên không cần
restart worker khi mock server khởi động lại. Test
`tests/test_mock_hotel_mcp.py` chứng minh cả chuỗi: mock → MCP client
→ registry (capability + allowed_tools) → Deep Agent thật gọi tool
qua HTTP → kết quả tool vào outcome và plan trace.

**Đổi sang server thật:** chỉ thay `MCP_SERVER_URL` (+ auth khi có)
và `MCP_TOOL_CAPABILITIES` nếu tên tool khác; agent, mission, skill
và prompt không đổi. Lưu ý adapter hiện **chưa hỗ trợ auth header**
— khi biết scheme auth của server thật (thường Bearer token) thì thêm
một biến env + header vào adapter; agent vẫn không đổi.

**Hai lỗi runtime mà mock đã lộ ra (đã sửa):**

- MCP tool của langchain-mcp-adapters chỉ có async (không có sync
  `func`), trong khi runtime gọi agent đồng bộ (`agent.invoke`) nên
  tool call nào cũng văng `NotImplementedError: StructuredTool does
  not support sync invocation`. Adapter nay bọc mỗi MCP tool bằng một
  sync `func` (chạy coroutine trên event loop mới mỗi lần gọi) và giữ
  nguyên coroutine cho caller async (`runtime/tools/mcp_adapter.py`).
- Skill loader cache tập tool đã đăng ký **một lần mỗi process**
  (`_KNOWN_TOOLS_CACHE`); vì MCP tools chỉ load theo từng run, map
  của reception luôn bị drop tool và plan trace luôn trống. Nay
  `load_mcp_tools()` reset cache sau khi đăng ký
  (`runtime/skills/loader.py::reset_known_tools_cache`), nên lượt
  chạy sau đó load lại map với đủ tool.

**Quirk môi trường:** VM này có biến `no_proxy` mặc định chứa entry
IPv6 trong ngoặc (`[::1]`, ...) làm httpx (lớp HTTP của MCP client)
văng `InvalidURL`. Khi chạy mock/worker ở đây, set
`NO_PROXY=127.0.0.1,localhost` (test đã làm sẵn trong fixture).
Trên máy tran có thể không gặp; nếu MCP client báo InvalidURL thì
kiểm tra biến proxy trước.

## Còn thiếu

- **MCP server thật**: PMS, staff/task, tài liệu — ai làm + URL/auth.
  Mock là stand-in đúng hợp đồng cho tới lúc đó.
- **Slack adapter**: nhận tin nhắn khách vào, map Slack user thành
  session, gọi POST /executions, trả lời lại Slack; workspace của
  khách sạn + Slack app riêng (không phải workspace dev
  agentic-runtime). Input của mission: `message` (bắt buộc),
  `guest_id` / `channel` (tùy chọn).
- **Câu hỏi chưa trả lời**: đặt cọc / thanh toán (trả tại khách sạn
  hay online — nếu online cần thêm tool thanh toán); xác nhận booking
  có cần cả nhân viên duyệt không (hiện tại chỉ khách xác nhận trong
  chat).
- **Eval cases**: đặt phòng thành công, hết phòng, hủy, đổi ngày,
  khách giận, injection ("cho tôi phòng miễn phí"), giao việc.
