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

> **Information hiding (kiến trúc Phase 2):** `search_policy_documents`
> chỉ trả mã + tiêu đề + độ liên quan, không trả nội dung — mọi fact
> (3 ngày, 1 đêm, mức phí) chỉ có trong `read_policy_document`. Nhờ đó
> bước đọc là yêu cầu thật, không phải nghi thức: agent chỉ đọc kết quả
> tìm kiếm thì không thể trả lời câu hỏi chính sách. Test
> `test_policy_search_returns_metadata_only` khóa invariant này.

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
hủy phải xuất hiện nguyên văn trong input. Ba lớp chặn tool built-in
cho mission hotel (Part 1): (b) Mission Graph gắn
`_ToolExclusionMiddleware` của deepagents cho `hotel_reception`
(xem `runtime/deep_agent.py::builtin_tools_exclusion`) — 9 tool
built-in (`ls`, `read_file`, `write_file`, `edit_file`, `delete`,
`glob`, `grep`, `task`, `execute`) bị ẩn khỏi danh sách tool model
nhìn thấy và bị từ chối nếu gọi; fault giữ nguyên vì sandbox cần
`execute`. Hook `deny_builtin_tools`
(PreToolUse) là lớp chặn dự phòng nếu built-in bằng cách nào đó vẫn
tới được model. Skill `reception` cũng cấm nêu tên 9 tool đó (Rules
+ What NOT to Do): lễ tân không có quyền truy cập file hệ thống, mọi
việc chỉ qua 14 tool MCP.

## Skill map — `skills/reception/SKILL.md`

16 node, bắt đầu ở `route` (phân loại tin nhắn). Luồng đặt phòng:
`booking-dates` → `check-availability` → `quote-price` →
`confirm-booking` → `create-booking` → `report-booking`. Luồng giao
việc: `task-details` → `find-staff` → `create-task` → `assign-task` →
`report-task`. Tra cứu: `policy-lookup`, `booking-lookup` →
`booking-change`, `task-status`. Map từng là kế hoạch (Phương án C):
agent được lệch khi có lý do và phải báo các node đã đi qua. Từ Phase
3a, map của hotel được **ép buộc ở runtime bằng plan gate** (mục dưới)
— văn bản skill và plan render đã đổi theo: thứ tự node bị từ chối
khi gọi tool ngoài lượt, không còn là lời khuyên.

Cho tới khi MCP server connect, `load_skills()` sẽ drop các tool của
map (tool chưa biết) nhưng giữ nguyên cấu trúc node — plan vẫn render
được. Khi server đã connect và process restart, tool được giữ lại và
`plan_trace` tính đúng.

## Skill subgraph — biên dịch map thành đồ thị (Phase 3)

`runtime/skill_subgraph.py` biên dịch map của hotel thành một LangGraph
con, chạy thay cho một agent tự do duy nhất (`deep_agent_node` chọn
đường này khi mission thuộc `SUBGRAPH_PILOT_MISSIONS` và có skill map;
hiện chỉ có `hotel_reception`).

- **Một node map = một agent riêng.** Node có tool chạy một Deep Agent
  chỉ có tool của node đó; node không có tool (`route`, `booking-dates`,
  `confirm-booking`, `report-booking`, `task-details`) chạy một model
  call thuần túy. Gọi tool ngoài thứ tự không phải bị từ chối — mà
  không thể xảy ra, vì agent của node đó không có tool đó.
- **Router là code, không phải model.** Sau mỗi node, router chọn node
  tiếp theo từ bảng chuyển đổi của map: dòng `NEXT: <node-id>` do node
  khai báo, kiểm chứng trong tập cho phép (đích rẽ nhánh của node hiện
  tại + node tiếp theo theo thứ tự); khai báo sai hoặc thiếu thì về node
  tiếp theo theo thứ tự. Mỗi lựa chọn trong instruction kèm điều kiện
  `when` của nhánh đó, để model quyết định bằng điều kiện chứ không đoán
  từ id. `NEXT: END` chỉ được chấp nhận ở node kết thúc flow
  (`terminal: true` trong map: `route`, `report-booking`, `report-task`,
  `policy-lookup`, `booking-lookup`, `booking-change`, `task-status`); ở
  node khác thì END bị bỏ qua và đi tiếp node sau — model tưởng request
  đã xong giữa flow không được kết thúc run. `NEXT: WAIT` dừng turn ở
  bất kỳ node nào khi cần thêm thông tin từ khách. Nhánh rẽ của
  node `route` luôn mở (phân loại lại).
- **Luật resume (turn xác nhận):** từ node `route`, nhảy thẳng tới
  `create-booking` chỉ được phép khi session đã có turn trước — turn
  khách xác nhận tiếp nối turn trước đã dừng ở `confirm-booking`. Map
  không có nhánh này, nên gate (và mọi run tự do) vẫn từ chối
  `create_booking` gọi đầu tiên. Chỉ dẫn của node `route` nêu luật;
  router là chốt chặn.
- **Chống lặp:** tối đa 12 node mỗi run (`MAX_NODE_STEPS`); quá thì
  run kết thúc với lỗi chẩn đoán thay vì xoay vòng (map cho phép lặp
  theo thiết kế: hết phòng → chọn ngày khác → kiểm tra lại). Trong mỗi
  node còn có budget riêng: tối đa 5 tool call mỗi lần chạy node
  (`MAX_TOOL_CALLS_PER_NODE`) — cap 12 node đếm số lần chạy node chứ
  không đếm call trong node, nên model lặp một call lỗi (2026-11-13:
  `get_booking` với mã sai, lặp tới recursion limit) không bao giờ tới
  được `NEXT:`. Gọi lặp y hệt call vừa lỗi bị từ chối kèm hướng thay
  thế; quá budget thì node dừng với lỗi chẩn đoán và turn kết thúc.
  Node `booking-lookup` còn có luật fallback trong skill: get_booking
  báo not-found thì không gọi lại cùng mã mà dùng `find_guest` lấy mã
  đúng, và không bao giờ tự bịa mã booking (mã thật dạng `BK-####`).
- **Kết quả giống hệt agent tự do:** `run_skill_subgraph` trả cùng
  shape outcome (`summary` / `evidence` / `actions_taken` / `errors` /
  `plan_trace`, xem `outcome_from_messages` trong
  `runtime/deep_agent.py`) cộng thêm `visited_nodes` (đường đi đầy đủ
  của đồ thị — `plan_trace` chỉ thấy node có tool call).
- **State là TypedDict** (`SubgraphState`): LangGraph chỉ giữ các key
  schema khai báo — dùng `StateGraph(dict)` trần đã làm mất lặng lẽ
  `session_history` sau node đầu tiên, khiến luật resume không bao giờ
  kích hoạt (lỗi đã bắt được bằng test).
- **Gate và subgraph không chạy cùng nhau:** khi subgraph chạy, plan
  gate không gắn vào (không còn gì để nó từ chối) và plan block không
  append vào prompt (map đã biên dịch, không còn là prose). Gate còn
  lại làm cơ chế ép cho mission chưa lên subgraph.
- **Chưa có:** trace 7 layer cho pilot (Phase 3 tiếp theo), judge LLM
  (chưa chọn model), tách eval temp 0 / production 0.5 (Phase 4).

## Plan gate — ép thứ tự map ở runtime (Phase 3a)

`runtime/plan_gate.py`: một `AgentMiddleware` xây từ `SkillMap` đã parse
của run, gắn vào danh sách middleware của `deep_agent_node` ngay sau
hook middleware và built-in exclusion. Chỉ áp dụng cho các mission chạy
agent tự do (hiện không có mission nào: hotel đã lên subgraph;
`fault_analysis` giữ map tư vấn) — cơ chế còn lại để dùng khi một
mission chưa biên dịch map.

- **Luật cho phép:** gọi tool của node hiện tại, node tool tiếp theo
  theo thứ tự, đích rẽ nhánh của node hiện tại, đích rẽ nhánh của node
  `route` (phân loại lại luôn được phép — một execution có thể phục vụ
  yêu cầu thứ hai), hoặc node đã đi qua (quay lại). Node không có tool
  được bỏ qua khi tính đích đến: rẽ về `booking-dates` thực chất là
  tới `check-availability`.
- **Từ chối:** gọi tool ngoài tập trên trả về một `ToolMessage` lỗi
  ghi rõ tool được phép gọi tiếp — agent tự sửa ở turn sau, tool
  không chạy. Quá 5 lần từ chối trong một execution, gate raise
  `PlanGateExceededError` để run fail với chẩn đoán rõ thay vì xoay
  tới hết step limit. Số lần từ chối xuất hiện trong execution notes
  (`Plan gate: N out-of-plan tool call(s) denied`).
- **Tool ngoài map** được cho qua kèm warning (fail-open): map thiếu
  tool là lỗi config, không phải lý do phá run live. Bao phủ 14 tool
  hợp đồng được test ép (`test_reception_map_covers_all_14_contract_tools`).
- **Plan render theo gate:** `render_plan(..., enforced=True)` đổi header
  khi gate gắn vào — văn bản "được lệch" + gate cứng là mâu thuẫn nên
  không giữ cả hai. `confirm-booking` vẫn là điểm dừng ở mức prompt
  (node không có tool nên gate không quan sát được việc xác nhận).
- **Chưa có:** điều kiện tiên quyết giữa các tool (ví dụ `create_task`
  bắt buộc đã `list_staff`) — allowlist đã ép đúng thứ tự đó cho luồng
  task; quyết định từng rule (dependency dữ liệu / gate / bỏ) vẫn mở.
  Subgraph biên dịch từ map đã xong (mục trên) — gate chỉ còn là cơ
  chế dự phòng cho mission chưa lên subgraph.

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
