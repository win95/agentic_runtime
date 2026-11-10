# Hooks — vòng đời của một lần chạy agent (trích xuất từ Hooks v2)

## Vì sao không dùng thẳng Hooks v2

Hooks v2 của Deep Agents Code (CLI `dcode`) chỉ chạy qua file
`hooks.json` + shell script bên ngoài, và `create_deep_agent` không có
tham số hooks nào — hệ thống này chạy server-side bằng SDK nên không
dùng được nó trực tiếp. Cái trích xuất ở đây là **ngữ nghĩa**: cùng 12
event, cùng matcher, cùng tác dụng theo từng event (chặn tool, inject
feedback, buộc chạy tiếp, chặn prompt) — viết lại bằng Python callables
+ `langchain` `AgentMiddleware` native. Chi tiết đối chiếu với tài liệu
chính thức nằm trong trang nghiên cứu Hooks v2 đã gửi tran.

## Bảng đối chiếu: event → tương đương trong hệ thống → tác dụng lên thành phần nào

| Event (Hooks v2) | Tương đương ở đây | Tác dụng lên thành phần |
|---|---|---|
| `SessionStart` | `HookMiddleware.before_agent` | `runtime/deep_agent.py` (prompt): context thêm từ hook được inject thành system message khi agent bắt đầu. Mặc định built-in không thêm gì → không đổi hành vi. |
| `UserPromptSubmit` | Dispatch trong `deep_agent_node`, trước khi build agent | `runtime/mission_graph.py` + `runtime/security.py` (Layer 1): input vi phạm (thiếu field bắt buộc, quá 8KB, asset_id sai...) bị chặn, run FAILED với lỗi "Input blocked by hook" — không chạm tới agent, tool, hay version lookup. Đây là lưới cho execution đi qua schedule/admin dispatch (bỏ qua API router). |
| `SessionEnd` | `HookMiddleware.after_agent` + dispatch trong `app/worker/tasks.py` sau `run_mission` | `app/worker/tasks.py` (execution lifecycle): chỉ ghi log chẩn đoán (audit hook). Worker dispatch bao phủ cả run deterministic không chạy agent. |
| `PermissionRequest` | Đã định nghĩa event, **chưa nối dây** | Chưa tác dụng lên thành phần nào — cần quyết định kênh phê duyệt của tran (giống human-in-the-loop còn thiếu ở Layer 4). |
| `Notification` | Dispatch qua registry (audit hook ghi log) | Chỉ chẩn đoán. Hiện chưa có nơi nào phát Notification ngoài log audit. |
| `PreToolUse` | `HookMiddleware.wrap_tool_call` (trước khi gọi tool) | `runtime/tools/registry.py` + mọi tool (đặc biệt MCP tool, vốn `read_only=False`): `deny` trả `ToolMessage(status="error")` và tool **không chạy**. Built-in `block_dangerous_tool_call` chặn `rm -rf`, `DROP TABLE`, `DELETE FROM`, `curl ... \| sh`... trong args. |
| `PostToolUse` | `HookMiddleware.wrap_tool_call` (sau khi tool trả về) | `runtime/deep_agent.py` (context agent thấy): `feedback` được nối vào nội dung tool result. Built-in `injection_feedback` quét tín hiệu Layer 6 và nhắc agent coi output là DATA. |
| `PostToolUseFailure` | `HookMiddleware.wrap_tool_call` (tool trả error hoặc raise) | Như PostToolUse + `runtime/tools/*`: tool raise exception không còn crash run mà thành `ToolMessage(status="error")` — cùng quy tắc "error evidence is still evidence" như capability graph. Built-in `tool_error_feedback` nhắc agent báo lỗi thay vì bịa dữ liệu. |
| `PreCompact` | Đã định nghĩa event, **chưa nối dây** | Không tác dụng — hệ thống không có compaction/summarization (không có conversation memory theo quyết định kiến trúc). |
| `Stop` | `HookMiddleware.after_model` + `@hook_config(can_jump_to=["model"])` | `runtime/mission_graph.py` (deep_agent_node) + agent loop: agent muốn dừng mà chưa gọi tool nào thì bị buộc chạy thêm 1 turn (nudge HumanMessage + `jump_to="model"`), tối đa `stop_max_continues` (mặc định 3, cấu hình theo mission). Đã kiểm chứng qua graph thật với fake model (33 test). |
| `SubagentStart` / `SubagentStop` | Đã định nghĩa event, **chưa nối dây** | Không tác dụng — chưa có subagent nào (chỉ 1 deep agent). Khi thêm subagent thì nối vào đây. |

## 6 hook built-in (`runtime/hooks/builtin.py`)

| Tên | Event | Tác dụng |
|---|---|---|
| `audit` | SessionStart, SessionEnd, Notification, SubagentStart, SubagentStop | Chỉ ghi log. Không đổi hành vi. |
| `block_dangerous_tool_call` | PreToolUse | Chặn tool call có pattern phá hoại trong args. |
| `injection_feedback` | PostToolUse | Báo agent khi tool output chứa text giống chỉ thị. |
| `tool_error_feedback` | PostToolUseFailure | Nhắc agent báo lỗi thay vì bịa dữ liệu. |
| `stop_completion` | Stop | Buộc chạy tiếp khi agent dừng mà chưa gọi tool nào (cap 3). |
| `validate_input` | UserPromptSubmit | Chạy Layer 1 validation trong graph. |
| `retrieval_context` | SessionStart | Pre-retrieval: đọc `execution.retrieval` của mission, kéo dữ liệu thật của asset (faults, maintenance, documents, relations, sensors) thành một block có provenance, inject làm system message trước lượt model đầu tiên. Chỉ chạy khi mission khai báo `retrieval:` và input có `asset_id`. |

## Pre-retrieval — `retrieval_context` (mục riêng)

Khai báo trong `execution:` của YAML mission:

```yaml
execution:
  retrieval:
    sources: [faults, documents, relations]
    # max_chars không đặt ở đây: mặc định từ context budget
    # (12.288 ký tự = 3.072 retrieval token).
```

`fault_analysis.yaml` khai báo đúng như trên. Nguồn nào cũng có thể
chọn: `faults` (Postgres, tối đa 8 dòng gần nhất), `maintenance`
(Postgres, 5 dòng), `documents` (Qdrant, mặc định 5 doc, `doc_limit`
đổi được), `relations` (FalkorDB, 10 dòng), `sensors` (Postgres, tóm
tắt 1 dòng/metrics: số readings + max + ngày — không bao giờ dump
raw readings). Không khai báo `sources` thì mặc định
`[faults, documents, relations]`; `max_chars` mặc định 12.288 ký tự
(từ `runtime/context_budget.py`, 3.072 token retrieval).

Luật: store nào chết thì section đó thành một dòng `unavailable: ...`
(không crash run, không bịa dòng nào); không có record thì ghi `no
records found` để agent biết là đã tìm mà không thấy; block vừa
`max_chars` bằng cách bỏ nguyên dòng từ cuối (không cắt giữa dòng).
Block được inject qua `HookMiddleware.before_agent` nên agent thấy
ngay ở lượt model đầu tiên — khác với evidence của capability graph
(vốn chỉ nằm trong state, agent không thấy).

## Cấu hình theo mission

Khai báo trong `execution:` của YAML mission (giống `skills:`):

```yaml
execution:
  mode: autonomous
  agent_id: admin_bot
  hooks:
    - audit
    - block_dangerous_tool_call
    - injection_feedback
    - tool_error_feedback
    - stop_completion
    - validate_input
  stop_max_continues: 3   # cap của Stop hook, mặc định 3
```

Không khai báo `hooks:` thì **mọi built-in đều chạy** (mặc định an
toàn — chặn lệnh nguy hiểm và validate input là lưới an toàn, không
phải opt-in). Tên lạ trong danh sách được bỏ qua kèm warning.

`system_admin.yaml` khai báo tường minh đủ 6 hook; `fault_analysis.yaml`
không khai báo nên chạy mặc định.

## Thêm hook của riêng mình

Viết một hàm `(context) -> HookDecision | None` rồi đăng ký vào
registry của run (trong `deep_agent_node`, sau `default_registry()`):

```python
from runtime.hooks import HookDecision, HookEvent

def my_hook(context):
    if context.tool_name == "delete_mission":
        return HookDecision.deny("deletes need a second pair of eyes")
    return None

hook_registry.register(HookEvent.PRE_TOOL_USE, my_hook, name="my_hook")
```

Matcher (chỉ có ý nghĩa ở 3 event tool): `None`/`""`/`"*"` = tất cả,
tên chính xác = một tool, còn lại là regex không neo (`"Write|Edit"`).
Regex sai không bao giờ match (fail closed).

## Quan sát một run đã chạy

- `MissionState.hook_events`: một record mỗi lần dispatch (event,
  tool, các decision, message). `deep_agent_node` trả về trong state;
  `run_mission` trả về trong final state.
- `notes` của execution có một dòng tóm tắt khi hook can thiệp, ví dụ
  "Hooks intervened: 1 tool call(s) denied, 2 feedback(s) injected".
- Audit hook ghi log ở mức info cho SessionStart/SessionEnd.

## Giới hạn (nói thẳng)

- **Hook không bao giờ làm hỏng run**: handler raise exception thì
  registry log warning và bỏ qua handler đó; middleware không chạm
  DB. Hook chỉ có thể deny / feedback / buộc chạy tiếp / chặn input
  qua đúng decision của nó.
- **Không có shell script bên ngoài** như Hooks v2: handler là Python
  trong repo, không có `hooks.json`, không có timeout/process riêng.
  Muốn chạy lệnh ngoài thì tự viết handler gọi subprocess (chưa có
  built-in nào làm vậy).
- **PermissionRequest, PreCompact, SubagentStart/Stop** đã định nghĩa
  event nhưng chưa có dispatch site (xem bảng trên).
- **Stop hook chỉ chạy trong agent**: mission deterministic không chạy
  agent nên không có Stop hook — đó là đúng thiết kế (không có gì để
  "dừng sớm" ở đó).
- **Chưa kiểm chứng live**: jump loop đã kiểm chứng qua graph thật
  với fake model; chưa chạy với LLM thật qua proxy (cần 3 giá trị
  LiteLLM của tran).
