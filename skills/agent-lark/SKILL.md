---
name: agent-lark
description: Bot Lark — hỏi–đáp trong chat đúng thread, cảnh báo sự cố (tạo group, thêm người phụ trách, gửi card), retrospective từ history rồi đổi tên group — dùng với mission `agent_lark_me_mission`.
keywords: lark bot chat group alert card retrospective cảnh báo nhóm
map:
  start: route
  nodes:
    - id: route
      goal: "Một lần mỗi lượt — chọn đúng một flow: alert khi user ra lệnh báo sự cố; answer khi user hỏi/chat; resolve khi user ra lệnh đã xử lý xong hoặc bấm nút Đã xử lý trên card; task-remind khi input có mode == remind_tasks (lần chạy định kỳ nhắc task, không phải lệnh trong chat). Không gọi tool ở bước này."
      tools: []
      terminal: true
      branches:
        - when: "input có mode == remind_tasks — lần chạy định kỳ nhắc task"
          goto: task-remind
        - when: "user ra lệnh báo sự cố (vd 'báo sự cố ... cho <team>')"
          goto: alert-create
        - when: "user hỏi hoặc chat — cần trả lời trong thread"
          goto: answer
        - when: "user ra lệnh đã xử lý xong / bấm nút Đã xử lý — cần retrospective + đổi tên group"
          goto: alert-resolve
    - id: alert-create
      goal: "Tạo group cảnh báo bằng lark_create_group (tên + mô tả vụ việc), rồi thêm người phụ trách bằng lark_add_members (team → open_id trong roster dưới; team không rõ thì hỏi lại, không đoán). Ghi chat_id của group mới — các bước sau dùng nó."
      tools: [lark_create_group, lark_add_members]
    - id: alert-send
      goal: "Gửi card cảnh báo vào group mới bằng lark_send_message (msg_type interactive, content là JSON card đã dựng bằng CardKit — không viết tay). Card có nút resolve/assign/escalate, mỗi nút có value {action, chat_id, incident_ref}."
      tools: [lark_send_message]
      requires: [lark_create_group]
      terminal: true
    - id: answer
      goal: "Trả lời đúng thread bằng lark_reply_message (message_id của tin user gửi, reply_in_thread true). Trả lời ngắn; câu hỏi mơ hồ thì hỏi lại 1 câu rồi dừng. Không @all."
      tools: [lark_reply_message]
      terminal: true
    - id: alert-resolve
      goal: "Input lấy chat_id + incident_ref từ card action hoặc lệnh trong chat — không dựa vào session memory (người bấm nút không phải người tạo cảnh báo). Đọc history bằng lark_get_history (limit 100) → tóm tắt diễn biến + kết luận → gửi tóm tắt vào group bằng lark_send_message → đổi tên group bằng lark_update_group đánh dấu resolved."
      tools: [lark_get_history, lark_send_message, lark_update_group]
      terminal: true
    - id: task-remind
      goal: "Nhắc task đang làm dở theo input của lần chạy (execution_input — đọc mới mỗi lần chạy, không hard-code): recipients / task_filter / scope, thiếu thì dùng default (xem flow task-remind dưới). Đọc tasklists → đọc tasks → lọc → gom theo người nhận → một DM mỗi chủ task (hoặc một bản tóm tắt cho đích cố định)."
      tools: [lark_list_tasklists, lark_list_tasks, lark_get_task, lark_send_message]
      terminal: true
---

# Agent Lark

## When to Use

Khi yêu cầu tới từ Lark (DM hoặc group) và cần **trả lời trong chat**,
**báo sự cố** (tạo group + thêm người + gửi card), hoặc **xong việc**
(retrospective + đổi tên group) — dùng với mission
`agent_lark_me_mission`. Khi lần chạy là định kỳ với
`mode == remind_tasks` trong input thì chạy flow **task-remind** dưới
đây thay vì chờ lệnh trong chat.

## Steps

1. **Hiểu yêu cầu.** User hỏi gì, hay ra lệnh báo sự cố / đã xử lý
   xong? Thiếu thông tin (vd team nào, sự cố gì) thì hỏi lại trước khi
   làm — đừng đoán.
2. **Báo sự cố.** Tạo group bằng `lark_create_group` (tên + mô tả vụ
   việc), thêm người phụ trách bằng `lark_add_members` (team → open_id
   trong roster dưới; team không rõ thì hỏi lại, không đoán), rồi gửi
   card cảnh báo bằng `lark_send_message` (`msg_type: interactive`,
   content là JSON card đã dựng bằng CardKit — không viết tay). Ghi
   `chat_id` của group mới vào `actions_taken`.
3. **Trả lời trong chat.** Dùng `lark_reply_message` với `message_id`
   của tin user gửi (`reply_in_thread: true`) — trả lời đúng thread.
   Trả lời ngắn; không @all.
4. **Xong việc (resolve).** Input lấy `chat_id` + `incident_ref` từ
   card action hoặc lệnh trong chat — không dựa vào session memory.
   Đọc history bằng `lark_get_history` (limit 100), tóm tắt diễn biến
   + kết luận, gửi tóm tắt vào group bằng `lark_send_message`, rồi đổi
   tên group bằng `lark_update_group` đánh dấu resolved.
5. **Nhắc task (task-remind, chỉ khi input có
   `mode == remind_tasks`).** Xem flow riêng dưới đây — đọc cấu hình
   từ `execution_input` mỗi lần chạy, không hard-code.
6. **Trả lời tóm tắt.** Câu trả lời cuối ghi rõ đã làm gì
   (`actions_taken`: chat_id, message_id của các tin đã gửi).

## Nhắc task (task-remind)

Lần chạy định kỳ nhắc task đang làm dở. Mọi cấu hình đọc từ
`execution_input` của lần chạy này — mỗi loop khai khác nhau, không
hard-code giá trị nào trong skill:

| Field trong input | Thiếu thì dùng default |
| --- | --- |
| `recipients` | `task_assignees` |
| `task_filter.status` | `todo` |
| `task_filter.due` | `overdue_or_today` |
| `scope` | `all_tasklists` |

Cap gửi tin của lần chạy nhắc việc suy ra từ context của chính lần
chạy đó (`max(1, (history_budget - reserved_history) // 256)` — xem
§3.7 của thiết kế loop): session mới nhắc được 16 tin, session cũ
có history giữ chỗ thì ít hơn; chạm cap thì agent báo đã nhắc ai /
còn ai chưa nhắc. Loop không khai gì thêm. Lần chạy tương tác giữ
cap cố định 5 của mission.

1. **Đọc tasklists** mà app đọc được bằng `lark_list_tasklists`
   (phạm vi theo `scope` trong input; default `all_tasklists` — quét
   tasklist, không cần roster).
2. **Đọc tasks** của mỗi tasklist bằng `lark_list_tasks`
   (`completed=false`). Cần chi tiết một task thì đọc thêm bằng
   `lark_get_task`.
3. **Lọc** theo `task_filter` trong input (default: `status == todo`
   và quá hạn hoặc đến hạn hôm nay). "Đang làm" = todo chưa xong;
   nếu tran dùng custom field / section riêng để đánh dấu đang làm
   thì lọc theo đó — xác nhận từ dữ liệu thật sau khi đọc được task.
4. **Gom theo người nhận** (`recipients` trong input):
   - `task_assignees`: mỗi assignee (lấy từ `members` của task) nhận
     **một** DM liệt kê tất cả task của họ — mỗi task một dòng gồm
     tóm tắt (`summary`) + hạn (`due`) + link task (`url`). Không
     gửi một DM mỗi task.
   - `fixed_user` / `fixed_chat`: gửi **một** bản tóm tắt chung tới
     đích cố định đó (user / chat trong input), liệt kê task của mọi
     người.
5. **Không dedupe liên lần chạy:** bộ lọc theo ngày nên task đến hạn
   hôm nay được nhắc hôm nay; mai vẫn chưa xong thì thành quá hạn và
   được nhắc lại — đó là nudge hằng ngày theo thiết kế, không phải
   spam.
6. **Chỉ nhắn cho người lần chạy này đọc được:** DM assignee chỉ tới
   user mà chính lần chạy này đọc được từ Task API (scope_tool chặn
   user khác); đích cố định chỉ lấy từ input. Không bịa user id,
   không nhắn cho người ngoài danh sách đó.

## Roster team → open_id (v1)

Config trong skill này — tran cung cấp. Team không có trong roster thì
hỏi lại user, không đoán open_id.

| Team | open_id |
| --- | --- |
| _(chưa có — tran cung cấp)_ | |

## Card cảnh báo

- Card dựng bằng CardKit rồi copy JSON — không viết tay.
- Mỗi nút có `value` là JSON string:
  `{"action": "resolve"|"assign"|"escalate", "chat_id": "<group chat_id>", "incident_ref": "<alert message_id>"}`.
  `incident_ref` là message_id của tin cảnh báo.
- Nút bấm được xử lý ở Step 4 của plan (card callback) — handler lấy
  `chat_id` từ context của event (cross-check với `value`), kiểm tra
  allowlist + rate limit của người bấm rồi mới enqueue execution
  resolve với input lấy từ `value`.

## Output — định dạng bắt buộc

| Field | Bắt buộc | Nội dung |
| --- | --- | --- |
| `answer` | Có | Câu trả lời text tóm tắt việc đã làm |
| `actions_taken` | Có | Danh sách hành động: tool đã gọi + chat_id / message_id trả về |

## What NOT to Do

- Không bịa chat_id, message_id, user id hay kết quả tool — mọi id lấy
  từ input hoặc kết quả tool trả về.
- Không @all.
- Chỉ đổi tên group của run này — chat trong input hoặc group do
  chính run này tạo (`lark_update_group`; scope_tool + guard chặn
  ngoài ra). Không cần bước duyệt trên platform: người dùng Lark
  không có đường nào duyệt, và guard + scope đã là ủy quyền đủ.
- Không gửi tin vào chat khác chat của run này (scope_tool chặn).
- Không dùng session memory để lấy context vụ việc khi resolve — đọc
  history bằng `lark_get_history`.
- Không gọi tool Lark ngoài 11 tool đã đăng ký (8 tool nhắn tin /
  group + 3 tool đọc task `lark_list_tasklists`, `lark_list_tasks`,
  `lark_get_task`); không tìm web, không chạy code, không xuất file
  (không có sandbox).
- Nhắc task: không hard-code người nhận, bộ lọc hay phạm vi — đọc
  `recipients` / `task_filter` / `scope` từ `execution_input` mỗi lần
  chạy (thiếu thì default: `task_assignees`, `status: todo`,
  `due: overdue_or_today`, `scope: all_tasklists`). Một DM mỗi chủ
  task, không phải một DM mỗi task. Chỉ DM assignee mà chính lần chạy
  này đọc được từ Task API — không nhắn user chưa đọc được trong run
  (scope_tool chặn).
