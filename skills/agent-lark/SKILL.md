---
name: agent-lark
description: Bot Lark — hỏi–đáp trong chat đúng thread, cảnh báo sự cố (tạo group, thêm người phụ trách, gửi card), retrospective từ history rồi đổi tên group — dùng với mission `agent_lark_me_mission`.
keywords: lark bot chat group alert card retrospective cảnh báo nhóm
map:
  start: route
  nodes:
    - id: route
      goal: "Một lần mỗi lượt — chọn đúng một flow: alert khi user ra lệnh báo sự cố; answer khi user hỏi/chat; resolve khi user ra lệnh đã xử lý xong hoặc bấm nút Đã xử lý trên card. Không gọi tool ở bước này."
      tools: []
      terminal: true
      branches:
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
---

# Agent Lark

## When to Use

Khi yêu cầu tới từ Lark (DM hoặc group) và cần **trả lời trong chat**,
**báo sự cố** (tạo group + thêm người + gửi card), hoặc **xong việc**
(retrospective + đổi tên group) — dùng với mission
`agent_lark_me_mission`.

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
5. **Trả lời tóm tắt.** Câu trả lời cuối ghi rõ đã làm gì
   (`actions_taken`: chat_id, message_id của các tin đã gửi).

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
- Không gọi tool Lark ngoài 8 tool đã đăng ký; không tìm web, không
  chạy code, không xuất file (không có sandbox).
