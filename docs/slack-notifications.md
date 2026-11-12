# Slack notifications — chỉ báo khi lỗi + hành động admin

Workspace: **agentic-runtime** (`https://agentic-runtime.slack.com/`),
kênh: **#agent-activity**. Phạm vi do tran chốt ngày 2026-11-11: chỉ
báo khi execution **FAILED** hoặc khi mission `system_admin` (agent
`admin_bot`) thực sự làm gì đó (output có `actions_taken` không rỗng).
Không báo mọi execution.

## 1. Tạo Incoming Webhook cho #agent-activity (tran làm một lần)

1. Mở workspace `agentic-runtime` trên Slack, vào
   **Settings & administration → Manage apps** (hoặc mở trực tiếp
   https://agentic-runtime.slack.com/apps).
2. Tìm app **Incoming Webhooks** của Slack, bấm **Add to Slack**
   (nếu chưa cài thì **Add** trước).
3. Ở bước chọn kênh, chọn **#agent-activity** rồi **Add Incoming
   Webhooks Integration**.
4. Copy **Webhook URL** (dạng
   `https://hooks.slack.com/services/T…/B…/…`) — đây là secret, chỉ
   dán vào file `.env` trên máy chạy worker, **không commit, không
   gửi qua chat**.

Webhook này gắn cố định với kênh đã chọn lúc tạo; đổi kênh thì tạo
webhook mới (hoặc sửa trong cấu hình app Incoming Webhooks).

## 2. Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `SLACK_WEBHOOK_URL` | `""` (trống) | Webhook URL ở bước 1. Trống = không gửi gì cả. |
| `SLACK_CHANNEL` | `#agent-activity` | Chỉ để ghi rõ kênh đích trong config. Incoming Webhook post về kênh đã cấu hình lúc tạo webhook, không đọc giá trị này để chọn kênh. |
| `SLACK_NOTIFICATIONS_ENABLED` | `true` | Công tắc chính: `false` tắt hết thông báo dù đã có webhook. |

Thêm vào `.env` (xem `.env.example`):

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T…/B…/…
SLACK_CHANNEL=#agent-activity
SLACK_NOTIFICATIONS_ENABLED=true
```

Sau đó restart worker (Celery) để nhận env mới. **Cho tới khi
`SLACK_WEBHOOK_URL` được set, không có thông báo nào được gửi** — code
no-op sạch, execution chạy bình thường.

## 3. Phạm vi — cái gì báo, cái gì không

| Trường hợp | Báo? |
|---|---|
| Execution `FAILED` (bất kỳ mission nào, kể cả hết retry) | ✅ Có |
| `system_admin` `SUCCEEDED` và `actions_taken` không rỗng | ✅ Có |
| `system_admin` `SUCCEEDED` nhưng chỉ đọc (`actions_taken` rỗng / thiếu) | ❌ Không |
| `fault_analysis` (hay mission khác) `SUCCEEDED` | ❌ Không |
| `CANCELLED` / execution không tìm thấy / đã xong rồi | ❌ Không |

## 4. Nội dung tin nhắn

Một tin nhắn text ngắn, tiếng Việt, không dùng mrkdwn:

- Lỗi: dòng trạng thái + mission + execution id + lỗi (cắt 500 ký tự).
- Hành động admin: dòng trạng thái + mission + execution id + danh
  sách hành động (tối đa 5 mục, mỗi mục cắt 200 ký tự).

Không gửi secret, input hay toàn bộ output — chỉ lỗi / tên hành động.

## 5. Cách nối dây (code)

- `runtime/notifications/slack.py` — hàm thuần, không import app code:
  `should_notify()` (phạm vi), `build_message()` (payload `{"text": …}`),
  `post_notification()` (POST JSON bằng `urllib`, stdlib, timeout 5s),
  `notify_execution()` (gộp 3 bước). Mọi hàm không bao giờ raise.
- `app/worker/tasks.py` — `run_execution` gọi `notify_execution`
  best-effort **sau** `db.commit()` ở đường success, và sau khi persist
  `FAILED` ở đường hết retry. Không gọi ở các early return
  (not-found, cancelled, already-finished). Mỗi lần gọi bọc
  `try/except` chỉ log — thông báo lỗi không bao giờ làm hỏng hay
  chặn execution.
- Test: `tests/test_slack_notifications.py` (ma trận phạm vi, nội dung
  tin nhắn, sender lỗi → `False` không raise, không dùng mạng thật).

## 6. Kiểm tra sau khi set webhook

1. Dispatch một execution `system_admin` chỉ đọc (ví dụ
   `get_executions`) → không có tin nhắn.
2. Dispatch một execution `system_admin` có ghi (ví dụ tạo mission) →
   có tin nhắn liệt kê hành động.
3. Gây một execution lỗi (mission không tồn tại / input sai) → có tin
   nhắn `FAILED` kèm lỗi.

Nếu không thấy tin nhắn: kiểm tra `SLACK_WEBHOOK_URL` đã set cho
worker chưa, worker đã restart chưa, và log worker có dòng
`Slack notification failed` không.
