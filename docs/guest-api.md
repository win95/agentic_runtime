# Guest API (demo) — public chat with the hotel receptionist

Khách đăng ký/đăng nhập qua user service, lấy JWT rồi chat với agent
qua API public — không qua Slack. Demo, chưa phải production cứng
(xem cuối file).

## Luồng

```
đăng ký → đăng nhập (lấy access token 24h)
  → POST /guest/chat {"message": "..."} (Bearer token)
  → 202 {"execution_id", "session_id", "status": "QUEUED"}
  → poll GET /executions/{execution_id}/result (cùng Bearer token)
  → {"output": {"answer": "...", ...}}
```

Tin tiếp theo gửi cùng `session_id` để giữ mạch hội thoại. `GET
/guest/sessions` liệt kê các session của chính caller.

## Ví dụ

```bash
# Đăng ký + đăng nhập (user service, :8001)
curl -s -X POST localhost:8001/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email": "guest@example.com", "password": "secret123"}'
TOKEN=$(curl -s -X POST localhost:8001/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email": "guest@example.com", "password": "secret123"}' \
  | python3 -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')

# Chat (API platform, :8000) — mock MCP phải đang chạy
curl -s -X POST localhost:8000/guest/chat \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message": "Tôi muốn đặt phòng Deluxe từ 20/12 đến 22/12"}'

# Poll kết quả (thay EXEC_ID bằng execution_id trả về)
curl -s localhost:8000/executions/EXEC_ID/result \
  -H "Authorization: Bearer $TOKEN"
```

## Luật

- **Chỉ JWT, không API key.** Service key không bao giờ thay được
  khách — nếu không, một caller đọc được session của mọi khách.
- **Mission khóa cứng `hotel_reception`.** Khách không gọi được
  mission khác (`system_admin` cũng không). `POST /executions`
  không public.
- **Session gắn user.** Session của khách khác thì 404 (không bao
  giờ 403 — không lộ sự tồn tại). Tạo execution với `session_id`
  của người khác thì 422. Service-key call không bị ràng buộc (đường
  admin nội bộ giữ nguyên).
- **Agent biết user là ai.** `user_id` (JWT sub) đi từ execution row
  qua worker vào state, rồi thành một dòng trong system prompt
  (`Current user: ...`) và vào `guest_id`/`channel` của input —
  cả hai đều do server đặt từ JWT, khách không tự khai được.
- **Rate limit 20 tin/phút/khách** (Redis, shared store — xem
  `app/core/rate_limit.py`). Tin nhắn tối đa 4.000 ký tự.
- **History chỉ load turn của chính user** (cộng các turn chưa có
  chủ từ trước thời user service). Turn của khách khác không bao giờ
  vào prompt.

## Demo-only (chưa làm cho production)

- Không streaming/SSE — client poll kết quả. Worker + LiteLLM proxy
  + mock MCP phải chạy thì chat mới có trả lời thật.
- Chưa có trần chi phí LLM theo user, chưa có refresh token.
- PMS thật chưa nối — agent đang nói chuyện với mock
  (`scripts/mock_hotel_mcp_server.py`).
