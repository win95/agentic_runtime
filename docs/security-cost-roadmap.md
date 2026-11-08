# Lộ trình bảo mật & chi phí vận hành — Agentic Runtime

Mục tiêu: đưa sản phẩm từ trạng thái "chạy được cho 1 người" lên 50 người
dùng thật, mà không lộ dữ liệu và không có hóa đơn LLM bất ngờ.

Trạng thái xuất phát (kiểm chứng 2026-11-08 trên code ở `main`):
- Auth chỉ là 1 API key chung; để trống là tắt auth hoàn toàn.
- Hạ tầng publish hết cổng với mật khẩu mặc định (Postgres `postgres/postgres`,
  MinIO `minioadmin/minioadmin`, Redis và Qdrant không mật khẩu).
- Không TLS, không rate limit, không giới hạn request.
- Không ghi token usage, không hạn mức, không trần chi phí LLM.
- Container chạy root, image không pin version, không lockfile, không quét
  lỗ hổng dependency.

## Bước lớn 1 — Khóa hạ tầng lại

Mục tiêu: server có deploy lên cũng không ai sờ được vào ruột.

- Chỉ mở cổng 443 qua reverse proxy có TLS (Caddy hoặc Nginx) trỏ vào API;
  toàn bộ Postgres, Redis, Qdrant, FalkorDB, MinIO, MLflow chỉ nói chuyện
  trong mạng docker nội bộ, không publish ra ngoài nữa.
- Bắt buộc mật khẩu thật qua `.env`; chương trình tự dừng nếu còn dùng mật
  khẩu mặc định (Postgres, Redis, Qdrant API key, MinIO).
- Container không chạy root, pin version image, đặt giới hạn RAM/CPU từng
  service.
- Backup Postgres + MinIO theo lịch, và có bài kiểm tra khôi phục.

Xong khi: quét cổng từ ngoài chỉ thấy 443, đăng nhập bằng mật khẩu mặc định
thất bại, backup khôi phục được.

## Bước lớn 2 — Tài khoản từng người + tách dữ liệu

Mục tiêu: 50 người dùng mà không ai thấy dữ liệu của ai.

- Chọn phương án đăng nhập (**cần quyết**): tự quản user + JWT trong
  Postgres, hay đăng nhập qua OAuth2/OIDC (Google...).
- Thêm bảng users/tenants; mọi mission, execution, result gắn `tenant_id`;
  mọi truy vấn đều lọc theo tenant.
- Bỏ key chung toàn quyền; key cũ (nếu giữ) chỉ còn là service key nội bộ.
- Rate limit, giới hạn kích thước request, chặn trần tham số `limit`; ghi
  audit log ai làm gì.

Xong khi: chứng minh được user A gọi API cũng không đọc được execution của
user B.

## Bước lớn 3 — Khóa chi phí LLM

Mục tiêu: biết chính xác tiền đi đâu, và không bao giờ có hóa đơn bất ngờ.

- Ghi token từng lần gọi LLM (qua callback của LangChain) vào DB, gắn với
  execution và user.
- Đặt trần: tối đa token/số bước mỗi execution, hạn mức theo user theo
  ngày/tháng; vượt là tự dừng.
- Trang xem chi phí theo user/execution + cảnh báo khi chạm ngưỡng (**cần
  quyết** ngưỡng bao nhiêu).
- Sửa luôn lỗ hổng tool MCP bypass kiểm tra quyền (đưa qua registry như
  tool thường).

Xong khi: trả lời được 3 con số — $/execution, $/user/tháng, và một
execution chạy loạn tự dừng ở trần.

## Bước lớn 4 — Phòng thủ prompt injection

Mục tiêu: dữ liệu bẩn không lái được agent.

Chi tiết phân tích trong `docs/prompt-injection-defense.md`. Tóm tắt:

- Tách rạch ròi instruction và dữ liệu trong prompt (delimiter + system
  prompt nêu rõ thứ bậc: dữ liệu trong tool output không bao giờ là lệnh).
- Giới hạn kích thước input và output của tool; validate input ở API.
- Tool MCP phải qua registry check như tool thường; tool ghi dữ liệu phải
  có xác nhận của người trước khi chạy.
- Validate kết quả: root cause phải có căn cứ trong evidence đã thu thập.
- Bộ test injection mẫu chạy trong CI (`make eval` mở rộng).

Xong khi: các câu injection thử nghiệm không lái được agent làm việc ngoài
mission.

## Bước lớn 5 — Vận hành và giám sát

Mục tiêu: hệ thống lỗi thì mình biết trước user.

- Health/metrics endpoint, log tập trung, cảnh báo khi lỗi hoặc chi phí
  tăng đột biến.
- Quy trình deploy lặp lại được, CI chạy test + quét lỗ hổng dependency,
  lockfile cho bản build.

Xong khi: deploy từ đầu tới cuối làm theo một hướng dẫn duy nhất, lỗi hệ
thống có cảnh báo chứ không im lặng.

## Thứ tự làm

1 → 2 → 3, bước 4 làm song song với 2–3, bước 5 rải đều suốt quá trình.

## Quyết định còn chờ

- Phương án đăng nhập ở bước 2: JWT tự quản hay OAuth2/OIDC.
- Host ở đâu (VPS sẵn có hay cloud), ngân sách hạ tầng.
- Trần chi phí LLM mỗi user (ngày/tháng) là bao nhiêu.
