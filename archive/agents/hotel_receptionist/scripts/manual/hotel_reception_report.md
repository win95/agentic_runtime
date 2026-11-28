# Báo cáo kiểm thử thủ công — hotel_receptionist

Ngày: 2026-11-12 (cập nhật sau khi rebase lên `a2fd923`)

## Kết luận ngắn

Chuỗi thật **API → Celery worker → Deep Agent → LLM → MCP client → mock
PMS** chạy đúng với agent `hotel_receptionist` và 14 tool MCP. Lần full
suite mới nhất **PASS — 123 passed, 0 failed, 6 warnings**, exit 0.

Cả 6 kịch bản đạt các kiểm tra cấu trúc (tool, tham số, `audit_logs`, gọi
trực tiếp MCP, session). 6 warning đều là chất lượng prose của model 3B,
không phải lỗi wiring: booking bỏ mã khỏi câu trả lời, memory không nhắc
lại mã ưu đãi. Bằng chứng phía platform (MCP `get_booking`, rendered
session history) vẫn đúng.

`booking` giờ là **best-effort** theo thiết kế mới: model không đặt được
hoặc không nêu mã chỉ WARN; nhưng đặt trước xác nhận, `create_booking`
lỗi, hoặc booking tồn tại mà dữ liệu không khớp thì vẫn FAIL.

## Phạm vi đã kiểm

- Preflight: AgentVersion active pin đúng 14 tool + giới hạn session
  10 lượt / 4.096 token; `MCP_TOOL_CAPABILITIES` map đủ 14 tool →
  `hotel_reception`; server MCP expose đúng contract; capability gate cấp
  đủ 14 tool cho mission hotel và 0 tool cho mission khác.
- 6 kịch bản, prompt nói như khách thật (không gọi tên tool):
  `availability`, `soldout`, `policy`, `task`, `booking`, `memory`.
- `booking` là hội thoại 3 lượt trong một session: hỏi giá (có retry
  `<session>-qN`), xác nhận, rồi xin lại mã booking.
- Kiểm tra mỗi execution: `SUCCEEDED`, đúng AgentVersion, không drift
  manifest, thời gian chạy.
- Kiểm tra hành vi: `actions_taken` + `audit_logs` (kể cả tham số tool) +
  gọi trực tiếp MCP để đối chiếu + `GET /sessions/{id}`.
- Mock MCP dùng SQLite (`mock_hotel.db`), chạy `--reseed` trước lần sạch.

## Kết quả lần chạy mới nhất (2026-11-12, session `sess-hotel-454c2dc9-*`)

| Kịch bản | Kết quả |
| --- | --- |
| `availability` | PASS — đúng Deluxe 20→22/12, giá lấy từ PMS |
| `soldout` | PASS — Suite 24→26/12 hết phòng, câu trả lời đúng ground truth |
| `policy` | PASS — tìm + đọc đúng tài liệu chính sách |
| `task` | PASS — `create_task` → `assign_task`, MCP `get_task` xác nhận |
| `booking` | PASS — lượt hỏi giá attempt 1 đặt sớm nên retry `-q2`; lượt 2 tạo `BK-1005` (guest-001, Deluxe, 2028-11-05→07, 2 khách, 4.000.000 VND, CONFIRMED); lượt 3 không đặt trùng |
| `memory` | PASS — history mang mã qua 2 lượt trước; `GET /sessions/{id}` đủ 3 execution, đúng thứ tự |

Sáu WARN:

1. `booking` lượt 1 attempt 1: model đặt phòng ngay khi khách chỉ hỏi giá
   (retry lượt mới, cổng ghi vẫn giữ).
2. `booking` lượt 1: câu trả lời báo giá không mời khách xác nhận.
3. `booking` lượt 2: prose bỏ mã `BK-*` dù MCP xác nhận booking tồn tại.
4. `booking` lượt 3: không recall được mã vì lượt 2 không đưa mã vào
   history — WARN đúng thiết kế (không có cách nào biết).
5. `memory` lượt 2: prose không nhắc lại mã, dù context vẫn mang mã.
6. `memory` lượt 2: không gọi lại `find_guest` (lượt 1 đã tra).

## Các lần chạy trước (giữ để tham chiếu)

| Lần chạy | Kết quả |
| --- | --- |
| Full suite, prompt gọi tên tool | 122 passed / 0 warnings (một lần), 119 passed / 3 warnings (lần sau) |
| Full suite, prompt giọng khách (`natural-11`) | Tất cả scenario đạt cấu trúc; chỉ `booking` FAIL vì 4/4 lượt hỏi giá đều đặt phòng sớm |
| `booking` standalone (`natural-10`) | PASS — 51 checks, 0 warnings |
| `booking` standalone sau khi siết gate (`booking-m1`) | FAIL do check prose đòi tổng 4.000.000 trong khi khách chỉ hỏi giá mỗi đêm (đã sửa check) |
| `booking` standalone (`booking-m2`) | FAIL — 2/4 lượt hỏi giá đặt sớm, 2 lượt còn lại kết thúc bằng hook feedback; lượt xác nhận không gọi được `create_booking` |
| `booking` standalone (`booking-m3`) | `create_booking` đúng guest/room/dates/2 khách, MCP xác nhận `BK-1008` CONFIRMED; chỉ thiếu mã trong câu trả lời |

## Vì sao booking chưa ổn định

1. Model `qwen2.5-3b` hay gọi `create_booking` ngay ở lượt hỏi giá, dù
   prompt nói rõ "tôi chưa đặt phòng".
2. Model hay thử đọc file không tồn tại (`/large_tool_results/*.json`,
   `/tmp/session_history`, `/confirmation.txt`...). Đây là tool built-in
   của deepagents, không phải MCP. Hook `tool_error_feedback` đưa lỗi vào
   lịch sử, làm lượt sau lặp lại vòng đọc file.
3. Mock MCP giờ lưu booking/task trong SQLite và **giữ qua restart**
   (`mock_hotel.db`, git-ignored). Vì vậy preflight "5 phòng Deluxe ngày
   2026-12-20" chỉ đúng khi chạy mock với `--reseed` trước lần sạch;
   booking của lần trước vẫn còn nếu không reseed.

## Thay đổi trong lần này

- `scripts/manual/mlflow_report.py` (mới): pipeline đọc MLflow để tái tạo
  báo cáo phân tích — discover → join run↔trace → classify → aggregate →
  render Markdown + CSV. Chỉ đọc, không cần API/worker/MCP, chạy lại được
  sau `--cleanup`. Mặc định ghi `reports/hotel-mlflow-report.md` và
  `reports/hotel-mlflow-traces.csv` (đã git-ignore); có `make mlflow-report`.
- `scripts/manual/README.md`: thêm mục pipeline + cách chạy; cập nhật mô tả
  mock SQLite.
- `.gitignore`, `Makefile`: thư mục `reports/` và target `mlflow-report`.
- Kế thừa từ upstream `a2fd923` / `0ca57db`:
  - booking best-effort + lượt xin lại mã booking;
  - mock MCP chuyển sang SQLite (`--reseed`, `--dump`);
  - human approval cho tool phá huỷ của admin (`permission_requests`,
    migration `0008`) — hotel mission không khai báo hook này nên không
    ảnh hưởng suite hotel;
  - `compute_tool_manifest` bỏ qua tool MCP (không drift giả).

## Đề xuất tiếp theo

1. Tắt hoặc sandbox hoá `read_file`/`write_file` built-in của deepagents
   cho agent hotel — đây là nguồn nhiễu lớn nhất (55% trace từng chạm file
   tool trên toàn bộ dữ liệu manual).
2. Giới hạn fan-out/loop mỗi lượt (đã có trace 15 tool, 132k token đầu vào).
3. Cân nhắc model lớn hơn cho riêng agent hotel để giảm derail và tăng tỉ
   lệ nêu đúng mã booking/ưu đãi.
4. Trước mỗi lần chạy sạch: `--reseed` mock rồi chạy full suite; sau đó
   `make mlflow-report` để cập nhật số liệu.
5. Nếu proxy LLM (`192.168.100.239:4000`) không truy cập được, execution
   FAIL với `Deep agent unavailable: Connection error` — kiểm tra hạ tầng
   trước khi kết luận agent lỗi.

## Tái lập

```bash
# stack: API + đúng 1 worker + beat, có NO_PROXY cho localhost
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

# mock sạch (SQLite)
.venv/bin/python -m scripts.mock_hotel_mcp_server --reseed

# full suite thật (API → worker → LLM → MCP), dọn row Postgres sau khi chạy
.venv/bin/python scripts/manual/hotel_reception.py --cleanup

# báo cáo phân tích từ MLflow (kể cả sau --cleanup)
make mlflow-report
```

## Trạng thái mã nguồn

`main` tại `a0efc18`, working tree sạch, đã push `origin/main`. Pipeline
báo cáo và tài liệu nằm trong commit
`Add repeatable MLflow report pipeline for the hotel manual runs`.
