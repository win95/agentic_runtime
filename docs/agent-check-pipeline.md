# Quy trình kiểm tra agent theo yêu cầu (agent-check pipeline)

Kiểm tra các agent đang chạy (live mission) bằng LLM thật, qua mission
graph thật, rồi ghi kết quả vào MLflow. Script:
`scripts/manual/agent_check.py` (`make agent-check`).

## KHI NÀO — chỉ khi tran yêu cầu

Chạy sau khi một agent thay đổi: mission YAML, skill, hoặc code của
agent. Và **chỉ khi tran yêu cầu kiểm tra** — không có chạy tự động
nào cả: **không cron, không hook, không CI, không auto-run**. Không
target Makefile nào khác gọi `agent-check`.

## CHẠY NHƯ THẾ NÀO

Script seed mission + agent vào SQLite in-memory (không chạm Postgres
thật), dựng mission dict từ DB row bằng `mission_to_dict` (đúng dạng
worker dùng — không dùng raw YAML), rồi chạy `run_mission` với LLM thật.
Provider / model / key lấy từ environment qua factory chuẩn của repo
(LiteLLM proxy hoặc Cloudflare, theo `LLM_PROVIDER` / `LLM_MODEL`
trong `.env`) — không hardcode provider, model, account hay token nào
trong code.

Trên máy tran (LiteLLM proxy + MLflow trong dev compose):

```bash
make agent-check
# hoặc chỉ một mission:
make agent-check MISSIONS=system_admin
```

Trên VM này (Cloudflare; token lấy từ connector của Muse, không paste
giá trị vào đây):

```bash
LLM_PROVIDER=cloudflare LLM_MODEL=@cf/google/gemma-4-26b-a4b-it \
  EMBEDDING_MODEL=@cf/qwen/qwen3-embedding-0.6b \
  CLOUDFLARE_ACCOUNT_ID=<account id> CLOUDFLARE_API_TOKEN=<token> \
  .venv/bin/python scripts/manual/agent_check.py \
  --tracking-uri sqlite:////tmp/mlruns-agent-check/mlflow.db
```

Lưu ý: MLflow 3.x trên VM này từ chối backend file (`file://...`
— filesystem store đang ở maintenance mode), nên dùng SQLite thay
cho `file:///tmp/mlruns-agent-check`.

## KIỂM TRA CÁI GÌ

Một probe cố định cho mỗi live mission (không gồm agent đã archived
— `fault_analysis` / `hotel_reception` thuộc `make eval`):

| Mission | Probe |
|---|---|
| `agent_lark_me_mission` | `{"message": "ping", ...}` — bot Lark trả lời ping |
| `system_admin` | `{"request": "Liệt kê các mission hiện có trong hệ thống."}` |
| `agent_analytis_mission` | Câu hỏi nghiên cứu thật: "Giá vàng SJC hôm nay là bao nhiêu? Tìm nguồn mới nhất trên web và tóm tắt ngắn gọn." — phải là câu nghiên cứu (skill map chỉ chọn flow research khi cần tìm nguồn mới; câu số học như "17 + 25" không vào flow nào) |

## PASS / LIMIT / FAIL

- **PASS** — run `SUCCEEDED` và không có giới hạn môi trường nào xuất hiện.
- **LIMIT** — `SUCCEEDED` hoặc `FAILED` chỉ vì một giới hạn môi trường
  đã biết (ghi trong code dưới dạng chuỗi lỗi chính xác của tool):
  - `LARK_APP_ID / LARK_APP_SECRET are not set` — VM này chưa có
    credential của app Lark riêng của agent, nên tool Lark báo "not
    configured" nhưng run vẫn `SUCCEEDED` kèm ghi chú đó.
  - `TAVILY_API_KEY is not set` — chưa có key Tavily, nên flow research
    của agent_analytis không tìm được nguồn web thật.
- **FAIL** — mọi trường hợp còn lại.

## KẾT QUẢ Ở ĐÂU

- Bảng verdict in ra stdout trước (kể cả khi MLflow lỗi thì bảng vẫn
  có; lỗi MLflow không che kết quả case — exit code vẫn theo verdict).
- MLflow experiment **`agent-check`**: một parent run cho mỗi lần kiểm
  tra (`agent-check-<YYYYMMDD-HHMMSS>`), một child run cho mỗi case
  (params: mission, case, model; metrics: success, duration_seconds,
  tool_calls từ plan_trace), artifact `report.md` (bảng mission /
  verdict / duration / errors-notes) ở cả parent và child.
- Exit code: `0` khi mọi case PASS hoặc LIMIT, `1` khi có FAIL,
  `2` khi sai tham số / không case nào chạy được.
