# Worker riêng theo agent

Trước đây mọi execution chạy chung một worker pool, một queue `celery`,
một registry tool global, một harness index toàn mission. Lỗi của agent
này hiện trong log của agent khác: ví dụ `agent_lark_me` không dùng MCP
nhưng mỗi run vẫn gọi `load_mcp_tools()` tới mock server `:8082` và log
warning khi server đó chết; một mission lỗi harness thì flood log của
mọi mission khác mỗi lần index refresh (30s).

Hướng đi: mỗi agent tách riêng queue + worker process/container + env
+ log của nó; các agent **chỉ dùng chung harness** (schema + loader),
còn kết nối thì khác nhau — đúng thực tế hệ thống vốn phát triển rời rạc
theo từng agent. Tách dần từng agent một, không tách hết cùng lúc.

## Map queue hiện tại

| Mission | Queue | Worker |
|---|---|---|
| `agent_lark_me_mission` | `lark` | `worker-lark` |
| Mọi mission khác (mặc định) | `celery` | `worker` |

Routing ở một chỗ duy nhất: `MISSION_QUEUE_MAP` trong
`app/services/execution_service.py`, hàm `queue_for_mission(mission_id)`.
Mission không có trong map → queue `celery`. Cả `POST /executions` và
resume lẫn schedule (beat → `create_execution` → `dispatch_execution`)
đều đi qua đó.

`worker-lark` (`docker-compose.yml`): cùng build `.` và env như worker
chung, **trừ** `MCP_SERVER_URL`, **cộng** `LARK_APP_ID` /
`LARK_APP_SECRET` (fail-fast `${...:?}` — worker chạy Lark tools +
failure notifier nên bắt buộc phải có) và
`HARNESS_MISSION_ALLOWLIST=agent_lark_me_mission` (chỉ load harness của
mission đó, log không còn flood lỗi của 4 mission khác).

Lưu ý về `env_file`: mọi service đều đọc `.env` qua `env_file`, nên nếu
`.env` thật có set `MCP_SERVER_URL` / `MCP_TOOL_CAPABILITIES` thì
worker-lark vẫn nhìn thấy chúng — vô hại, vì gate MCP trong
`mission_graph.py` chặn không gọi `load_mcp_tools()` với mission không
dùng MCP. Đừng cố chặn ở tầng env_file.

Vẫn dùng chung: harness (schema + loader code), Postgres / Redis /
Qdrant / Neo4j / MinIO, API, beat. Worker-lark vẫn cần đủ env store chung
vì code graph chung chạm tới chúng; chỉ tách kết nối MCP + key Lark.

## Thêm agent tiếp theo thì làm gì

1. Thêm mission vào `MISSION_QUEUE_MAP` (kèm test map → queue đúng).
2. Thêm service compose mới (copy `worker-lark`, đổi tên service, `-Q <queue>`,
   env riêng của agent đó, `HARNESS_MISSION_ALLOWLIST=<mission>`).
3. Deploy + kiểm tra log riêng của worker mới trước khi coi là xong.

## Lệnh ops

```bash
# Bật worker-lark (lần đầu: build lại image vì command/env đổi)
docker compose up -d --build worker-lark
# Worker chung restart với -Q celery tường minh
docker compose up -d worker

# Xem log riêng từng worker
docker compose logs -f worker-lark
docker compose logs -f worker

# Dừng / chạy lại worker-lark
docker compose stop worker-lark
docker compose start worker-lark
```

Kiểm tra routing đúng: dừng `worker-lark` rồi DM bot "ping" — execution
nằm ở `QUEUED` (xem `GET /executions?status=QUEUED`), không trả lời;
`start worker-lark` → run kẹt chạy tiếp. Hotel run vẫn phải xuất hiện
trong log worker chung, không phải worker-lark.

## Rủi ro đã biết

`reap_stale_executions` (chạy mỗi 300s) chỉ reap execution RUNNING quá
hạn, **không** reap QUEUED. Routing sai → execution nằm QUEUED cho tới
khi có người nhìn log. Sau mỗi lần deploy routing, kiểm tra thủ công
`GET /executions?status=QUEUED` + log worker mới. Watchdog / cảnh báo
queue-depth là việc follow-up, chưa làm trong lần tách này.
