# Agentic Runtime

Runtime cho các Agent tự hành: **FastAPI** quản lý Platform, **Celery + Redis**
dispatch execution, **LangGraph** điều phối Mission, **Deep Agents** thực thi
tự hành, **Capability Graphs / Tools** cung cấp năng lực nghiệp vụ,
**LangSmith** quan sát toàn bộ quá trình.

> Đây là **MVP chạy được thật** (v0.1.0): LLM và embedding chạy qua proxy
> LiteLLM chuẩn OpenAI (cấu hình bằng env), dữ liệu demo O21 nằm trong
> Postgres/Qdrant/FalkorDB thật, report lưu vào MinIO. Xem mục
> [Stub & giới hạn](#10-stub--giới-hạn) để biết phần nào vẫn là stub.

## 1. Tổng quan kiến trúc (6 lớp)

| Lớp | Thành phần | Vai trò |
| --- | ---------- | ------- |
| 1 | **Platform API — FastAPI** (`app/`) | Quản lý Mission, Agent, Tool, Model, Policy và Execution API. Mission là **cấu hình**, không phải graph. |
| 2 | **Execution Dispatcher — Celery + Redis** (`app/worker/`) | Nhận yêu cầu chạy Mission, đưa vào queue, worker chạy LangGraph. Tách vòng đời HTTP khỏi vòng đời Agent. |
| 3 | **Agent Runtime — LangGraph** (`runtime/mission_graph.py`) | Một Mission Graph cho mỗi nhóm workflow: understand → router → capability / deep agent → validate → END. Hỗ trợ checkpoint Postgres (tùy chọn). |
| 4 | **Autonomous Agent — Deep Agents** (`runtime/deep_agent.py`) | Lập kế hoạch, phân rã công việc, dùng subagents và tools. Nằm **bên trong** Mission Graph (Deep Agents build trên LangGraph, trả về compiled graph). |
| 5 | **Capability & Tool Layer** (`runtime/capabilities/`, `runtime/tools/`) | Capability = LangGraph subgraph tái sử dụng (vd: `fault_analysis`). Tool = hành động nhỏ Agent có thể gọi. Quyền thực tế kiểm tra ở runtime theo mission/tenant/execution. |
| 6 | **Data & Observability** (`infra/`) | PostgreSQL, Redis, Qdrant, FalkorDB, MinIO, MLflow, LangSmith (chỉ quan sát, không phải nơi lưu trạng thái). |

### Lưu trữ

| Hệ thống | Dữ liệu |
| -------- | ------- |
| PostgreSQL | Mission, Agent, Tool metadata, execution, kết quả, fault_history, maintenance_records |
| Redis | Celery queue, cache, trạng thái điều phối tạm thời |
| Qdrant | Document embeddings, vector retrieval (`technical_documents`) |
| FalkorDB | Knowledge Graph, entity relationships |
| MinIO | File, tài liệu, artifact, report (`reports` bucket) |
| MLflow | Eval runs của mission (`agentic-runtime-eval`) |
| LangSmith | Agent trace, graph execution, tool calls |

## 2. Một Execution chạy như thế nào?

Ví dụ: *“Phân tích nguyên nhân các lỗi của thang máy O21 trong tháng 10/2026.”*

1. **FastAPI tạo Execution** — `POST /executions` với `mission_id: fault_analysis`
   và input `{asset_id: "O21", start_date: "2026-10-01", end_date: "2026-10-31"}`.
   API kiểm tra quyền, validate input, tạo Execution record trong PostgreSQL
   và trả về `{execution_id, status: "QUEUED"}`. Chỉ `execution_id` được gửi
   vào Celery.
2. **Celery nhận Execution** — Worker nhận `execution_id`, load Mission Version
   và cấu hình Agent tương ứng.
3. **LangGraph khởi chạy Mission Graph** — tạo execution state, hiểu yêu cầu,
   xác định chiến lược (mode `hybrid` của mission này).
4. **Capability Graph chạy trước** — `fault_analysis` chạy các node: lịch sử lỗi
   (Postgres), bảo trì (Postgres), tìm tài liệu (RAG/Qdrant), quan hệ thiết bị
   (FalkorDB), rồi tổng hợp evidence.
5. **Deep Agent (nếu evidence chưa đủ)** — lập kế hoạch điều tra thêm, tự quyết
   định gọi tools cho đến khi đủ evidence.
6. **Validate kết quả** — kiểm tra đủ evidence, đúng output schema
   (`root_causes`, `evidence`, `recommendations`) và success criteria.
7. **Lưu kết quả** — PostgreSQL lưu kết quả và trạng thái; report Markdown lưu
   vào MinIO bucket `reports` tại `executions/<execution_id>/report.md`
   (nếu MinIO không tới được, report lưu inline trong ExecutionResult và ghi
   rõ `report_storage: "inline"`). Toàn bộ quá trình được trace trên LangSmith.

Sơ đồ Mission Graph:

```
START → Understand Mission → Router → Capability Graph → (Deep Agent) → Validate → END
                                      ↘ Deep Agent (autonomous) ↗
```

Router chọn nhánh theo `mission.execution.mode`:
`deterministic` → chỉ Capability Graph; `autonomous` → chỉ Deep Agent;
`hybrid` → Capability Graph trước, Deep Agent chỉ khi evidence chưa đủ.

## 3. Chạy MVP (Docker Compose — khuyên dùng)

### Bước 1: cấu hình LiteLLM

```bash
cp .env.example .env   # điền key thật vào .env, KHÔNG commit file này
```

Điền **4 giá trị bắt buộc** trong `.env` (tên model phải khớp tên đã đăng ký
trên proxy LiteLLM của bạn):

| Biến | Ví dụ | Ghi chú |
| ---- | ----- | ------- |
| `LLM_API_BASE` | `http://localhost:4000` | URL proxy LiteLLM của bạn |
| `LLM_API_KEY` | *(để trống trong .env.example)* | Key của proxy |
| `LLM_MODEL` | `gpt-4o-mini` | Tên model chat trên proxy |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Tên model embedding trên proxy |

`EMBEDDING_API_BASE` / `EMBEDDING_API_KEY` nếu để trống sẽ dùng chung proxy
của LLM. `LLM_TEMPERATURE` mặc định `0.2`.

### Bước 2: chạy hạ tầng + API + worker

```bash
docker compose up -d --build
```

- API: http://localhost:8000 — docs tại `/docs`
- MinIO console: http://localhost:9001
- Qdrant: http://localhost:6333
- MLflow (tùy chọn): `docker compose --profile observability up -d`

Compose có healthcheck cho postgres, redis, qdrant, falkordb, minio;
`api` và `worker` chỉ chạy khi các service hạ tầng đã healthy.

### Bước 3: seed dữ liệu demo

```bash
docker compose --profile setup run --rm seed
# hoặc chạy local: make seed
```

Lệnh này nạp (idempotent — chạy lại không bị trùng):
- Mission `fault_analysis` (từ `seeds/missions/fault_analysis.yaml`)
- PostgreSQL: 8 lỗi + 6 bản ghi bảo trì của thang máy **O21**, tháng 10/2026
- Qdrant: 6 tài liệu kỹ thuật, embed bằng model embedding đã cấu hình
  (vector dim được đo từ probe embedding, không hardcode 1536/3072)
- FalkorDB: quan hệ O21 `PART_OF` tòa nhà B1, `HAS_COMPONENT` motor /
  door sensor / controller / brake

Nếu Qdrant/FalkorDB seed bị skip, lệnh in `WARNING` kèm lý do (thường là
chưa điền key LiteLLM hoặc service chưa lên) — sửa rồi chạy lại seed.

### Bước 4: chạy thử end-to-end

```bash
make demo
```

Script `scripts/smoke_e2e.py` kiểm tra 4 biến LiteLLM, tạo một execution
`fault_analysis` cho O21 (01–31/10/2026), poll tới khi xong (tối đa 300s)
rồi in ra. **Một lần chạy thành công in ra:**

```
Execution created: <execution_id> (status QUEUED)
  ... RUNNING
  ... SUCCEEDED

Status: SUCCEEDED
Root causes: [...]
Evidence items: <số evidence>
Report location: minio://reports/executions/<execution_id>/report.md
Smoke test PASSED.
```

### Gọi thử API bằng tay

```bash
# Tạo execution (cần API_KEY nếu đã cấu hình, header X-API-Key)
curl -X POST http://localhost:8000/executions \
  -H "Content-Type: application/json" \
  -d '{"mission_id": "fault_analysis", "input": {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"}}'

# Theo dõi
curl http://localhost:8000/executions/<execution_id>
curl http://localhost:8000/executions/<execution_id>/result
```

### Chạy local không qua compose

```bash
make install           # pip install -e ".[dev]"
make run-api           # uvicorn app.main:app --reload
make run-worker        # celery -A app.worker.celery_app worker
make test              # pytest -q
make lint              # ruff check
```

### Troubleshooting

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
| ----------- | ---------------------- | ---------- |
| `make demo` báo thiếu biến env | Chưa điền `.env` | Điền đủ 4 biến `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL` |
| Execution FAILED, lỗi kết nối proxy | LiteLLM proxy chưa chạy / sai URL/key | Kiểm tra `LLM_API_BASE` có tới được proxy không, key đúng không |
| Qdrant seed báo dimension mismatch | Đổi `EMBEDDING_MODEL` sang model có dim khác sau khi đã seed | Đổi lại đúng model cũ, hoặc xóa collection `technical_documents` rồi seed lại |
| Qdrant trống / retrieval trả lỗi | Chưa seed, hoặc seed bị skip | Chạy lại `make seed` và đọc dòng `WARNING` |
| Report lưu inline thay vì MinIO | MinIO chưa lên hoặc sai key | Kiểm tra MinIO console http://localhost:9001; kết quả vẫn có trong `report_markdown` của ExecutionResult |

## 4. Debug bằng LangGraph dev server

Mission Graph được expose cho dev server qua `langgraph.json`
(graph `mission_graph` = biến compiled graph trong
`runtime/mission_graph.py`, dùng chung `.env`):

```bash
make langgraph-dev
# tương đương: langgraph dev --allow-blocking
```

- Dev server chạy ở **http://localhost:8123**
- LangGraph Studio: **https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123**
  (mở Studio, trỏ về base URL trên) — chạy thử graph, xem state từng node,
  chỉnh input mà không cần qua API/Celery.
- Dev server dùng đúng `.env` (LiteLLM base URL/key/model) nên không cần
  cấu hình riêng. Cần `langgraph-cli` (đã có trong dev dependencies).

## 5. LangSmith tracing

Bật trong `.env` (đã có sẵn trong `.env.example`, key để trống):

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<key của bạn>
LANGSMITH_PROJECT=agentic-runtime
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Tracing là chuẩn env-based của LangSmith — không tự build tracing riêng.
`infra/langsmith_setup.py` đọc các biến này và bật tracing cho cả API,
worker và dev server. Mọi run của Mission Graph / Deep Agent đều được trace
dưới project đã cấu hình.

**Kiểm tra:** chạy một execution (hoặc `make demo`), rồi mở
https://smith.langchain.com → project `agentic-runtime` → thấy run mới với
từng node (understand → capability → deep_agent → validate) và các tool call.
Nếu không thấy trace: kiểm tra `LANGSMITH_TRACING=true` và `LANGSMITH_API_KEY`
đã điền chưa; nếu tracing bật mà không có key, log sẽ cảnh báo và tracing
tắt (không crash).

## 6. Eval / test bằng MLflow

Harness `scripts/eval_mission.py` chạy mission `fault_analysis` trên 3 test
case cố định (O21 tháng 10/2026: deterministic, autonomous, tuần đầu hybrid)
và log kết quả thật đo được vào MLflow experiment **`agentic-runtime-eval`**:

- params: `mission`, `mode`, `model`, `llm` (`mock` | `litellm`)
- metrics: `success` 0/1, `evidence_count`, `error_evidence_count`,
  `validation_passed` 0/1, `duration_seconds`
- artifact: report Markdown (`report.md`)

```bash
make eval        # mock LLM — KHÔNG cần key/proxy, chạy được ngay
make eval-real   # LLM thật qua LiteLLM (cần 4 biến env)
```

Xem kết quả:

```bash
# Cách 1: MLflow qua compose (cần profile observability)
docker compose --profile observability up -d
# mở http://localhost:5000 → experiment agentic-runtime-eval

# Cách 2: chạy mlflow ui local (sqlite backend, vì file store đã lỗi thời)
mlflow ui --backend-store-uri sqlite:///tmp/mlflow-test/mlflow.db
```

Lưu ý: mock mode được ghi rõ `llm=mock` trong MLflow — không lẫn với run
thật. Harness chỉ log giá trị đo thật từ lần chạy (không bịa metric).

## 7. Biến môi trường

Xem đầy đủ trong `.env.example`. Các biến quan trọng:

| Biến | Mặc định | Ghi chú |
| ---- | -------- | ------- |
| `DATABASE_URL` | postgres local | PostgreSQL cho mission/execution |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | redis local | Queue + result backend |
| `API_KEY` | *(trống)* | Trống = tắt auth (chỉ dev). Production **bắt buộc** đặt. |
| `LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL` / `LLM_TEMPERATURE` | `http://localhost:4000` / *(trống)* / `gpt-4o-mini` / `0.2` | LiteLLM proxy chuẩn OpenAI (bắt buộc cho run thật). |
| `EMBEDDING_API_BASE` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` | *(trống = dùng chung LLM)* / `text-embedding-3-small` | Model embedding trên proxy. |
| `ANTHROPIC_API_KEY` | *(trống)* | Alternative provider: `pip install -e ".[anthropic]"`. Chỉ là alternative, mặc định dùng LiteLLM. |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` / `LANGSMITH_ENDPOINT` | `true` / *(trống)* / `agentic-runtime` / `https://api.smith.langchain.com` | Bật trace khi có key. |
| `QDRANT_URL` / `QDRANT_COLLECTION` | localhost:6333 / `technical_documents` | RAG retrieval |
| `FALKOR_HOST` / `FALKOR_PORT` | localhost / 6379 | Knowledge Graph |
| `MINIO_ENDPOINT` / `MINIO_BUCKET` / `MINIO_REPORTS_BUCKET` | localhost:9000 / `agentic-runtime` / `reports` | Artifacts, reports |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server (compose service `mlflow`, profile `observability`) |
| `MCP_SERVER_URL` / `MCP_TRANSPORT` | *(trống)* | URL MCP server generic, không hardcode. Trống = không load MCP tools. |
| `CHECKPOINT_ENABLED` | `false` | Bật checkpoint Postgres cho LangGraph (cần `pip install -e ".[postgres-checkpoint]"`). |
| `EXECUTION_TIMEOUT_SECONDS` | `3600` | Timeout task Celery |

## 8. Các quyết định kiến trúc đã chốt

| Vấn đề | Quyết định |
| ------ | ---------- |
| Mission có phải LangGraph không? | Không, Mission là cấu hình |
| Runtime chính là gì? | LangGraph |
| Deep Agent nằm ở đâu? | Bên trong Mission Graph |
| Capability là gì? | LangGraph subgraph hoặc nhóm năng lực có thể tái sử dụng |
| Tool là gì? | Đơn vị hành động được Agent gọi |
| Có cần tự xây Trace không? | Không, dùng LangSmith |
| Execution có cần state riêng không? | Có, mỗi execution có state riêng |
| Có dùng Celery không? | Có, cho async dispatch |
| Có cần conversation memory không? | Không, mỗi execution độc lập |

## 9. Cấu trúc thư mục

```
app/                  FastAPI platform (main, config, models, schemas, routers,
                      services, core/db + security, worker/celery_app + tasks)
runtime/              LangGraph runtime: state, mission_graph, deep_agent,
                      llm_factory (LiteLLM), report (Markdown), capabilities/
                      fault_analysis, tools/registry + fault_tools + mcp_adapter
infra/                Thin clients: postgres (reuse app core), qdrant, falkor,
                      minio, langsmith_setup, repositories (Postgres/RAG/KG thật)
seeds/                Mission mẫu (fault_analysis.yaml) + loader + demo_data
                      (Postgres/Qdrant/FalkorDB seed) + seed_demo_data
scripts/              smoke_e2e.py (make demo), eval_mission.py (make eval)
tests/                Pytest, không cần infra thật hay API key thật
langgraph.json        Expose mission_graph cho `langgraph dev`
```

## 10. Stub & giới hạn

- **Repository thật**: `infra/repositories.py` truy vấn Postgres/Qdrant/FalkorDB
  thật. Khi store không tới được, node ghi evidence lỗi (`kind: "error"`) và
  run vẫn hoàn thành — **không trả dữ liệu bịa**.
- **Sensor data**: chưa có store cảm biến — `get_sensor_data` vẫn trả stub có
  đánh dấu `"stub": true`.
- **Auth**: `X-API-Key` so với `API_KEY` env. Đây là stub auth tối giản cho
  scaffold; production cần thay bằng JWT/OAuth theo Policy Manager.
- **Policy enforcement**: `ToolRegistry.check()` kiểm tra capability +
  allowed tools + permissions theo execution context. Policy CRUD đã có;
  giới hạn rate/quota theo policy chưa triển khai.
- **Checkpoint**: mặc định tắt. Bật bằng `CHECKPOINT_ENABLED=true` + extra
  `postgres-checkpoint`.
- **MLflow**: service trong compose ở profile `observability`, mặc định không
  chạy; eval harness (`scripts/eval_mission.py`) là nơi gọi MLflow API.
- **Chưa verify live**: không có Docker daemon và không có LiteLLM endpoint/key
  ở môi trường build này, nên `docker compose up` và một execution LLM thật
  chưa được chạy — cần chạy `make demo` sau khi điền `.env`.
