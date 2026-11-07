# Agentic Runtime

Runtime cho các Agent tự hành: **FastAPI** quản lý Platform, **Celery + Redis**
dispatch execution, **LangGraph** điều phối Mission, **Deep Agents** thực thi
tự hành, **Capability Graphs / Tools** cung cấp năng lực nghiệp vụ,
**LangSmith** quan sát toàn bộ quá trình.

> Repo này là scaffold khởi tạo (v0.1.0). Một số repository phía dưới vẫn là
> stub có đánh dấu rõ ràng — xem mục [Stub & giới hạn](#stub--giới-hạn).

## 1. Tổng quan kiến trúc (6 lớp)

| Lớp | Thành phần | Vai trò |
| --- | ---------- | ------- |
| 1 | **Platform API — FastAPI** (`app/`) | Quản lý Mission, Agent, Tool, Model, Policy và Execution API. Mission là **cấu hình**, không phải graph. |
| 2 | **Execution Dispatcher — Celery + Redis** (`app/worker/`) | Nhận yêu cầu chạy Mission, đưa vào queue, worker chạy LangGraph. Tách vòng đời HTTP khỏi vòng đời Agent. |
| 3 | **Agent Runtime — LangGraph** (`runtime/mission_graph.py`) | Một Mission Graph cho mỗi nhóm workflow: understand → router → capability / deep agent → validate → END. Hỗ trợ checkpoint Postgres (tùy chọn). |
| 4 | **Autonomous Agent — Deep Agents** (`runtime/deep_agent.py`) | Lập kế hoạch, phân rã công việc, dùng subagents và tools. Nằm **bên trong** Mission Graph (Deep Agents build trên LangGraph, trả về compiled graph). |
| 5 | **Capability & Tool Layer** (`runtime/capabilities/`, `runtime/tools/`) | Capability = LangGraph subgraph tái sử dụng (vd: `fault_analysis`). Tool = hành động nhỏ Agent có thể gọi. Quyền thực tế kiểm tra ở runtime theo mission/tenant/execution. |
| 6 | **Data & Observability** (`infra/`) | PostgreSQL, Redis, Qdrant, FalkorDB, MinIO, MLflow (tùy chọn), LangSmith (chỉ quan sát, không phải nơi lưu trạng thái). |

### Lưu trữ

| Hệ thống | Dữ liệu |
| -------- | ------- |
| PostgreSQL | Mission, Agent, Tool metadata, execution, kết quả |
| Redis | Celery queue, cache, trạng thái điều phối tạm thời |
| Qdrant | Document embeddings, vector retrieval |
| FalkorDB | Knowledge Graph, entity relationships |
| MinIO | File, tài liệu, artifact, report |
| MLflow | Model và embedding experiment, evaluation |
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
   (SQL), bảo trì, tìm tài liệu (RAG/Qdrant), quan hệ thiết bị (FalkorDB),
   rồi tổng hợp evidence.
5. **Deep Agent (nếu evidence chưa đủ)** — lập kế hoạch điều tra thêm, tự quyết
   định gọi tools cho đến khi đủ evidence.
6. **Validate kết quả** — kiểm tra đủ evidence, đúng output schema
   (`root_causes`, `evidence`, `recommendations`) và success criteria.
7. **Lưu kết quả** — PostgreSQL lưu kết quả và trạng thái; MinIO lưu report
   nếu có. Toàn bộ quá trình được trace trên LangSmith.

Sơ đồ Mission Graph:

```
START → Understand Mission → Router → Capability Graph → (Deep Agent) → Validate → END
                                      ↘ Deep Agent (autonomous) ↗
```

Router chọn nhánh theo `mission.execution.mode`:
`deterministic` → chỉ Capability Graph; `autonomous` → chỉ Deep Agent;
`hybrid` → Capability Graph trước, Deep Agent chỉ khi evidence chưa đủ.

## 3. Chạy local

### Cách 1: Docker Compose (khuyên dùng)

```bash
cp .env.example .env   # điền key thật vào .env, không commit file này
docker compose up -d --build
make seed              # nạp mission mẫu fault_analysis
```

- API: http://localhost:8000 — docs tại `/docs`
- MinIO console: http://localhost:9001
- Qdrant: http://localhost:6333
- MLflow (tùy chọn): `docker compose --profile observability up -d`

### Cách 2: chạy trực tiếp (cần Postgres/Redis local hoặc qua compose)

```bash
make install           # pip install -e ".[dev]"
make run-api           # uvicorn app.main:app --reload
make run-worker        # celery -A app.worker.celery_app worker
make test              # pytest -q
make lint              # ruff check
```

### Gọi thử API

```bash
# Tạo execution (cần API_KEY nếu đã cấu hình, header X-API-Key)
curl -X POST http://localhost:8000/executions \
  -H "Content-Type: application/json" \
  -d '{"mission_id": "fault_analysis", "input": {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"}}'

# Theo dõi
curl http://localhost:8000/executions/<execution_id>
curl http://localhost:8000/executions/<execution_id>/result
```

## 4. Biến môi trường

Xem đầy đủ trong `.env.example`. Các biến quan trọng:

| Biến | Mặc định | Ghi chú |
| ---- | -------- | ------- |
| `DATABASE_URL` | postgres local | PostgreSQL cho mission/execution |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | redis local | Queue + result backend |
| `API_KEY` | *(trống)* | Trống = tắt auth (chỉ dev). Production **bắt buộc** đặt. |
| `LLM_PROVIDER` / `LLM_MODEL` | `openai` / `gpt-4o-mini` | Provider chính là OpenAI (`langchain-openai`). Anthropic là alternative: `pip install -e ".[anthropic]"` rồi đặt `LLM_PROVIDER=anthropic`. |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | *(trống)* | Không có key thì Deep Agent không chạy được (node ghi lỗi, execution FAILED sau validate). |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | `false` / *(trống)* | Chỉ bật trace khi cả hai đều có. LangSmith chỉ để quan sát/debug, không thay PostgreSQL. |
| `QDRANT_URL` / `QDRANT_COLLECTION` | localhost:6333 | RAG retrieval |
| `FALKOR_HOST` / `FALKOR_PORT` | localhost / 6379 | Knowledge Graph |
| `MINIO_ENDPOINT` / `MINIO_BUCKET` | localhost:9000 | Artifacts, reports |
| `MCP_SERVER_URL` / `MCP_TRANSPORT` | *(trống)* | URL MCP server generic, không hardcode. Trống = không load MCP tools. |
| `CHECKPOINT_ENABLED` | `false` | Bật checkpoint Postgres cho LangGraph (cần `pip install -e ".[postgres-checkpoint]"`). |
| `EXECUTION_TIMEOUT_SECONDS` | `3600` | Timeout task Celery |

## 5. Các quyết định kiến trúc đã chốt

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

## 6. Cấu trúc thư mục

```
app/                  FastAPI platform (main, config, models, schemas, routers,
                      services, core/db + security, worker/celery_app + tasks)
runtime/              LangGraph runtime: state, mission_graph, deep_agent,
                      capabilities/fault_analysis, tools/registry + fault_tools + mcp_adapter
infra/                Thin clients: postgres (reuse app core), qdrant, falkor,
                      minio, langsmith_setup, repositories (SQL/RAG/KG + stub fallback)
seeds/                Mission mẫu (fault_analysis.yaml) + loader
tests/                Pytest smoke tests, không cần infra thật hay API key
```

## 7. Stub & giới hạn

- **Repository stubs**: `infra/repositories.py` thử truy vấn thật (bảng
  `fault_history` / `maintenance_records` nếu có, Qdrant, FalkorDB); khi chưa
  có dữ liệu hoặc client chưa cấu hình, trả về dữ liệu demo có đánh dấu
  `"source": "stub"`. Thay bằng repository thật khi có schema dữ liệu.
- **Sensor data**: chưa có store cảm biến trong scaffold — `get_sensor_data`
  trả stub.
- **Auth**: `X-API-Key` so với `API_KEY` env. Đây là stub auth tối giản cho
  scaffold; production cần thay bằng JWT/OAuth theo Policy Manager.
- **Policy enforcement**: `ToolRegistry.check()` kiểm tra capability +
  allowed tools + permissions theo execution context. Policy CRUD đã có;
  giới hạn rate/quota theo policy chưa triển khai.
- **Checkpoint**: mặc định tắt. Bật bằng `CHECKPOINT_ENABLED=true` + extra
  `postgres-checkpoint`.
- **MLflow**: service trong compose ở profile `observability`, mặc định không
  chạy; code chưa gọi MLflow API (chỉ cấu hình URI).
