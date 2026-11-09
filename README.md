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
| MLflow | Observability chính: trace LangGraph, Prompt Registry, eval runs (`agentic-runtime-eval`) |
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

### Bước 3: nạp mission mẫu + dữ liệu thật

Mission mẫu `fault_analysis` (định nghĩa trong
`seeds/missions/fault_analysis.yaml`) được nạp bằng:

```bash
python -m seeds.load_missions
```

Dữ liệu chạy mission **phải là dữ liệu thật** từ nguồn của bạn — repo không
còn chứa dữ liệu demo nào:

- PostgreSQL: ghi lỗi vào bảng `fault_history`, bảo trì vào
  `maintenance_records` (schema trong `app/models.py`).
- Qdrant: index tài liệu kỹ thuật thật vào collection
  `technical_documents` (vector dim được đo từ probe embedding của model
  bạn cấu hình, không hardcode 1536/3072).
- FalkorDB: ghi quan hệ tài sản thật vào graph (ví dụ `PART_OF`,
  `HAS_COMPONENT`).

Chưa có dữ liệu thì mission vẫn chạy được, nhưng evidence sẽ rỗng và kết
quả phân tích không có căn cứ — hệ thống không tự bịa dữ liệu thay thế.

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

### Chạy ngoài Docker (local)

Cách này dùng Python trên máy bạn, không cần Docker cho app/worker
(chỉ Redis/Postgres/Qdrant… vẫn cần chạy ở đâu đó — xem ghi chú bên dưới).
Yêu cầu Python **3.12** (tối thiểu 3.11, xem `requires-python` trong
`pyproject.toml`).

**Bước 1 — tạo venv và kích hoạt:**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
```

**Bước 2 — cài dependencies:**

```bash
pip install -r requirements.txt   # 188 package đã pin sẵn, khớp .venv đang chạy tốt
pip install -e .                  # cài chính project agentic-runtime (editable)
```

Hai bước riêng vì `requirements.txt` chỉ pin thư viện ngoài; bản thân
project cài bằng `pip install -e .`. Muốn cài nhanh bằng range trong
`pyproject.toml` thay vì pin cứng thì dùng `make install`
(`pip install -e ".[dev]"`). `make install-requirements` chỉ chạy bước
`pip install -r requirements.txt`.

**Bước 3 — cấu hình `.env`:**

```bash
cp .env.example .env
```

Điền các giá trị vào `.env` (chỉ tên biến, không có giá trị mẫu ở đây).
Để chạy LLM thật cần đủ 4 biến LiteLLM: `LLM_API_BASE`, `LLM_API_KEY`,
`LLM_MODEL`, `EMBEDDING_MODEL`. Key tuyệt đối không đưa lên Git hay gửi
qua chat. Muốn chạy thử offline không cần key thì dùng chế độ mock
(`make eval`).

**Bước 4 — chạy API và worker (2 terminal riêng):**

```bash
make run-api      # uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
make run-worker   # celery -A app.worker.celery_app worker --loglevel=info
```

Worker cần Redis đang chạy: cài local rồi chạy `redis-server`, hoặc dùng
compose chỉ cho Redis: `docker compose up -d redis`. API/DB cũng vậy:
Postgres (`DATABASE_URL`), Qdrant, FalkorDB, MinIO đều có thể lấy từ
compose (`docker compose up -d postgres redis qdrant falkordb minio`)
trong khi API + worker chạy local — cách lai này tiện khi debug.

**Bước 5 — debug bằng LangGraph dev server:**

```bash
make langgraph-dev   # langgraph dev --allow-blocking → http://localhost:8123
```

**Kiểm tra, eval, demo:**

```bash
make test             # pytest -q
make lint             # ruff check app runtime infra tests seeds scripts
make eval             # eval mock LLM, offline, không cần key/proxy
make eval-real        # eval với model thật (cần 4 biến LiteLLM)
make demo             # smoke test end-to-end qua API (cần 4 biến LiteLLM)
python scripts/register_prompts.py   # đăng prompt fault_investigator lên MLflow (cần MLflow server chạy)
```

> `make eval` ghi kết quả lên MLflow server ở `MLFLOW_TRACKING_URI`
> (mặc định `http://localhost:5000`). Server compose lưu artifact trên
> MinIO (S3), nên máy chạy eval cần `boto3` (đã có trong
> `requirements.txt`) và 3 biến `MLFLOW_S3_ENDPOINT_URL`,
> `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` trong `.env` (xem
> `.env.example`). Thiếu `boto3` sẽ báo `No module named 'boto3'` và
> dừng ở exit 3 — lỗi này là ở máy client, không phải server.
> Không muốn dùng server: `python scripts/eval_mission.py --mock-llm
> --tracking-uri sqlite:///tmp/mlflow-test/mlflow.db` (lưu local, không cần S3).

**Bảng lệnh nhanh:**

| Việc | Lệnh |
| ---- | ---- |
| Tạo venv | `python3 -m venv .venv && source .venv/bin/activate` |
| Cài deps pin sẵn | `pip install -r requirements.txt && pip install -e .` |
| Cài deps theo range | `make install` |
| Cấu hình | `cp .env.example .env` rồi điền giá trị |
| Chạy API | `make run-api` |
| Chạy worker | `make run-worker` (cần Redis) |
| LangGraph dev | `make langgraph-dev` |
| Test / lint | `make test` / `make lint` |
| Eval mock (offline) | `make eval` |
| Eval + demo thật | `make eval-real` / `make demo` |
| Đăng prompt lên MLflow | `python scripts/register_prompts.py` |

### Troubleshooting

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
| ----------- | ---------------------- | ---------- |
| `make demo` báo thiếu biến env | Chưa điền `.env` | Điền đủ 4 biến `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL` |
| Execution FAILED, lỗi kết nối proxy | LiteLLM proxy chưa chạy / sai URL/key | Kiểm tra `LLM_API_BASE` có tới được proxy không, key đúng không |
| Qdrant báo dimension mismatch | Đổi `EMBEDDING_MODEL` sang model có dim khác sau khi collection đã tạo | Đổi lại đúng model cũ, hoặc xóa collection `technical_documents` rồi index lại tài liệu thật |
| Qdrant trống / retrieval trả lỗi | Chưa có tài liệu thật trong Qdrant | Index tài liệu kỹ thuật thật vào collection `technical_documents` |
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

## 6. Observability chính & Eval bằng MLflow 3.x

MLflow 3.x là hệ thống observability chính (trace + eval + prompt), còn
LangSmith (mục 5) giữ lại để debug trong lúc dev. MLflow không tính chi
phí LLM bằng bảng giá LiteLLM (bạn tự host LiteLLM gọi vLLM) — chỉ token
usage mà tracing ghi lại được giữ lại.

### Tracing

`infra/mlflow_setup.py` bật `mlflow.langchain.autolog()` cho cả API, worker
và `langgraph dev` (server MLflow 3.x phải chạy, xem bên dưới). Mỗi
execution được trace thành các span lồng nhau: node của Mission Graph
(understand → capability → deep_agent → validate), từng lần gọi LLM, từng
tool call, từng lần retrieval — kèm latency và token usage khi upstream
trả về. Xem tại **http://localhost:5000** → experiment `agentic-runtime`
→ tab Traces. Nếu server chưa chạy hoặc tắt tracing
(`MLFLOW_TRACING_ENABLED=false`), app vẫn chạy bình thường, chỉ không có
trace.

**Nâng cấp server lên 3.x:** service `mlflow` trong compose build từ
`docker/mlflow/Dockerfile` (`ghcr.io/mlflow/mlflow:v3.18.0` +
`mlflow[genai]`). Nếu đang chạy server 2.x, backup rồi chạy một lần trước
khi start server mới:

```bash
docker compose run --rm mlflow mlflow db upgrade postgresql://postgres:postgres@postgres:5432/agentic_runtime
```

### Prompt Registry

System prompt của Deep Agent (`fault_investigator`) nằm trong Prompt
Registry của MLflow — đổi prompt không cần deploy lại code:

```bash
python scripts/register_prompts.py            # đăng ký version mới + gắn alias production
python scripts/register_prompts.py --alias staging   # gắn alias khác
```

`runtime/prompts.py` load theo thứ tự `@production` → `@latest` → fallback
là hằng số `DEFAULT_SYSTEM_PROMPT` trong `runtime/deep_agent.py`. Chưa đăng
ký prompt hoặc server không tới được thì agent vẫn chạy bằng fallback.

### Eval

Harness `scripts/eval_mission.py` chạy mission `fault_analysis` trên 3 test
case cố định (O21 tháng 10/2026: deterministic, autonomous, tuần đầu hybrid)
và log kết quả thật đo được vào MLflow experiment **`agentic-runtime-eval`**:

- params: `mission`, `mode`, `model`, `llm` (`mock` | `litellm`)
- metrics: `success` 0/1, `evidence_count`, `error_evidence_count`,
  `validation_passed` 0/1, `duration_seconds`
- artifact: report Markdown (`report.md`)

```bash
make eval            # mock LLM — KHÔNG cần key/proxy, chạy được ngay
make eval-real       # LLM thật qua LiteLLM (cần 4 biến env)
make eval-genai      # + mlflow.genai.evaluate, judge qua LiteLLM proxy
make eval-genai-real # judge + prediction đều dùng LLM thật
```

`--genai` chạy thêm `mlflow.genai.evaluate()` trên cùng 3 case với một
judge `evidence_grounding` (đi qua LiteLLM proxy bằng `make_judge(...,
base_url=<proxy>/v1)`), chấm xem root cause có căn cứ trong evidence
không. Nếu chưa cấu hình LiteLLM thì `--genai` in `SKIPPED` và bỏ qua phần
này, kết quả eval cổ điển không bị ảnh hưởng.

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
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server 3.x (compose service `mlflow`, profile `observability`) |
| `MLFLOW_EXPERIMENT` / `MLFLOW_TRACING_ENABLED` | `agentic-runtime` / `true` | Experiment cho trace + eval; `false` = tắt tracing MLflow |
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
| Có cần tự xây Trace không? | Không, dùng MLflow 3.x làm observability chính (trace + eval + prompt); LangSmith giữ lại để debug dev |
| Execution có cần state riêng không? | Có, mỗi execution có state riêng |
| Có dùng Celery không? | Có, cho async dispatch |
| Có cần conversation memory không? | Không, mỗi execution độc lập |

## 9. Cấu trúc thư mục

```
app/                  FastAPI platform (main, config, models, schemas, routers,
                      services, core/db + security, worker/celery_app + tasks)
runtime/              LangGraph runtime: state, mission_graph, deep_agent,
                      llm_factory (LiteLLM), report (Markdown), prompts
                      (MLflow Prompt Registry), capabilities/
                      fault_analysis, tools/registry + fault_tools + mcp_adapter
infra/                Thin clients: postgres (reuse app core), qdrant, falkor,
                      minio, langsmith_setup, mlflow_setup (tracing MLflow 3.x),
                      repositories (Postgres/RAG/KG thật)
seeds/                Mission mẫu (fault_analysis.yaml) + loader nạp mission
scripts/              smoke_e2e.py (make demo), eval_mission.py (make eval),
                      register_prompts.py (MLflow Prompt Registry)
training/             Vòng đời fine-tune: common/ (schemas, dữ liệu tổng
                      hợp, hook thu thập), embedding/ (train + eval
                      retrieval), llm/ (train SFT + eval tool-use),
                      configs/, requirements-training.txt
docs/                 security-cost-roadmap.md, prompt-injection-defense.md,
                      fine-tuning-lifecycle.md
docker/mlflow/        Dockerfile cho MLflow 3.x server (genai extras)
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
- **MLflow 3.x**: observability chính — tracing (autolog LangGraph),
  Prompt Registry (`fault_investigator`), eval (`agentic-runtime-eval`).
  Service trong compose ở profile `observability`, build từ
  `docker/mlflow/Dockerfile` (image 3.18.0 + `mlflow[genai]`).
- **Chưa verify live**: không có Docker daemon và không có LiteLLM endpoint/key
  ở môi trường build này, nên `docker compose up` và một execution LLM thật
  chưa được chạy — cần chạy `make demo` sau khi điền `.env`.

## 11. Fine-tuning lifecycle (chuẩn bị — chưa train thật)

Quy trình đầy đủ: [`docs/fine-tuning-lifecycle.md`](docs/fine-tuning-lifecycle.md).
Scaffold: [`training/`](training/) (xem `training/README.md`).

Ba track — embedding (retrieval Qdrant), LLM chính (Deep Agent), tool-use —
chạy qua 6 giai đoạn: thu thập dữ liệu → chuẩn hóa dataset → train →
eval offline (phải thắng baseline) → registry + deploy qua vLLM →
giám sát + retrain. Hiện chưa có dữ liệu hoạt động thật nên scaffold
chạy bằng dữ liệu tổng hợp (`source: "synthetic"`, 20 ví dụ/track) và
hook thu thập để dữ liệu thật tích lũy dần.

```bash
make gen-synthetic-data   # sinh dữ liệu tổng hợp vào training/data/
make eval-tool-use        # eval tool-use bằng mock predictor (chạy được ngay)
make train-embedding      # cần torch/GPU + base model thật trong config
make train-llm            # cần torch/GPU + base model thật trong config
make eval-retrieval MODEL=<model> [BASELINE=<model>]
```
