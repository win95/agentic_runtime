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

### Bước 3: nạp mission mẫu + dữ liệu demo

Mission mẫu `fault_analysis` (định nghĩa trong
`seeds/missions/fault_analysis.yaml`) được nạp bằng:

```bash
python -m seeds.load_missions
```

Dữ liệu demo cho thang máy O21, tháng 10/2026 (8 lỗi + 6 bảo trì trong
PostgreSQL, 6 tài liệu kỹ thuật trong Qdrant, 5 quan hệ trong FalkorDB)
được nạp bằng:

```bash
make seed
# hoặc: python -m seeds.seed_demo_data
# kiểm tra dữ liệu mà không chạm vào kho nào: make seed-dry-run
# qua compose: docker compose --profile setup run --rm seed
```

Seed là idempotent — chạy lại không nhân đôi dữ liệu. Qdrant cần
LiteLLM proxy để embedding tài liệu (thiếu key/proxy thì phần Qdrant
được bỏ qua với cảnh báo, chạy lại sau khi sửa). Sau này thay dữ liệu
demo bằng dữ liệu thật của bạn:

- PostgreSQL: ghi lỗi vào bảng `fault_history`, bảo trì vào
  `maintenance_records` (schema trong `app/models.py`).
- Qdrant: index tài liệu kỹ thuật thật vào collection
  `technical_documents` (vector dim được đo từ probe embedding của model
  bạn cấu hình, không hardcode 1536/3072).
- FalkorDB: ghi quan hệ tài sản thật vào graph (ví dụ `PART_OF`,
  `HAS_COMPONENT`).

Đường đưa dữ liệu thật vào theo lô (không cần viết code nạp tay):
xem `docs/ingest.md` — thả file JSON vào `ingest/inbox/` rồi chạy
`make ingest` (beat tự chạy mỗi 30 phút khi stack lên).

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
make lint             # ruff check app runtime infra ingest tests seeds scripts
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
                      fault_analysis, tools/registry + fault_tools + admin_tools
                      + mcp_adapter, middleware/ (token + audit callbacks),
                      skills/ (skill loader: SKILL.md selection)
skills/               Skill playbooks (SKILL.md): fault-investigation,
                      admin-operations, report-writing + README
infra/                Thin clients: postgres (reuse app core), qdrant, falkor,
                      minio, langsmith_setup, mlflow_setup (tracing MLflow 3.x),
                      repositories (Postgres/RAG/KG thật)
seeds/                Mission mẫu + agent + schedule (YAML) + loader,
                      dữ liệu demo O21 (`demo_data.py`, `seed_demo_data.py`)
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
- **Sensor data**: số liệu thật trong bảng Postgres `sensor_readings`
  (`seeds/demo_data.py` nạp 62 số liệu O21 tháng 10/2026); thiếu số liệu thì
  trả danh sách rỗng, không bịa.
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

## 12. Admin bot (system_admin) — quản trị chính hệ thống

Bot admin (`admin_bot` + mission `system_admin`, mode autonomous) dùng
Deep Agent để trả lời yêu cầu quản trị bằng tiếng Việt: xem execution
lỗi, xem missions/agents, kiểm tra sức khỏe hạ tầng, tạo/sửa/xóa mission,
dispatch và cancel execution. Nạp seed rồi gọi như mọi mission khác:

```bash
python -m seeds.load_missions   # nạp cả system_admin + admin_bot
python scripts/register_prompts.py --prompt admin_bot  # đăng ký prompt (cần MLflow server)

# Gọi thử API bằng tay (cần API_KEY nếu đã cấu hình, header X-API-Key)
curl -X POST localhost:8000/executions \
  -H 'Content-Type: application/json' \
  -d '{"mission_id": "system_admin", "input": {"request": "execution nào lỗi hôm nay?"}}'
```

Phạm vi **đọc + ghi** — bot có 11 admin tool trong
`runtime/tools/admin_tools.py`, tất cả `permission="admin"`. Registry chỉ
đưa các tool này cho mission nào khai `capabilities: [system_admin]`
(mission `fault_analysis` không bao giờ thấy chúng). Hành động phá hủy
(`delete_mission`, `cancel_execution`) cần đúng id trong yêu cầu; yêu cầu
mơ hồ thì bot từ chối và hỏi lại id. `delete_mission` không cascade: mission
còn execution thì bị từ chối (409) thay vì xóa lịch sử chạy.

## 13. Middleware — log token + audit (không tính chi phí)

Mỗi lần Deep Agent chạy, `deep_agent_node` gắn một `TokenAuditCallbackHandler`
(`runtime/middleware/`) qua config của `agent.invoke` — một instance cho mỗi
execution, mang theo `execution_id` / `mission_id` / `tenant`, không dùng
global. Hai bảng mới (tạo cùng `Base.metadata.create_all`, không có migration
riêng — repo chưa dùng alembic):

- `llm_usage`: một dòng cho mỗi lần gọi LLM — model, `prompt_tokens` /
  `completion_tokens` / `total_tokens`, `usage_source`. Số token chỉ lấy từ
  `response.llm_output["token_usage"]` của provider; provider không báo thì
  các cột để NULL và `usage_source="missing"` — không ước lượng, không bịa số.
- `audit_logs`: một dòng cho mỗi lần gọi tool — `tool_name`, input đã redact
  và cắt 2000 ký tự, `status` (`started` / `success` / `error`), `error`,
  `duration_ms`.

**Quy tắc không tính chi phí:** chỉ log token, không có cột chi phí, không tra
bảng giá LiteLLM — bạn tự host LiteLLM → vLLM và tự quản chi phí. **Bảo mật:**
giá trị của các key chứa `key` / `token` / `password` / `secret` trong input
tool được thay bằng `[redacted]` trước khi ghi; ghi log là best-effort — lỗi
ghi log không bao giờ làm hỏng execution. Đọc số liệu bằng helper trong
`runtime/middleware/queries.py` (`get_usage_for_execution`,
`get_usage_totals`, `get_audit_for_execution`, `get_tool_call_counts`):

```python
from app.core.db import SessionLocal
from runtime.middleware import get_usage_totals, get_audit_for_execution

db = SessionLocal()
print(get_usage_totals(db, mission_id="system_admin"))
print(len(get_audit_for_execution(db, "exec-123")))
```

## 14. Skills — playbook nạp cho Deep Agent khi liên quan

Skills là playbook dạng markdown (`skills/<tên>/SKILL.md`: frontmatter
`name` / `description` / `keywords` + các mục When to Use / Steps /
What NOT to Do), tương đương phần "skills" trong LangSmith Managed Deep
Agents nhưng tự host — không dùng LangSmith. `deep_agent_node` chọn skill
một cách xác định, không gọi LLM chỉ để chọn: mission khai `skills:` trong
`execution:` (seed YAML) thì lấy đúng các skill đó; nếu không thì chấm
điểm keyword (≥1 keyword riêng biệt xuất hiện trong mission id +
objective + capabilities + request, đã đưa về dạng số ít) và lấy các
skill đạt ngưỡng, log điểm ở mức debug. Chỉ skill được chọn mới được đưa
cho agent — dùng cơ chế native của `deepagents` 0.7.34
(`create_deep_agent(skills=["/skills/"])` + `invoke(files={...})` qua
`StateBackend` mặc định, vẫn chạy trong bộ nhớ, không cấp quyền đọc/ghi
filesystem thật). Run nào không khớp skill nào thì chạy y như trước
(không có `skills=`, không có `files`).

3 skill có sẵn: `fault-investigation` (thứ tự thu thập evidence, không
bịa dữ liệu, root cause chỉ từ evidence fault/maintenance),
`admin-operations` (đọc trước khi ghi, id chính xác cho thao tác phá
hủy, luật 409 khi xóa mission), `report-writing` (answer +
actions_taken + evidence, trích dẫn đúng evidence đã dùng, "unknown"
trung thực khi evidence toàn lỗi). Thêm skill mới chỉ cần tạo
`skills/<tên>/SKILL.md` theo format trong `skills/README.md`; gán skill
cho mission bằng `skills: [...]` dưới `execution:` trong seed YAML.

## 15. Schedules — mission chạy định kỳ (cron)

Một Schedule là "chạy mission này, với input này, theo cron này". Beat
(`docker compose` service `beat`, `celery -A app.worker.celery_app beat`)
gọi task `run_due_schedules` mỗi 60 giây; task này tạo một Execution
(QUEUED) cho mỗi schedule đang đến hạn rồi worker chạy như thường.
Cron là 5 trường, giờ UTC naive (07:00 UTC = 14:00 giờ Sài Gòn).

```bash
# Tạo schedule qua API (cần X-API-Key nếu đã cấu hình)
curl -X POST localhost:8000/schedules \
  -H 'Content-Type: application/json' \
  -d '{"mission_id": "system_admin", "cron": "0 7 * * *",
       "input_data": {"request": "kiểm tra sức khỏe hệ thống"},
       "enabled": true}'

# Bật / tắt / xóa
curl -X POST localhost:8000/schedules/<id>/enable
curl -X POST localhost:8000/schedules/<id>/disable
curl -X DELETE localhost:8000/schedules/<id>
```

Luật an toàn của dispatcher: schedule đến hạn được đẩy `next_run_at`
lên lần chạy kế tiếp và commit **trước** khi dispatch, nên hai tick
beat chồng nhau không dispatch trùng. Dispatch lỗi thì ghi vào
`last_error` và **giữ nguyên `next_run_at` ở lần đã lỡ** — tick beat
sau chạy lại lần đó thay vì mất hẳn (schedule hỏng hẳn sẽ retry mỗi
phút; tắt schedule để dừng). Không có catch-up: schedule tắt lâu ngày
khi bật lại chỉ chạy một lần ở lần tới, không replay các lần đã lỡ.
Mission có schedule thì không xóa được (409, như luật xóa mission có
execution). Chạy local cần cả worker lẫn beat (hai terminal):
`make run-worker` + `make run-beat` — chỉ chạy worker thì schedule
không bao giờ fire.

Seed mẫu: `seeds/schedules/system_admin_daily_health.yaml` — health
check hàng ngày lúc 07:00 UTC cho mission `system_admin`, được nạp ở
trạng thái **tắt** (`enabled: false`); nạp lại seed không tắt schedule
mà bạn đã bật. Bật nó lên bằng `POST /schedules/sched_system_admin_daily_health/enable`.

## 16. Prompt-injection defense — 6 lớp đã triển khai (Lớp 7 chờ quyết định)

Chi tiết đầy đủ: `docs/prompt-injection-defense.md`. Phòng thủ là xác
suất, không tuyệt đối — các lớp dưới đây chặn phần lớn, phát hiện phần
còn lại, và giới hạn thiệt hại.

- **Lớp 1 — Validate input ở API** (`runtime/security.py`,
  `app/services/execution_service.py`): input JSON tối đa 8 KB, field
  chuỗi tối đa 500 ký tự, `asset_id` chỉ nhận `^[A-Za-z0-9_-]{1,64}$`,
  `start_date`/`end_date` đúng định dạng ngày và `end_date >= start_date`,
  ký tự điều khiển bị lược trước khi lưu. Vi phạm → 422.
- **Lớp 2 — Tách instruction/dữ liệu** (`runtime/deep_agent.py`):
  system prompt nêu thứ bậc (chỉ system prompt + objective mới là lệnh;
  mọi thứ trong `<data>...</data>` là dữ liệu, gặp lệnh trong dữ liệu thì
  bỏ qua và ghi nhận), input của execution được bọc trong thẻ `<data>`.
  Prompt của hai agent trong seed YAML đồng bộ với hằng số trong code.
- **Lớp 3 — Vệ sinh output tool** (`infra/repositories.py`,
  `runtime/tools/fault_tools.py`): field text chuẩn hóa khoảng trắng và
  cắt ngắn ở ranh giới kho (`description`/`notes` tối đa 2.000 ký tự,
  `content` tài liệu tối đa 4.000), tool output kèm `provenance` ghi rõ
  nguồn (`postgres:fault_history`, `qdrant:technical_documents`, ...).
- **Lớp 4 — Least privilege** (`runtime/tools/registry.py`,
  `runtime/tools/mcp_adapter.py`): tool MCP phải map capability trong
  `MCP_TOOL_CAPABILITIES` (JSON) mới được đăng ký — không map là bị từ
  chối mặc định; mọi tool (built-in + MCP) đều qua `registry.check()`.
  Registry gắn nhãn `read_only` (6 tool fault + 6 tool admin đọc là
  `True`; 5 tool admin ghi là `False`, chỉ hiện với mission khai
  `system_admin`). Tool nhận `asset_id` chỉ được gọi với `asset_id` của
  execution hiện tại (trừ khi mission khai
  `execution_config.allow_cross_asset`); gọi lệch trả lỗi có cấu trúc.
  **Chưa làm:** node xác nhận của người (human-in-the-loop) trước khi
  tool ghi chạy không giám sát — cần quyết định kênh phê duyệt.
- **Lớp 5 — Kiểm chứng đầu ra** (`runtime/mission_graph.py`,
  `runtime/report.py`): `validate_results` kiểm tra grounding (mỗi
  `root_causes` phải có căn cứ trong evidence, không thì chuyển sang
  `ungrounded_root_causes` và không vào báo cáo chính thức), trace gọi
  tool (chỉ flag khi gọi tool ngoài danh sách cho phép), và quét tín
  hiệu injection trong evidence. Các flag nằm trong
  `result["validation_flags"]` — chỉ để xem lại, không làm fail run.
  Báo cáo render tối đa 50 evidence và có mục "Flagged".
- **Lớp 6 — Phát hiện & eval** (`runtime/security.py`,
  `scripts/eval_mission.py`): heuristic phát hiện cụm mệnh lệnh
  (`ignore previous instructions`, `disregard`, `you must`, ... kể cả
  tiếng Việt) — chỉ để gắn cờ, không chặn. Eval adversarial:
  `make eval-adversarial` (14 case: injection trực tiếp trong input,
  gián tiếp trong title lỗi / nội dung tài liệu / ghi chú bảo trì, mô
  tả tool MCP độc hại, delimiter trong prompt) — chạy mock, không cần
  MLflow.
- **Lớp 7 — Chưa làm:** trần token/số bước mỗi execution và tách tenant
  thật — thuộc Bước lớn 2 và 3 của roadmap, cần 3 quyết định đang chờ
  (phương thức đăng nhập, hosting, trần chi phí LLM mỗi user).

Sau khi đổi system prompt trong code, chạy lại
`python scripts/register_prompts.py` để đăng ký phiên bản mới lên
MLflow Prompt Registry (cần MLflow server đang chạy).

## 17. Batch ingest — dữ liệu thật theo lô (Phương án B)

Chi tiết đầy đủ: `docs/ingest.md`. Dữ liệu thật đi vào hệ thống bằng
file JSON thả vào inbox (`ingest/inbox/` local, hoặc bucket MinIO
`ingest-inbox`), một loại một file: `faults`, `maintenance`,
`sensor_readings` → Postgres; `documents` → Qdrant (embed qua LiteLLM);
`relations` → FalkorDB (MERGE). Beat chạy
`app.worker.tasks.run_ingest_batch` mỗi `INGEST_INTERVAL_SECONDS`
(mặc định 1800 = 30 phút); chạy tay bằng `make ingest`, kiểm tra
định dạng bằng `make ingest-dry-run`.

Mỗi dòng được validate trước khi load (schema Pydantic + luật Lớp 1,
raise ValueError chứ không dùng `assert` — validation còn nguyên cả
khi chạy `python -O`); dòng lỗi không bị vứt mà vào bảng
`ingest_quarantine` để sửa và chạy lại (`python -m ingest.run --retry`).
Mỗi file có một dòng `ingest_runs` làm watermark — chạy lại skip file
đã SUCCEEDED, và mọi loader đều idempotent nên chạy lại không nhân đôi
dữ liệu. Postgres loader merge từng dòng trong một SAVEPOINT riêng,
nên dòng lỗi chỉ rollback chính nó — số dòng `ok` báo cáo là số dòng
thật sự đã lưu.

Sensor đã có store thật: bảng Postgres `sensor_readings`
(`get_sensor_data` đọc từ đó, thiếu số liệu thì trả danh sách rỗng).
`GET /tools/db` giờ trả 17 tool (mirror từ registry, nạp bởi
`make seed`) và có 1 policy mặc định cho tenant `default`.

## 18. Agent versioning — version toàn bộ agent config (Phương án C)

Chi tiết đầy đủ: `docs/agent-versioning.md`. Một agent chạy bằng 4
thứ (system prompt, model/temperature, allowed tools, và bản thân các
tool) — chỉ version prompt thì không pin được hành vi, vì đổi mô tả
tool cũng đổi cách agent chọn tool. Mỗi `AgentVersion` là một snapshot
bất biến gồm cả 4: prompt text chính xác, MLflow prompt version (nếu
có), provider/model/temperature, allowed tools, và tool manifest kèm
SHA-256 (+ `git_sha` khi chụp trong git checkout).

Deploy: sửa YAML agent rồi chạy
`python scripts/register_agent_version.py --agent-id <id>` (hoặc
`--all`); `make seed` cũng tạo version (v1 lần đầu, không đổi thì
không tạo thêm, đổi YAML rồi seed lại thì thành v2). Model trong YAML
agent là `${LLM_MODEL}` — lúc seed lấy từ `.env` (mặc định
`gpt-4o-mini` nếu không đặt); đổi model thì sửa `.env` rồi seed lại,
version mới được tạo. Rollback qua API:
`POST /agents/{id}/versions/{v}/activate`, xem danh sách ở
`GET /agents/{id}/versions`.

Lúc chạy, `deep_agent_node` dùng active version: prompt đã pin (không
lookup MLflow), model/temperature đã pin, `allowed_tools` thành cổng
chặn thật trong `registry.check()`. Drift (tool live khác manifest đã
pin) được log warning và ghi `agent_version_drift=true` trên execution;
mỗi execution cũng ghi `agent_version` đã chạy.

## 19. Hooks — vòng đời của một lần chạy agent (trích xuất từ Hooks v2)

Chi tiết đầy đủ: `docs/hooks.md`. Hooks v2 của Deep Agents Code chỉ
chạy qua CLI `dcode` + `hooks.json`, còn hệ thống này chạy server-side
bằng SDK nên không dùng được trực tiếp — đã trích xuất ngữ nghĩa của
nó thành `runtime/hooks/`: cùng 12 event, cùng matcher, cùng tác dụng
theo event, viết bằng Python callables + `langchain` `AgentMiddleware`
native (không shell script bên ngoài).

Cái đã nối dây: `UserPromptSubmit` (chặn input vi phạm Layer 1 ngay
trong `deep_agent_node`, trước khi build agent — lưới cho execution đi
qua schedule/admin dispatch), `PreToolUse` (deny → tool không chạy;
built-in chặn `rm -rf`/`DROP TABLE`/... trong args, quan trọng nhất
với MCP tool), `PostToolUse`/`PostToolUseFailure` (feedback nối vào
tool result; tool raise thành error message thay vì crash run),
`Stop` (agent muốn dừng mà chưa gọi tool nào thì bị buộc chạy thêm,
tối đa `stop_max_continues` mặc định 3, qua `after_model` +
`jump_to="model"` — đã kiểm chứng qua graph thật với fake model),
`SessionStart`/`SessionEnd` (chẩn đoán; worker dispatch SessionEnd
bao phủ cả run không chạy agent).

Khai báo trong `execution:` của YAML mission (`hooks: [...]`, giống
`skills:`); không khai báo thì mọi built-in đều chạy. `system_admin`
khai báo tường minh đủ 6 hook. Mỗi lần dispatch ghi một record vào
`MissionState.hook_events`; can thiệp được tóm tắt một dòng trong
`notes`. Chưa nối dây: `PermissionRequest` (cần quyết định kênh phê
duyệt), `PreCompact` (không có compaction), `SubagentStart/Stop`
(chưa có subagent).

Pre-retrieval: hook built-in thứ 7, `retrieval_context` (SessionStart),
chỉ chạy khi mission khai báo `execution.retrieval:` — `fault_analysis`
khai báo `sources: [faults, documents, relations]`, `max_chars: 3000`.
Nó kéo dữ liệu thật của asset thành một block có provenance rồi inject
làm system message trước lượt model đầu tiên, để agent không phải tự
gọi tool tìm lại những gì capability graph đã biết. Store chết thì
ghi `unavailable`, không record thì ghi `no records found`, không bao
giờ bịa. Chi tiết: `docs/hooks.md`.

## 20. Migrations — thay đổi schema bằng Alembic

Trước đây schema chỉ có `init_db()`/`create_all` nên không alter được
bảng đã tồn tại — mọi thay đổi cột đều phải ALTER tay. Từ giờ mọi thay
đổi schema đi qua Alembic (`alembic/versions/`):

```bash
make migrate                 # alembic upgrade head — DB mới
alembic stamp head           # DB đã có sẵn từ create_all: đánh dấu
                             # baseline đã áp dụng, CHỈ chạy một lần
alembic upgrade head         # các migration sau đó áp dụng bình thường
alembic downgrade base       # gỡ toàn bộ (dev/test)
```

- `0001` baseline: toàn bộ schema tính đến commit `b445dc98`, đúng
  những gì `create_all` tạo ra — vì vậy `stamp head` an toàn cho DB
  cũ. Đừng stamp DB nào đã bị sửa tay.
- `0002`: thêm cột `tools.read_only` (backfill `true`, `make seed`
  chạy lại sẽ sửa các tool ghi thành `false` theo registry) và unique
  constraint `(agent_id, version)` trên `agent_versions`.
- Viết migration mới: `alembic revision -m "..."` rồi điền
  `upgrade()`/`downgrade()`; chạy `make test` (có test chạy
  `upgrade head` → `downgrade base` trên SQLite tạm).
- `init_db()`/`create_all` vẫn còn cho test và dev nhanh, nhưng
  production dùng migration.

## 21. Context window budget — 32.000 token (Phương án A)

Một nguồn duy nhất: `runtime/context_budget.py`. Cửa sổ 32.000 token
chia như sau (đo thực tế trước khi chia: system prompt 200–266 token,
17 tool ~1.275 token, 2 skill tối đa ~773 token, retrieval cap cũ 750
token, input cap 8 KB = 2.048 token):

| Thành phần | Token | Tỉ lệ |
|---|---|---|
| System prompt | 1.024 | 3,2% |
| Tool definitions | 2.048 | 6,4% |
| Skills | 1.024 | 3,2% |
| Retrieved context | 3.072 | 9,6% |
| Execution input | 2.048 | 6,4% |
| Hội thoại + tool output | 15.616 | 48,8% |
| Output reserve | 5.120 | 16% |
| KV cache / agent | 2.048 | 6,4% |

Enforce: system prompt/tools/skills đo lúc build agent trong
`deep_agent_node` — vượt thì log warning + ghi vào `notes`, run vẫn
chạy (đây là config đã pin, fail run sẽ giấu việc thật). Retrieval
lấy `max_chars` từ budget (3.072 token ≈ 12.288 ký tự); YAML mission
vẫn override được `max_chars` riêng. Input đã cap sẵn 8.192 byte ở
Layer 1 (= 2.048 token). History: `HookMiddleware.before_model` cắt
các turn cũ nhất (cả AIMessage + ToolMessage của turn đó) cho tới khi
vừa 15.616 token — không bao giờ cắt system/human message; mỗi lần cắt
ghi vào `notes`. Output reserve: `max_tokens=5.120` khi build chat
model, một response không bao giờ ăn hơn phần đã chừa. KV cache là
khoản reserve riêng từng agent; phía prompt (6 thành phần đầu) phải
vừa 24.832 token.

Đếm token là ước lượng `ceil(ký tự / 4)` — proxy phục vụ model custom
không có tokenizer local để đếm chính xác (tiktoken đã tắt vì lý do
này ở `llm_factory`), nên budget enforce trên ước lượng; vòng cắt
chạy lại trước mỗi lượt model nên sai số giới hạn trong một turn.
