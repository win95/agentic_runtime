# Vòng đời fine-tune — embedding, LLM chính, tool-use

> Trạng thái: **chuẩn bị quy trình + scaffold, chưa train thật.**
> Chưa có dữ liệu hoạt động thật (dữ liệu demo O21 đã xóa theo yêu cầu;
> Postgres/Qdrant/FalkorDB đang trống). Mọi ví dụ trong tài liệu này và
> trong `training/` đều là **dữ liệu tổng hợp, ghi rõ `source: "synthetic"`**,
> chỉ để chạy thử pipeline — không được dùng làm bằng chứng chất lượng
> model. Khi dữ liệu thật tới, nó đi qua đúng các hợp đồng dữ liệu ở mục 3.

## 0. Bối cảnh và nguyên tắc

Hệ thống hiện tại chạy bằng model có sẵn qua LiteLLM proxy → vLLM tự host
(`LLM_MODEL`, `EMBEDDING_MODEL` trong `.env`). Fine-tune là bước tiếp theo
để model hiểu đúng nghiệp vụ phân tích lỗi thiết bị, thay vì chỉ dùng
model chung:

- **Embedding model** — retrieval tài liệu kỹ thuật trong Qdrant. Model
  chung không biết từ ngữ nghiệp vụ (mã lỗi, tên thiết bị, tiếng Việt
  kỹ thuật), nên retrieval trả tài liệu gần giống nhưng sai.
- **LLM chính** — Deep Agent lập kế hoạch, gọi tool, tổng hợp kết luận.
  Model chung hay gọi sai tool, sai tham số, hoặc kết luận không grounded
  vào evidence.
- **Tool-use** — lát cắt riêng của LLM chính, tập trung vào kỷ luật gọi
  tool: đúng tool, đúng tham số, không gọi tool ngoài phạm vi mission.

Nguyên tắc:

1. **Không train khi chưa có dữ liệu thật.** Giai đoạn hiện tại chỉ chuẩn
   bị quy trình, hợp đồng dữ liệu, code train/eval và hook thu thập — để
   dữ liệu thật tới đâu là cắm vào tới đó.
2. **Không bịa dữ liệu.** Dữ liệu tổng hợp chỉ để kiểm tra pipeline chạy
   được (schema đúng, train chạy được, metric tính được). Mọi con số chất
   lượng trên dữ liệu tổng hợp đều vô nghĩa với production.
3. **Baseline là luật.** Model fine-tune chỉ được deploy khi thắng model
   gốc trên cùng một bộ test cố định (mục 4). Không có ngoại lệ "cảm giác
   tốt hơn".
4. **Deploy có đường lui.** Bản cũ luôn chạy song song để so sánh, và
   rollback được trong một lần đổi cấu hình (mục 5).
5. **Mọi run đều log lên MLflow.** Không có run train/eval nào chạy ngoài
   MLflow (experiment `agentic-runtime-finetune`), kể cả run thất bại.

## 1. Ba track fine-tune

### 1.1 Track A — Embedding model (retrieval)

| Mục | Nội dung |
| --- | -------- |
| Mục tiêu | Retrieval tài liệu kỹ thuật đúng hơn cho `search_technical_documents` (Qdrant) |
| Phương pháp | Contrastive fine-tune: mỗi ví dụ gồm `query` + `positive` (tài liệu đúng) + `negatives` (tài liệu gần giống nhưng sai). Loss `MultipleNegativesRankingLoss` (sentence-transformers). Bắt đầu từ model embedding đang dùng, không train từ đầu |
| Dữ liệu | Cặp query–tài liệu từ log retrieval thật + tài liệu kỹ thuật thật; hard negative = tài liệu cùng thiết bị khác lỗi, hoặc cùng lỗi khác thiết bị |
| Metric | `recall@k` và `nDCG@k` (k = 5, mặc định) trên bộ test retrieval cố định; so với model gốc |
| Ngưỡng đi tiếp | Thắng baseline ở cả recall@5 và nDCG@5 (mức thắng tối thiểu do người quyết khi có số liệu đầu tiên; mặc định đề xuất +3 điểm) |
| Deploy | Đổi model embedding trên LiteLLM/vLLM → **re-index toàn bộ collection Qdrant** (`technical_documents`) vì vector cũ không tương thích với model mới. Không re-index = retrieval sai hoàn toàn |

### 1.2 Track B — LLM chính (Deep Agent)

| Mục | Nội dung |
| --- | -------- |
| Mục tiêu | Agent lập kế hoạch đúng, gọi tool đúng, kết luận grounded vào evidence |
| Phương pháp | **SFT trước**: học từ trace chạy đúng (prompt + tool call + kết quả + kết luận). **LoRA trước** (rẻ, nhanh, ít rủi ro); full fine-tune chỉ khi LoRA đã chạm trần và có đủ dữ liệu. DPO/RL chỉ tính sau khi đã có dữ liệu phản hồi tốt/xấu |
| Dữ liệu | Trace execution thật (từ MLflow traces) đã qua lọc: chỉ giữ run `SUCCEEDED` + `validation_passed`, có evidence thật, không có evidence lỗi |
| Metric | Task success (giống `success` trong `scripts/eval_mission.py`), `ToolCallCorrectness` / `ToolCallEfficiency` (scorer có sẵn trong `mlflow.genai`), tỉ lệ kết luận grounded (mỗi root cause phải map được tới evidence) |
| Ngưỡng đi tiếp | Thắng baseline ở task success và không thua ở ToolCallCorrectness |
| Deploy | Đăng ký version vào MLflow Model Registry → đổi model trên vLLM (giữ bản cũ chạy song song để so sánh) → chạy `make eval-real` + `make eval-genai-real` trên bản mới trước khi chuyển traffic |

### 1.3 Track C — Tool-use (lát cắt của LLM chính)

| Mục | Nội dung |
| --- | -------- |
| Mục tiêu | Kỷ luật gọi tool: đúng tool, đúng tham số, không gọi ngoài phạm vi |
| Phương pháp | Cùng quy trình SFT như track B, nhưng dataset chỉ gồm các lượt gọi tool (context + tool kỳ vọng + tham số kỳ vọng), kèm ca "không được gọi tool" và ca "bẫy" (yêu cầu ngoài mission) |
| Dữ liệu | `ToolUseExample`: `context`, `available_tools`, `expected_tool`, `expected_args`. Nguồn: log tool-call thật + ca bẫy viết tay từ registry (`runtime/tools/registry.py`) |
| Metric | `correct-tool rate` (đúng tên tool), `correct-args rate` (đúng cả tham số), `out-of-scope rate` (gọi tool không có trong `available_tools`) |
| Ngưỡng đi tiếp | `correct-tool rate` và `correct-args rate` thắng baseline; `out-of-scope rate` phải bằng 0 trên bộ test bẫy |
| Deploy | Đi cùng bản LLM chính (không deploy riêng), nhưng có gate riêng: nếu tool-use thua baseline thì cả bản LLM bị chặn deploy |

## 2. Sáu giai đoạn (áp cho cả ba track)

| # | Giai đoạn | Vào (entry) | Ra (exit) |
| - | --------- | ----------- | --------- |
| 1 | **Thu thập dữ liệu** | Chưa có dataset cho track | Có hợp đồng dữ liệu + hook thu thập đang chạy; dữ liệu thật bắt đầu tích lũy |
| 2 | **Chuẩn hóa dataset** | Dữ liệu thô đã tới | Dataset đã làm sạch, khử trùng, chia train/val/test, version trong MLflow; schema validate pass 100% |
| 3 | **Train** | Dataset đã version + config đã chốt | Model artifact + run MLflow đầy đủ (params, metrics, dataset version) |
| 4 | **Eval offline** | Model artifact + bộ test cố định | Báo cáo so sánh với baseline; chỉ đi tiếp khi thắng baseline (mục 1) |
| 5 | **Registry + deploy** | Eval pass | Version trong Model Registry; deploy song song; `make eval-real` pass trên bản mới |
| 6 | **Giám sát + retrain** | Model đang phục vụ | Dashboard metric theo dõi (trace, tỉ lệ lỗi, phản hồi); trigger retrain khi chất lượng tụt hoặc đủ dữ liệu mới |

### Giai đoạn 1 — Thu thập dữ liệu (việc làm ngay, khi chưa có dữ liệu)

- **Hợp đồng dữ liệu** (`training/common/schemas.py`): mỗi track có một
  schema Pydantic, mọi dataset (tổng hợp hay thật) đều phải validate qua
  schema này. Schema là luật, không phải gợi ý.
- **Hook thu thập** (`training/common/collect_from_traces.py`): chuyển kết
  quả execution (định dạng của `scripts/eval_mission.py`) thành ví dụ
  SFT/ToolUse. Hiện tại hook map được kết quả local; khi execution thật
  chạy qua MLflow traces, hook sẽ map trực tiếp từ trace (tool call thật,
  tham số thật, kết quả thật).
- **Dữ liệu tổng hợp** (`training/common/synthetic.py`): 20 ví dụ/track,
  miền thang máy O21, ghi rõ `source: "synthetic"`. Chỉ dùng để chạy thử
  pipeline, không dùng để kết luận chất lượng.

### Giai đoạn 2 — Chuẩn hóa dataset

- Làm sạch (bỏ record lỗi, bỏ run `FAILED`), khử trùng (cùng query +
  cùng kết quả), chia train/val/test theo thời gian (test = dữ liệu mới
  nhất, tránh rò rỉ).
- Mỗi dataset version được log lên MLflow (artifact JSONL + hash + số
  lượng + tỉ lệ split). Train luôn ghi rõ dataset version đã dùng.

### Giai đoạn 3 — Train

- Embedding: `training/embedding/train.py` (sentence-transformers,
  `MultipleNegativesRankingLoss`).
- LLM: `training/llm/train_sft.py` (TRL `SFTTrainer` + PEFT LoRA, config
  ở `training/configs/llm_sft.yaml`).
- Mọi run log params/metrics/artifact lên MLflow. Train cần GPU (xem
  mục 6, quyết định còn mở). Không có GPU thì chỉ chạy được eval và
  sinh dữ liệu, không train.

### Giai đoạn 4 — Eval offline

- Embedding: `training/embedding/eval_retrieval.py` (recall@k, nDCG@k,
  so với baseline).
- Tool-use: `training/llm/eval_tool_use.py` (correct-tool / correct-args /
  out-of-scope rate).
- LLM chính: dùng lại `scripts/eval_mission.py` (`--real` + `--genai`)
  trên model mới, so với model gốc trên cùng bộ case.

### Giai đoạn 5 — Registry + deploy

- Đăng ký model version vào MLflow Model Registry (tên + version + dataset
  version + eval report).
- Deploy qua vLLM (người dùng tự cấu hình LiteLLM/vLLM; repo chỉ cung cấp
  tên model + cách đổi). Giữ bản cũ chạy song song, so sánh bằng eval thật
  rồi mới chuyển traffic. Rollback = đổi lại tên model cũ trong cấu hình.
- Riêng embedding: đổi model **bắt buộc** re-index Qdrant, nên deploy
  embedding đi kèm một lần re-index có kiểm tra (recall@k sau re-index
  không thấp hơn trước).

### Giai đoạn 6 — Giám sát + retrain

- Theo dõi: trace MLflow (tỉ lệ lỗi execution, tool-call sai), eval
  định kỳ trên bộ test cố định, phản hồi người dùng.
- Trigger retrain: chất lượng tụt dưới ngưỡng, hoặc đủ dữ liệu mới
  (ngưỡng số lượng do người quyết khi có dữ liệu đầu tiên), hoặc có
  thiết bị/miền mới cần hỗ trợ.

## 3. Hợp đồng dữ liệu (schema)

Định nghĩa đầy đủ trong `training/common/schemas.py`. Tóm tắt:

- **EmbeddingExample** — `query` (câu hỏi), `positive` (tài liệu đúng),
  `negatives` (danh sách tài liệu sai), `source`, `split`.
- **SFTExample** — `messages` (system/user/assistant/tool, kèm
  `tool_calls` khi assistant gọi tool), `mission_id`, `source`, `split`.
- **ToolUseExample** — `context`, `available_tools`,
  `expected_tool` (null = không được gọi tool), `expected_args`,
  `mission_id`, `source`, `split`.

Mọi file JSONL phải pass `validate_jsonl()` trước khi train/eval.
Dữ liệu thật về sau phải qua đúng schema này; record nào không khớp
schema thì bị loại và ghi log, không được "sửa lặng" cho khớp.

## 4. Trạng thái hiện tại (2026-11-09)

Đã có:

- Tài liệu này + scaffold `training/` (schemas, dữ liệu tổng hợp,
  train/eval cho cả ba track, configs, hook thu thập).
- MLflow 3.18 làm observability chính (trace, eval, Prompt Registry);
  `scripts/eval_mission.py` đã có sẵn để đo baseline của LLM chính.
- Qdrant + vLLM/LiteLLM đã chạy (người dùng tự vận hành).

Chưa có:

- Dữ liệu hoạt động thật (dataset thật = 0). Chưa train lần nào.
- GPU train (môi trường build không có GPU; `torch` chưa được cài).
- Base model đã chốt cho embedding và LLM chính (mục 6).
- Bộ test cố định cho retrieval và tool-use (sẽ dựng từ dữ liệu thật
  đầu tiên; hiện chỉ có bộ tổng hợp để chạy thử pipeline).

## 5. Quy tắc deploy / rollback

1. Không deploy model chưa thắng baseline trên bộ test cố định.
2. Deploy song song với bản cũ; chỉ chuyển traffic sau khi eval thật pass.
3. Rollback bằng cách đổi lại tên model trong cấu hình LiteLLM/vLLM
   (và, với embedding, trỏ lại collection cũ nếu đã re-index sang
   collection mới — khuyến nghị re-index sang collection mới, giữ
   collection cũ nguyên cho tới khi bản mới được xác nhận).
4. Mọi lần deploy/rollback đều ghi lại: model version, dataset version,
   eval report, thời gian, người thực hiện.

## 6. Quyết định còn mở (cần người dùng)

| # | Quyết định | Gợi ý / ví dụ (chưa chốt) |
| - | ---------- | -------------------------- |
| 1 | Base model embedding | Model đang dùng trên LiteLLM, hoặc `BAAI/bge-m3` (đa ngữ, tốt với tiếng Việt) — cần chốt tên chính xác |
| 2 | Base model LLM chính | Model đang chạy trên vLLM của người dùng — cần chốt tên chính xác |
| 3 | Train ở đâu (GPU) | Máy người dùng (GPU gì?), thuê GPU cloud, hay train LoRA nhẹ trên CPU/GPU nhỏ |
| 4 | Nguồn dữ liệu thật đầu tiên | Trace MLflow, log execution, tài liệu kỹ thuật nào, phản hồi người dùng |
| 5 | Ngưỡng thắng baseline | Đề xuất +3 điểm recall@5 / nDCG@5 (embedding); task success không thua (LLM) — chốt khi có số liệu đầu tiên |
| 6 | Tần suất retrain | Theo trigger (mục 2, giai đoạn 6) hay theo lịch cố định |

## 7. Lộ trình gợi ý

1. **Bây giờ** — chốt 6 quyết định trên (ít nhất là 1–3); hook thu thập
   chạy kèm runtime để dữ liệu thật bắt đầu tích lũy.
2. **Khi có dữ liệu thật đầu tiên** — dựng bộ test cố định, đo baseline
   (model gốc) và ghi lại số liệu gốc.
3. **Track A trước** — embedding rẻ và nhanh nhất, hiệu quả thấy ngay ở
   retrieval; re-index Qdrant một lần.
4. **Track C** — tool-use SFT trên trace thật, gate riêng cho deploy.
5. **Track B** — SFT LLM chính khi đã có đủ trace sạch; LoRA trước.
6. **Vận hành** — giám sát + retrain theo trigger.
