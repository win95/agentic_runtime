# Training — vòng đời fine-tune

Scaffold cho vòng đời fine-tune 3 track (xem quy trình đầy đủ trong
[`docs/fine-tuning-lifecycle.md`](../docs/fine-tuning-lifecycle.md)).
Hiện tại **chưa có dữ liệu thật** — mọi thứ ở đây chạy được bằng dữ
liệu tổng hợp (ghi rõ `source: "synthetic"`) và sẵn sàng nhận dữ liệu
thật khi có.

## Cấu trúc

```text
training/
  README.md                  Tài liệu này
  requirements-training.txt  Deps chỉ dùng để train (torch/TRL/PEFT/...)
  configs/
    embedding.yaml           Config fine-tune embedding (model = placeholder)
    llm_sft.yaml             Config SFT LLM chính, LoRA (model = placeholder)
  common/
    schemas.py               Hợp đồng dữ liệu 3 track (Pydantic) + validate_jsonl
    synthetic.py             Sinh dữ liệu tổng hợp (20 ví dụ/track)
    collect_from_traces.py   Hook thu thập: kết quả execution -> ví dụ SFT/ToolUse
  embedding/
    train.py                 Contrastive fine-tune (sentence-transformers)
    eval_retrieval.py        Eval retrieval: recall@k / nDCG@k, so baseline
  llm/
    train_sft.py             SFT LLM chính, LoRA (TRL + PEFT)
    eval_tool_use.py         Eval tool-use: đúng tool / đúng args / ngoài phạm vi
  data/                      Dữ liệu sinh ra (gitignored, tạo bằng make)
  output/                    Model đã train (gitignored)
```

## Chạy từng bước (dữ liệu tổng hợp)

```bash
# 0. Sinh dữ liệu tổng hợp (20 ví dụ/track)
make gen-synthetic-data

# 1. Eval tool-use bằng mock predictor (không cần torch/GPU — chạy được ngay)
make eval-tool-use

# 2. Thu thập ví dụ từ kết quả execution (demo, không cần dữ liệu thật)
python -m training.common.collect_from_traces --demo --out-dir training/data

# 3. Train (CẦN torch + GPU — ở đây sẽ dừng với thông báo rõ ràng)
make train-embedding
make train-llm

# 4. Eval retrieval (CẦN sentence-transformers + model)
make eval-retrieval
```

## Cái gì chạy được ở đây, cái gì chưa

| Bước | Ở môi trường này (không GPU, không torch) | Cần gì để chạy thật |
| ---- | ---------------------------------------- | ------------------ |
| Sinh dữ liệu tổng hợp | Chạy được | — |
| Validate schema | Chạy được | — |
| Eval tool-use (mock) | Chạy được | Dự đoán thật từ model thay cho mock |
| Thu thập từ traces | Chạy được (demo) | Execution thật / MLflow traces |
| Train embedding | Dừng với thông báo rõ ràng | `pip install -r training/requirements-training.txt` + GPU + base model thật trong config |
| Train LLM (SFT) | Dừng với thông báo rõ ràng | Như trên + GPU |
| Eval retrieval | Dừng với thông báo rõ ràng | sentence-transformers + model |

## Placeholder — phải thay trước khi train thật

- `training/configs/embedding.yaml` → `base_model` (xem
  `docs/fine-tuning-lifecycle.md`, quyết định 1).
- `training/configs/llm_sft.yaml` → `base_model` (quyết định 2).
- Dữ liệu tổng hợp trong `training/data/` — thay bằng dataset thật
  (cùng schema, `source: "trace"` hoặc `"manual"`).
- Mock predictor trong `eval_tool_use.py` — thay bằng dự đoán thật.

## Quy tắc

- Không train khi chưa có dữ liệu thật. Dữ liệu tổng hợp chỉ để kiểm
  tra pipeline.
- Mọi dataset phải pass `validate_jsonl()` trước khi train/eval.
- Model fine-tune chỉ deploy khi thắng baseline trên bộ test cố định.
- Mọi run train/eval đều log lên MLflow
  (`agentic-runtime-finetune`); khi server MLflow không tới được,
  script chỉ cảnh báo và vẫn chạy tiếp (fail-soft).
