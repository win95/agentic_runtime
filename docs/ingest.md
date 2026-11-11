# Batch ingest — Phương án B

Dữ liệu thật đi vào hệ thống theo lô (batch), không qua API public.
Một job Celery beat chạy theo lịch, đọc file JSON trong inbox, validate
từng dòng, rồi load idempotent vào 4 kho. Đây là đường production;
seed demo (`make seed`) chỉ phục vụ dev/demo và không bị thay thế.

## Luồng

```
nguồn dữ liệu (CMMS/BMS export, file thả tay)
        │  file JSON, 1 loại / file
        ▼
inbox: ingest/inbox/  (local)  hoặc  bucket MinIO ingest-inbox
        │  Celery beat, mỗi INGEST_INTERVAL_SECONDS (mặc định 30 phút)
        ▼
ingest.run.run_batch()
  ├─ tên file → loại dòng (faults, maintenance, documents,
  │   relations, sensor_readings)
  ├─ validate từng dòng (ingest/schemas.py — Pydantic + luật Lớp 1:
  │   asset_id allowlist, giới hạn độ dài, strip control chars)
  ├─ load idempotent (ingest/loaders.py):
  │     Postgres (faults / maintenance / sensor_readings): db.merge theo id
  │     Qdrant (documents): upsert theo uuid5 của id, điểm đã có bị skip
  │     Ontology canonical (relations): OntologyWriteService,
  │     rồi projection → Neo4j (MERGE)
  ├─ dòng lỗi → bảng ingest_quarantine (giữ nguyên row JSON + lỗi)
  └─ mỗi file → 1 dòng ingest_runs (SUCCEEDED / PARTIAL / FAILED)
```

## Định dạng file

Một file là một JSON array; mỗi phần tử là một dòng. Tên file quyết
định loại: phần trước dấu `-` đầu tiên trong tên, ví dụ `faults.json`,
`faults-2026-11.json`, `sensor_readings.json`. File mẫu trong
`ingest/samples/` (dữ liệu O21 tháng 11/2026 — nối tiếp seed tháng 10).

| File | Dòng | Kho đích |
|---|---|---|
| `faults.json` | id, asset_id, code, title, description?, severity (low/medium/high), occurred_at, resolved_at?, technician?, action? | Postgres `fault_history` |
| `maintenance.json` | id, asset_id, date, action, technician?, notes? | Postgres `maintenance_records` |
| `sensor_readings.json` | id, asset_id, metric (lower_snake_case), value, unit?, at | Postgres `sensor_readings` |
| `documents.json` | id, title, content (≤10000 ký tự), category? | Qdrant `technical_documents` |
| `relations.json` | from_id, from_label, relation (UPPER_SNAKE_CASE), to_id, to_label | Ontology canonical → Neo4j graph |

`resolved_at` phải sau `occurred_at`. Ngày viết ISO (`2026-11-03T09:15:00`).

## Chạy

```bash
# Thả file vào ingest/inbox rồi chạy tay:
make ingest
# Chỉ kiểm tra định dạng, không ghi gì vào kho nào:
make ingest-dry-run
# Chạy lại các dòng đang ở quarantine (sau khi sửa):
python -m ingest.run --retry
# Chạy lại cả file đã SUCCEEDED (bình thường re-run sẽ skip):
python -m ingest.run --force
# Một file lẻ, không qua inbox:
python -m ingest.run --file /đường/dẫn/faults-2026-11.json
# Đọc inbox từ MinIO thay vì thư mục local:
python -m ingest.run --minio
```

Theo lịch, beat chạy `app.worker.tasks.run_ingest_batch` mỗi
`INGEST_INTERVAL_SECONDS` (mặc định 1800 = 30 phút). Beat không retry:
lịch sẽ gọi lại, và watermark (1 dòng `ingest_runs` / file) khiến
lần chạy sau skip file đã SUCCEEDED.

## Watermark & idempotency

- **Watermark**: file nào đã có `ingest_runs` SUCCEEDED thì lần chạy
  sau skip. File PARTIAL (có dòng lỗi) được chạy lại — dòng ok cũ
  upsert lại không đổi, dòng lỗi cũ đã nằm trong quarantine.
- **Idempotency**: Postgres upsert theo `id` cố định; Qdrant skip
  point đã có; canonical upsert + Neo4j MERGE (qua projection). Chạy lại cùng file không nhân đôi dữ liệu.
- **Quarantine**: dòng lỗi không bị vứt — giữ nguyên row JSON trong
  `ingest_quarantine` (status `pending`) để sửa và chạy
  `python -m ingest.run --retry`. Sửa được thì status thành `retried`.

## Lỗi thường gặp

- File không phải JSON array, hoặc tên file không thuộc 5 loại trên →
  `ingest_runs` FAILED, không dòng nào được load.
- Qdrant chưa chạy hoặc thiếu LiteLLM key/proxy → file `documents.json`
  FAILED (toàn bộ dòng vào quarantine với cùng lỗi). Sửa xong chạy lại
  file đó, hoặc `--retry`.
- MinIO: bucket `ingest-inbox` được tạo bởi service `minio-setup`
  trong compose. File trong bucket được giữ lại sau khi xử lý (chỉ
  watermark ngăn chạy lại) — dọn theo lịch giữ của bạn.

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `INGEST_INTERVAL_SECONDS` | `1800` | Chu kỳ beat chạy batch |
| `INGEST_INBOX_DIR` | `ingest/inbox` | Thư mục inbox local |
| `INGEST_USE_MINIO` | `false` | Đọc inbox từ MinIO thay vì local |
| `INGEST_MINIO_BUCKET` | `ingest-inbox` | Bucket inbox trên MinIO |
