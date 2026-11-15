# Document Management (base)

Quản lý tài liệu: file gốc nằm trên MinIO, metadata + versioning +
ACL nằm trên PostgreSQL. Retrieval vẫn đọc Qdrant như cũ — base
này chưa nối retrieval hay ingest vào ACL (xem cuối file).

## Thiết kế

| Tầng | Chi tiết |
|---|---|
| MinIO | Bucket `documents` (`MINIO_DOCUMENTS_BUCKET`, mặc định `documents`). Một object cho một version, bất biến: `docs/{doc_id}/v{n}/{filename}`. Upload mới là object mới, không ghi đè. |
| PostgreSQL | Bảng `documents` — một row cho một version (`doc_id`, `version`, `title`, `filename`, `object_key`, `content_type`, `content_hash` sha256, `size_bytes`, `owner_user_id`, `asset_id`, `created_at`). Unique `(doc_id, version)`. Bảng `document_acls` — grant hiển (`doc_id`, `principal_user_id`, `permission` read/write). Migration `0012`. |
| Service | `app/services/document_service.py`: `upload_document` (v1), `upload_new_version`, `get_document` (latest), `get_version`, `download`, `list_documents`, `grant_access`, `check_access`. |
| API | `app/routers/documents.py`, prefix `/documents`, auth `require_api_key_or_user` (JWT user hoặc service API key). |

## ACL — fail-closed, không lộ sự tồn tại

Quy tắc trong `check_access` (service là nguồn chân lý duy nhất,
router chỉ dịch `None` → 404):

- JWT user đọc/ghi khi là owner (`owner_user_id` == JWT `sub`) hoặc
  có grant hiển. Grant `write` bao hàm `read`.
- Service-key call (không có JWT) chỉ đọc/ghi document
  service-owned (`owner_user_id` IS NULL). Không có wildcard
  principal, không có bypass cho service.
- Thiếu và bị từ chối là một: mọi getter trả `None` cho cả hai
  trường hợp, API trả 404 — không bao giờ 403, không bao giờ xác
  nhận document của người khác tồn tại.
- Chỉ owner được cấp grant (service-key chỉ cho document
  service-owned). Grant chỉ thêm, base chưa có revoke.

Upload: JWT user → `owner_user_id = sub`; service-key →
`owner_user_id = NULL`. MinIO ghi TRƯỚC row DB (lỗi MinIO thì
không có row trỏ tới object thiếu); lỗi DB sau khi put thì object
mồ côi, không tới được vì không có row. Giới hạn upload 50 MB
(`MAX_CONTENT_BYTES`).

## API

| Method | Endpoint | Ghi chú |
|---|---|---|
| POST | `/documents` | multipart `file` (+ `title`, `asset_id` form) → 201, version 1 |
| GET | `/documents` | latest version của mọi document caller đọc được |
| GET | `/documents/{doc_id}` | metadata version mới nhất |
| GET | `/documents/{doc_id}/download` | bytes của version mới nhất |
| POST | `/documents/{doc_id}/versions` | multipart `file` → version mới (cần quyền write) |
| POST | `/documents/{doc_id}/grants` | body `{principal_user_id, permission}` — chỉ owner |

## Chưa có trong base

- Ingest: document JSON trong inbox vẫn đi thẳng vào Qdrant, chưa
  đăng ký file gốc vào Document Management.
- Retrieval: hook `retrieval_context` và tool
  `search_technical_documents` chưa kiểm tra ACL — chunk Qdrant
  vẫn trả cho mọi caller. Nối ACL vào retrieval là bước tiếp theo
  khi chốt identity cho agent (agent chạy bằng service key hay JWT
  của user nào).
- Không có presigned URL (download trả bytes qua API), không có
  revoke grant, không có xóa document (version bất biến — xóa là
  quyết định riêng, chưa làm).
