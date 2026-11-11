# Agent versioning — Phương án C

Một agent chạy bằng 4 thứ: system prompt, model/temperature, danh
sách tool được phép, và bản thân các tool (tên, mô tả, schema, quyền).
Chỉ version prompt (MLflow Prompt Registry) thì không pin được hành
vi — đổi một dòng mô tả tool cũng đổi cách agent chọn tool, mà thay
đổi đó nằm trong git, không gắn với prompt version nào.

Phương án C version **toàn bộ agent config như một đơn vị**: mỗi
`AgentVersion` là một snapshot bất biến gồm prompt text chính xác,
MLflow prompt version (nếu có), provider/model/temperature, allowed
tools, giới hạn đọc session-history (`session_history`, NULL = dùng
cap toàn cục), và tool manifest (tên, mô tả, capability, permission, input
schema, implementation, timeout, retry, read_only) kèm SHA-256.
`git_sha` ghi lại code lúc snapshot (NULL khi chụp ngoài git checkout).

## Deploy / rollback

```bash
# Sửa YAML agent rồi deploy thành version mới (và active luôn):
python scripts/register_agent_version.py --agent-id admin_bot
python scripts/register_agent_version.py --all

# Tạo version nhưng chưa active:
python scripts/register_agent_version.py --agent-id fault_investigator --no-activate

# Deploy kèm đăng ký prompt lên MLflow trước:
python scripts/register_agent_version.py --agent-id admin_bot --register-prompt
```

`make seed` / `python -m seeds.load_missions` cũng tạo version: lần
đầu là v1, seed lại không đổi thì không tạo thêm, sửa YAML rồi seed
lại thì thành v2 và active luôn. Tạo version từ snapshot giống hệt
version đang active là no-op (không tạo dòng mới).

Rollback / kích hoạt qua API (cần `X-API-Key`):

```bash
GET  /agents/{agent_id}/versions            # danh sách, mới nhất trước
GET  /agents/{agent_id}/versions/{version}  # một version
POST /agents/{agent_id}/versions/{version}/activate
```

## Runtime dùng version như thế nào

`run_mission` resolve active version của `mission.agent_id` một lần
trước khi rẽ nhánh (best-effort — DB lỗi hay chưa có version thì
chạy như cũ: model từ env, prompt từ registry/fallback, không có
cổng allowed_tools); `deep_agent_node` dùng lại version đó theo số
version, chỉ resolve lại khi node được gọi trực tiếp mà state chưa
có version. Sau đó node làm như sau, theo thứ tự:

1. Lấy version đã resolve ở trên (hoặc tự resolve khi chưa có).
2. Prompt chạy = `version.prompt_text` (text đã pin, không lookup
   MLflow lúc chạy — version vẫn tái lập được khi MLflow down).
3. Model chạy = `version.model` + `version.temperature` (build qua
   `build_chat_model` với config override; base URL/key vẫn từ env).
   `version.model` lấy từ đâu: YAML agent ghi `model: ${LLM_MODEL}`,
   lúc seed loader expand từ `.env` (không đặt thì lấy
   `settings.llm_model`, mặc định `gpt-4o-mini`) rồi pin vào snapshot.
   Đổi model = đổi `LLM_MODEL` trong `.env` rồi seed lại → version
   mới; version cũ vẫn giữ model cũ nên rollback được.
4. `version.allowed_tools` đưa vào context nên `registry.check()`
   thực thi nó — trước đây allowed_tools trong YAML chỉ là metadata,
   giờ nó là cổng chặn thật (tool ngoài danh sách không tới được agent,
   kể cả tool MCP).
5. Drift: hash manifest live so với `manifest_sha256` đã pin. Khác
   nhau thì log warning và ghi `agent_version_drift=true` — run vẫn
   chạy tiếp bằng tool live (tool là code, không chạy được code cũ),
   nhưng drift được ghi nhận để biết lần chạy đó không còn đúng
   version đã pin.

Mỗi execution ghi lại `agent_version` (version đã chạy) và
`agent_version_drift` trên dòng `executions` — xem lại được lần chạy
nào dùng config nào. Version được resolve một lần trong `run_mission`
trước khi rẽ nhánh theo mode, nên cả run hybrid chỉ chạy capability
(đủ evidence nên không escalate) cũng ghi version đã active.
`agent_version` là NULL khi không có version nào active, hoặc mission
không khai báo agent. NULL nghĩa là "không có version / không có
agent", không phải "không chạy Deep Agent". Agent vẫn tra lại được qua
`execution.agent_id` (lưu lúc tạo execution từ `mission.agent_id`).
`agent_version_drift` khi đó là false (không có version để so drift).

## Quan hệ với MLflow Prompt Registry

`scripts/register_prompts.py` vẫn là nơi đăng ký text prompt lên
MLflow (để review/rollback prompt riêng lẻ). AgentVersion ghi lại
`prompt_registry_version` của prompt đó (best-effort, chỉ khi chạy
CLI với `--register-prompt`). Runtime không phụ thuộc MLflow để lấy
prompt nữa — text đã pin trong version. Muốn đổi riêng prompt mà
không đụng model/tool: sửa text, chạy `register_agent_version.py`
(hoặc seed lại) để thành version mới.

## Giới hạn

- Manifest chỉ gồm tool built-in (fault + admin). Tool MCP phụ thuộc
  môi trường (cần MCP server đã cấu hình) nên không nằm trong manifest;
  chúng vẫn bị chặn bởi `allowed_tools` của version.
- Drift không chặn run — nó chỉ ghi nhận. Muốn chặn hẳn thì cần quyết
  định sản phẩm (fail-closed) riêng.
- Version không lưu skill selection — skills vẫn chọn động theo mission
  (xem README §14).
