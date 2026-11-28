# User service

Independent service for user management: register, login (JWT), admin
user management. Same repo, fully self-contained — see
`services/user_service/README.md` for the run instructions.

## Why a separate service

Auth and the user lifecycle are a different failure domain and release
cadence from the agent runtime. The user service can deploy, change its
schema, and add features (refresh tokens, password reset, OAuth) without
touching the path FastAPI → Celery → LangGraph. The runtime keeps its
own auth contract (API key for service calls) and only learns one new
fact: a request may also carry a user JWT.

## Independence contract

The service never imports the platform app, and the platform never
imports the service. The shared surface is exactly three things:

1. **The Postgres server** — not the database: the service owns database
   `user_service`; the platform owns `agentic_runtime`. Separate tables,
   separate Alembic chains (`alembic_version_user_service` vs
   `alembic_version_agentic_runtime`), separate version tables because
   MLflow already owns the default `alembic_version` table.
2. **The JWT secret** (`JWT_SECRET` env var, identical in both services).
   The service signs HS256 tokens; the platform verifies them. Nothing
   else is shared — no code, no models, no session.
3. **The token payload**: `{"sub": user_id, "email", "role", "iat",
   "exp"}`. The platform reads `sub` only.

## How a user reaches the agent

1. `POST /auth/login` on the user service (port 8001) → access token
   (24h by default).
2. `POST /executions` on the platform API (port 8000) with
   `Authorization: Bearer <token>` → the run is created with
   `user_id` set from the token's `sub`.
3. `GET /executions`, `GET /executions/{id}`, `GET
   /executions/{id}/result`, `POST /executions/{id}/cancel` filter by
   that `user_id`: a logged-in user only ever sees their own runs. A
   request for another user's execution returns 404 (not 403 — the API
   never confirms the other run exists).
4. Service-to-service calls keep using `X-API-Key` as before; API-key
   calls see all executions and leave `user_id` NULL.

The platform's auth dependency is `require_api_key_or_user`
(`app/core/security.py`): JWT first, API key second, 401 when neither
is valid. An invalid Bearer token is a hard 401 — it never falls
through to the API-key path.

## Lark sync — one user across Lark apps (union_id)

The agent (`agent_lark_me`) reaches users through Lark, not through
password login. On every routed Lark message the frame resolves the
sender lazily: `POST /lark/users/sync` on the platform (service API
key) → `GET /admin/users/by-identity` here, or `POST /admin/users`
to create the user with the synthetic email `lark-<open_id>@lark.local`
and a random unusable password.

The catch: a Lark `open_id` changes per app, so a user chatting via a
new app would come back as a brand-new platform user. The
`union_id` is stable across apps of the same tenant, so sync resolves
by union first, then by open_id — and whichever identity resolved the
user gets the other one linked to it. The union identity is stored as
provider `"lark"`, external_id `"union:<union_id>"`, linked with
`POST /admin/users/{user_id}/identities` (service-key only, 201 new /
200 already linked / 409 if the identity belongs to a different user —
never merge two accounts silently). The frame sends the union_id
whenever the Lark event carries it; without it, sync is exactly the
old open_id-only flow.

## What the base does NOT do

- **Rate limiting on login and register.** `POST /auth/login` giới
  hạn 5 lần/phút theo (email, IP) — key theo cả hai để kẻ tấn công
  không khóa được nạn nhân từ IP khác; giới hạn chạy TRƯỚC khi kiểm
  tra mật khẩu nên 429 không lộ email có tồn tại hay không.
  `POST /auth/register` giới hạn 5 lần/phút theo IP. Cả hai dùng
  Redis (db 2, `REDIS_URL`; xem `services/user_service/rate_limit.py`).
  Redis lỗi thì fail-open (ghi log) — giới hạn là rào chắn, không
  phải auth.
- **Sessions are user-bound** since the guest API demo
  (docs/guest-api.md): a session belongs to one user, and loading
  another user's session is a 404. The execution carries `user_id`,
  so the audit trail exists.
- **No refresh tokens, password reset, email verification, OAuth,
  per-user cost caps, or CORS.** Each is a small follow-up on top of
  this base. The per-user cost cap is Layer 7 of the security & cost
  roadmap (docs/security-cost-roadmap.md) — this base supplies the
  `user_id` it needs to be enforced against.
- **No tenant separation beyond user_id.** The platform still runs one
  tenant (`"default"` in the mission context); `user_id` is ownership,
  not a tenant boundary.

## Setup checklist (after pulling)

1. Create the service database (fresh compose volume does this via
   `docker/postgres/01-create-user-service-db.sql`; otherwise
   `createdb user_service`).
2. `make migrate-users` (user-service chain) and `make migrate`
   (platform chain, migration 0010 adds `executions.user_id`).
3. Set the same `JWT_SECRET` in both services' environments (see
   `.env.example`). Empty secret = JWT auth is not configured: login
   returns 503 and Bearer tokens are ignored.
4. `make seed-admin` with `ADMIN_EMAIL` / `ADMIN_PASSWORD` set (once).
5. Restart the API worker/container (code change) and start the
   user service (`make run-user-service` or `docker compose up -d
   user-service`).

## Verification

- Service tests: `pytest services/user_service/tests` (sqlite, no
  Postgres needed).
- Platform integration: `tests/test_user_auth.py` — JWT verification,
  execution ownership, API-key compatibility.
- Live: register → login → `POST /executions` with the Bearer token →
  the response carries `user_id`; `GET /executions` with another
  user's token does not list it.
