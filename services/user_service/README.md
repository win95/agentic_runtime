# User service

Independent service for user management: register, login (JWT), and
admin user management. Same repo, fully self-contained — it never
imports the platform app (`app/`), owns its own tables and its own
Alembic chain, and connects to the platform only through one contract:
the JWT it signs, the platform API verifies.

## Run locally

```bash
pip install -r services/user_service/requirements.txt
# Database: own DB on the shared Postgres server (see below).
# JWT_SECRET must match the platform API's JWT_SECRET.
uvicorn services.user_service.main:app --host 0.0.0.0 --port 8001
```

Or with Docker: `docker compose up -d --build user-service` (the
`user-service` container in docker-compose.yml).

## Database

Own database `user_service` on the shared Postgres server — never the
platform's `agentic_runtime` database. On a fresh compose volume the
`docker/postgres/01-create-user-service-db.sql` init script creates it;
on an existing volume (or local Postgres) create it once yourself:

```bash
createdb user_service
```

Migrations (own chain, own version table
`alembic_version_user_service`):

```bash
make migrate-users        # alembic upgrade head, from services/user_service
```

## First admin

```bash
ADMIN_EMAIL=admin@example.com ADMIN_PASSWORD='...' make seed-admin
```

Idempotent — re-running promotes the user to admin if the email already
exists, and refuses to run without both values. Real credentials go in
the environment, never in the repo.

## API

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| POST | `/auth/register` | — | `{email, password}` → 201 user (open registration) |
| POST | `/auth/login` | — | `{email, password}` → `{access_token, token_type, expires_in}` |
| GET | `/auth/me` | Bearer | Current user |
| GET | `/admin/users` | Bearer + admin | List users |
| PATCH | `/admin/users/{id}` | Bearer + admin | `{is_active: bool}` (an admin cannot disable themself) |
| GET | `/health` | — | `{"status": "ok"}` |

Tokens are HS256 JWTs, 24h by default (`JWT_EXPIRE_HOURS`). The
platform API accepts the same token as `Authorization: Bearer <token>`
on `/executions` — see docs/user-service.md.

## What is deliberately NOT in the base

Refresh tokens, password reset, email verification, OAuth login,
per-user cost caps, rate limiting on login, CORS. Each is a small
follow-up on top of this base, not a redesign.
