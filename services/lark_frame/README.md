# Lark frame — webhook / WebSocket front door for agent_lark_me

The frame only verifies + routes + enqueues, then returns 200
immediately. It never calls `run_mission` directly and never sends the
final answer — the agent replies via its Lark tools (D-LARK-09).
Full plan: `~/workspace/goals/agentic-runtime-base-project/files/ke-hoach-agent-lark-me/ke-hoach-agent-lark-me.md`.

## Run

```bash
# from the repo root, with the repo .venv
LARK_APP_ID=cli_... LARK_APP_SECRET=... \
  LARK_VERIFICATION_TOKEN=... LARK_ENCRYPT_KEY=... \
  PLATFORM_BASE_URL=http://localhost:8000 PLATFORM_API_KEY=... \
  LARK_ALLOWLIST=<open_id tran> \
  python -m services.lark_frame
```

The frame also loads the repo-root `.env` at startup (same file the
platform reads — `LARK_APP_ID`, `LARK_APP_SECRET`, `PLATFORM_API_KEY`,
`LARK_ALLOWLIST`, ...), so the vars above can live there instead of
on the command line. Shell env wins over file values; a missing file
is silently ignored.

- Webhook mode (default): serves `POST /lark/events` + `POST /lark/card`
  on `FRAME_HOST:FRAME_PORT` (default `0.0.0.0:8000`). Register the two
  URLs in the Lark console (Event Subscriptions → Request URL, Bot →
  Callback Configuration).
- WebSocket mode: `LARK_TRANSPORT=websocket` — outbound WS, no public
  URL needed. The console event subscription stays enabled, but the
  two URLs are not registered.

## Environment

| Var | Required | Notes |
| --- | --- | --- |
| `LARK_APP_ID` / `LARK_APP_SECRET` | yes | The agent's own Custom App (Step 1). Never in chat, never in the repo. |
| `LARK_VERIFICATION_TOKEN` | webhook | For the `url_verification` challenge. |
| `LARK_ENCRYPT_KEY` | webhook | Only when the app encrypts events; then the SHA256 signature is also verified. |
| `PLATFORM_BASE_URL` | yes | Platform API, default `http://localhost:8000`. |
| `PLATFORM_API_KEY` | yes | The platform's `API_KEY` (D-LARK-10). Never in chat, never in the repo. |
| `LARK_ALLOWLIST` | yes | Comma-separated sender open_ids (D-LARK-06). Empty = everyone refused. |
| `LARK_BOT_OPEN_ID` | no | Override for @bot detection; unset = fetched via `bot/v3/info` at startup, loose mode (any mention routes) when that fails. |
| `LARK_TRANSPORT` | no | `webhook` (default) or `websocket`. |
| `FRAME_HOST` / `FRAME_PORT` | no | Default `0.0.0.0:8000`. |
| `FRAME_HEALTH_PORT` | no | Health-only server in `websocket` mode, default `8002` (`FRAME_PORT` 8000 is taken by the platform). |
| `REDIS_URL` | no | Durable dedupe (seen-set). Unset = in-memory (lost on restart). |

## Health

- `GET /healthz` — liveness: `{"status": "ok", "transport": ..., "uptime_seconds": N}`.
- `GET /readyz` — 200 only when the platform's `GET /health` answers.
- In `websocket` mode the two endpoints run on a separate health-only server (`FRAME_HEALTH_PORT`, default `8002`), started before the WS connects — so `/healthz` answers even while the WS is down/reconnecting. It serves no POST routes (`/lark/events` is 404 there).

## Files

| File | What |
| --- | --- |
| `routing.py` | Pure routing logic (no I/O): allowlist, @bot, rate limit, session id, seen-set. Shared by both transports. |
| `platform.py` | Outbound HTTP (urllib): sync user, enqueue, platform health, the frame's own Lark sends, bot identity. |
| `webhook.py` | `ThreadingHTTPServer` + verification (SHA256, `app_id`, token, AES decrypt) + card dispatch. |
| `channel.py` | WS mode via `lark_oapi.channel.FeishuChannel` (legacy, supported until 2027-06-02). |
| `__main__.py` | Env → config → transport switch. |

## What the frame sends (only these)

1. The refusal (outsider, 1/hour/user).
2. The rate-limit notice (first drop in a window).
3. The platform-down notice (first in 5 min/user).
4. The echo test (Step 2 verify).

Every other reply comes from the agent, via its Lark tools.
