# Sandbox backend — one OpenSandbox sandbox per session

By default the Deep Agent's file tools (`ls`, `read_file`, `write_file`,
`edit_file`, `delete`, `glob`, `grep`) run on deepagents' in-memory
`StateBackend`: files live only in the agent's LangGraph state and
there is no shell (`execute` is not offered). This page documents the
opt-in alternative: a self-hosted OpenSandbox sandbox per session,
so the agent can write and run real analysis code in an isolated
container.

## When a sandbox is used

All three must hold, otherwise the run uses the in-memory backend
exactly as before:

1. `SANDBOX_ENABLED=true` in `.env` (master switch, default false),
   plus `SANDBOX_API_KEY` set (the server refuses to start without
   one, so an empty key means "no server to talk to").
2. The mission opts in: `execution.sandbox: true` in its YAML
   (today only `fault_analysis` does).
3. The execution has a `session_id`.

`admin_bot` / `system_admin` never opts in, and session-less runs
never get a sandbox. A sandbox failure (server down, dead sandbox,
create failure) never fails a run: the node falls back to the
in-memory backend and writes a note on the execution
(`Sandbox unavailable — in-memory file backend`); a working sandbox
is noted as `Sandbox backend: <sandbox_id>`.

## What runs where

- **Inside the sandbox:** only the Deep Agent's filesystem tools and
  `execute` — the agent writes analysis scripts and report files
  there and runs them.
- **Host-side (unchanged):** every business tool — Postgres fault /
  maintenance / sensor queries, Neo4j asset relations, Qdrant
  documents — stays in the tool registry. The sandbox never receives
  credentials, API keys, or database access, and it starts with no
  egress (the agent works from data the host-side tools already
  fetched).

The sandbox image MUST contain python3: deepagents' own file helpers
(`read`, `grep`, `glob`) run a small Python script inside the
sandbox. Default image is `python:3.12-slim`
(`SANDBOX_IMAGE`).

## Lifecycle

- **Create:** the first opted-in execution of a session creates one
  sandbox (`SandboxSync.create`) and stores the mapping in Postgres
  (`session_sandboxes`: `session_id` -> `sandbox_id`, plus agent /
  mission / image / `last_used_at`).
- **Reuse:** later executions of the same session re-attach by id
  (`SandboxSync.connect`) and touch `last_used_at`. If the old
  sandbox is unreachable server-side, a fresh one is created and the
  row is rewritten (the old id is not resurrected).
- **Reap:** Celery beat runs `app.worker.tasks.reap_idle_sandboxes`
  every `SANDBOX_IDLE_TTL_SECONDS` (default 3600). It destroys
  sandboxes idle longer than the TTL and marks their rows
  `destroyed`; a failed destroy stays `active` for the next reap.
- **Files are ephemeral past the TTL.** Anything the agent must keep
  beyond the session belongs in the execution result (summary /
  evidence) or the host-side report store — not in the sandbox. The
  reap does not download files first.

## Setup (after pulling)

1. Start the OpenSandbox server (Docker first):
   `uvx opensandbox-server init-config ~/.sandbox.toml --example docker`,
   then `uvx opensandbox-server` (listens on localhost:8080, API key
   required). Kubernetes later: Helm charts from a checkout of a
   `release-X.Y.Z` tag, ingress gateway required, server is
   single-replica (no general HA).
2. `pip install -r requirements.txt` (adds `opensandbox==1.2.0`).
3. `alembic upgrade head` (migration `0007`: `session_sandboxes`).
4. `make seed` (picks up `sandbox: true` in `fault_analysis.yaml`).
5. In `.env`: `SANDBOX_ENABLED=true`,
   `SANDBOX_API_KEY=<the server key>` (secret — `.env` only, never
   in chat or the repo), `SANDBOX_DOMAIN=localhost:8080` (or the
   server's address). If the server runs in Docker and the worker on
   the host, set `SANDBOX_USE_SERVER_PROXY=true`.
6. Restart the worker (and beat, for the reap task).

Verify without an LLM: `GET /executions` notes — a sandbox run
shows `Sandbox backend: <id>`; with the server down the same run
still succeeds with the `Sandbox unavailable` note.

## Code map

- `runtime/sandbox/opensandbox_backend.py` — `OpenSandboxBackend`
  (`BaseSandbox` subclass: `execute` / `upload_files` /
  `download_files` / `id`; everything else is `BaseSandbox` on top
  of `execute`). The `opensandbox` import is lazy, so the module
  imports without the SDK.
- `runtime/sandbox/provider.py` — the only place that touches the
  SDK (`create` / `connect` / `destroy` + `ConnectionConfigSync`);
  `OpenSandboxClient` is the injectable seam for tests.
- `app/services/sandbox_service.py` — resolve / destroy / reap over
  the `session_sandboxes` rows.
- `app/models.py` (`SessionSandbox`) + `alembic/versions/0007_*`.
- `runtime/mission_graph.py` (`deep_agent_node`) — resolves the
  backend per run and passes it to `build_deep_agent(backend=...)`;
  `runtime/deep_agent.py` — `backend=None` means the default
  `StateBackend`.
- `app/worker/tasks.py` + `app/worker/celery_app.py` — the reap beat
  task.

## Limits (known, not yet built)

- No artifact export: the reap destroys the sandbox without
  downloading files first. A future step can export a session's
  files (`download_files()`) on reap or on demand.
- No per-sandbox egress policy yet: the sandbox starts with the
  server default (no egress configured by us). Credential Vault is
  not used — the sandbox gets no credentials at all.
- Default Docker deployment is plain-container isolation
  (gVisor/Kata/Firecracker are opt-in server config, not default).
- Not verified live: no real sandbox run has happened yet (no server
  here, no Docker daemon in this environment). The wiring is covered
  by `tests/test_sandbox_backend.py`,
  `tests/test_sandbox_service.py`, `tests/test_sandbox_wiring.py`
  (fake SDK handle + sqlite).
