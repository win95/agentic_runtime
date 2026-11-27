# Cloudflare Workers AI as an LLM provider

`LLM_PROVIDER=cloudflare` routes every agent LLM call and every
embedding through Cloudflare Workers AI instead of the LiteLLM proxy.
No proxy process is needed. The factory (`runtime/llm_factory.py`)
builds the same `langchain_openai` clients, pointed at Workers AI's
OpenAI-compatible endpoint:

```
https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1
```

## Setup (.env on your own machine)

| Var | Value |
| --- | --- |
| `LLM_PROVIDER` | `cloudflare` |
| `CLOUDFLARE_ACCOUNT_ID` | Your Cloudflare account id. **Not a secret** — it is in your dashboard URLs. |
| `CLOUDFLARE_API_TOKEN` | A Workers AI API token (dash → AI → Workers AI → Use REST API → Create API Token). **Secret — paste it into your own `.env` only, never commit it, never send it in chat.** |
| `LLM_MODEL` | `@cf/qwen/qwen3.8-27b` |
| `EMBEDDING_MODEL` | `@cf/qwen/qwen3-embedding-0.6b` |

Create the token from the Workers AI page; it carries the Workers AI
permission only. It cannot list accounts (`GET /client/v4/accounts`
returns an empty list) — that is expected, not an error.

## What it changes

- All agents use the provider through the factory: chat model,
  embeddings, retrieval expansion, and the eval judge. Nothing else
  in the pipeline changes.
- **Per-agent model pins still apply.** Agents whose YAML says
  `model: ${LLM_MODEL}` (hotel, lark_me, admin, fault, analytis) follow
  the env value. `agent_analytis` used to be pinned to
  `deepseek-v4-flash` (2026-11-20 trial); the pin was reverted on
  2026-11-27 because Workers AI's free plan cannot call the
  V4.1-Flash replacement (needs Workers Paid) — analytis now follows
  `${LLM_MODEL}` and runs `@cf/qwen/qwen3.8-27b` on this provider.
  After any model change, re-seed (`make seed-agent`) — it creates a
  new AgentVersion.
- `EMBEDDING_MODEL=@cf/qwen/qwen3-embedding-0.6b` produces 1024-dim
  vectors, the same dimension as the existing Qdrant collections —
  switching the embedding provider to Workers AI does **not** require
  re-indexing, as long as the model stays qwen3-embedding-0.6b.

## Provider quirks

**System message must be first.** Workers AI rejects any request
whose system message is not the first message (HTTP 400 "System
message must be at the beginning"). The runtime can legitimately
produce a second system message mid-list: the `retrieval_context`
hook injects retrieved context as a SystemMessage after the user
message, so a hybrid run sends `[system, human, system]` — found
live 2026-11-27, `fault_analysis` failed exactly this way, while
the LiteLLM proxy accepts the same request. The factory fixes it
for every provider: `runtime/llm_factory.py` builds a
`_SystemFirstChatOpenAI` that merges all system messages into one
leading message at the generation choke point
(`merge_system_messages`; string contents joined, content-block
lists concatenated). The merge is semantics-preserving — the model
sees the same system text, in one message, first.

## Measured cost (live, 2026-11-27)

Free tier: 10,000 neurons/day. Measured on this account:

- `@cf/qwen/qwen3.8-27b`, one short Vietnamese answer
  (~43 prompt + ~100 completion tokens): **~26–31 neurons, ~26 s**.
- `@cf/meta/llama-3.1-8b-instruct`, same style of call: **~1.1 neurons**.
- `@cf/qwen/qwen3-embedding-0.6b`, 3 short Vietnamese sentences:
  **~0.03 neurons**. Embedding is effectively free at this scale.

The 27B model costs ~27x the 8B one per short answer. At ~30
neurons per short turn, the free quota covers roughly 300 short
turns/day — but a full agent run is far more expensive than a short
turn: every agent call carries the whole system prompt + skills +
evidence (several thousand tokens), so one 5-mission live run costs
on the order of a few thousand neurons. On 2026-11-27 the quota was
exhausted (HTTP 429, error 4006) during the live agent runs — plan
agent testing accordingly, or use the 8B model for cheap runs.

## Switching back

```env
LLM_PROVIDER=litellm
LLM_MODEL=<model name registered on your LiteLLM proxy>
EMBEDDING_MODEL=<embedding model name registered on your proxy>
```

Then restart the worker (and the API for the eval judge). No code
change, no re-seed — the switch is env-only.
