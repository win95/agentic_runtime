"""Context window budget — Phương án A (32,000 tokens).

One source of truth for how the window is split (approved 2026-11-10).
All numbers are tokens unless a name says ``chars``.

========================  =========  ======
Component                 Tokens     Share
========================  =========  ======
System prompt             1,024      3.2%
Tool definitions          2,048      6.4%
Skills                    1,024      3.2%
Retrieved context         3,072      9.6%
Execution input           2,048      6.4%
Conversation + tool out   15,616     48.8%
Output reserve            5,120      16.0%
KV cache / agent          2,048      6.4%
------------------------  ---------  ------
TOTAL                     32,000     100%
========================  =========  ======

Enforcement (best-effort where noted, never crash a run):

- System prompt / tools / skills: measured at build time in
  ``deep_agent_node``; an overrun logs a warning and is noted on the
  execution — the run still proceeds (these are pinned config, failing
  the run would hide the real work).
- Retrieved context: ``retrieval_max_chars()`` derives the hook's char
  cap from RETRIEVAL_TOKENS (3,072 tok x 4 chars). The mission YAML
  may still override ``max_chars`` per mission.
- Execution input: already capped upstream at 8,192 bytes
  (``MAX_INPUT_JSON_BYTES``) = 2,048 tokens — this table just names
  that cap as the input bucket.
- History: ``HookMiddleware.before_model`` trims the oldest complete
  turns (AIMessage + its ToolMessages) until the turn total fits
  HISTORY_TOKENS. System/human messages are never removed and are not
  counted against the history bucket (they have their own buckets).
- Output reserve: passed as ``max_tokens`` when the chat model is
  built (``runtime/llm_factory.py``) — one response can never eat more
  than the reserve.
- KV cache: a standalone reserve per agent. The prompt side
  (system + tools + skills + retrieval + input + history) must fit in
  ``PROMPT_SIDE_TOKENS`` = 24,832; the reserve itself is never spent
  by trimming.

Token counting is an ESTIMATE (``ceil(chars / 4)``): the LiteLLM proxy
serves a custom model with no local tokenizer to count against
(tiktoken is disabled for the same reason in ``llm_factory``), so
budgets are enforced on the estimate. Vietnamese text tokenizes
denser than English, so the estimate is optimistic for Vietnamese
tool output — the trim loop re-checks before every model call, which
bounds the error to one turn.
"""

from __future__ import annotations

import json
import math

MAX_CONTEXT_TOKENS = 32_000
SYSTEM_PROMPT_TOKENS = 1_024
TOOL_DEFINITIONS_TOKENS = 2_048
SKILLS_TOKENS = 1_024
RETRIEVAL_TOKENS = 3_072
EXECUTION_INPUT_TOKENS = 2_048
HISTORY_TOKENS = 15_616
OUTPUT_RESERVE_TOKENS = 5_120
KV_CACHE_TOKENS_PER_AGENT = 2_048

# Prompt side: everything except the output reserve and the KV cache
# reserve. 1,024 + 2,048 + 1,024 + 3,072 + 2,048 + 15,616 = 24,832.
PROMPT_SIDE_TOKENS = (
    SYSTEM_PROMPT_TOKENS
    + TOOL_DEFINITIONS_TOKENS
    + SKILLS_TOKENS
    + RETRIEVAL_TOKENS
    + EXECUTION_INPUT_TOKENS
    + HISTORY_TOKENS
)

COMPONENT_BUDGETS = {
    "system_prompt": SYSTEM_PROMPT_TOKENS,
    "tool_definitions": TOOL_DEFINITIONS_TOKENS,
    "skills": SKILLS_TOKENS,
    "retrieval": RETRIEVAL_TOKENS,
    "execution_input": EXECUTION_INPUT_TOKENS,
    "history": HISTORY_TOKENS,
    "output_reserve": OUTPUT_RESERVE_TOKENS,
    "kv_cache_per_agent": KV_CACHE_TOKENS_PER_AGENT,
}

CHARS_PER_TOKEN = 4


def estimate_tokens(text) -> int:
    """Rough token count for budgeting: ceil(chars / 4). See module docstring."""
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def retrieval_max_chars() -> int:
    """Char cap for the retrieval_context hook block, from its token budget."""
    return RETRIEVAL_TOKENS * CHARS_PER_TOKEN


def _message_text(message) -> str:
    """Content of a langchain message as plain text (duck-typed)."""
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        text = content
    else:  # content blocks: join the text parts
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        text = "\n".join(parts)
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        text += json.dumps(tool_calls, default=str)
    return text


def message_tokens(message) -> int:
    """Estimated tokens of one message (content + tool calls)."""
    return estimate_tokens(_message_text(message))


def _turns(messages: list) -> list[list]:
    """Group ai/tool messages into turns (ai + its following tools).

    System/human messages are not part of any turn. A tool message with
    no preceding ai message forms a singleton turn (defensive — should
    not happen in a well-formed history).
    """
    turns: list[list] = []
    for message in messages:
        mtype = getattr(message, "type", "")
        if mtype == "ai":
            turns.append([message])
        elif mtype == "tool":
            if turns and getattr(turns[-1][0], "type", "") == "ai":
                turns[-1].append(message)
            else:
                turns.append([message])
    return turns


def trim_history(messages: list, budget_tokens: int = HISTORY_TOKENS) -> list:
    """Messages to remove so the turn total fits ``budget_tokens``.

    Removes the oldest complete turns first. System/human messages are
    never removed and are not counted (they have their own buckets).
    Returns [] when already within budget. The most recent turn is
    removed last, and only if every older turn is gone and the total
    is still over — per-tool output caps make that unreachable in
    practice.

    Returns message objects (not ids): langchain messages carry
    ``id=None`` until they enter the graph, so the caller maps them to
    ``RemoveMessage`` from the state, where ids exist.
    """
    turns = _turns(messages)
    if not turns:
        return []
    totals = [sum(message_tokens(m) for m in turn) for turn in turns]
    total = sum(totals)
    remove: list = []
    idx = 0
    while total > budget_tokens and idx < len(turns):
        remove.extend(turns[idx])
        total -= totals[idx]
        idx += 1
    return remove


def check_prefix_budgets(
    system_prompt: str, tool_definitions: str, skills: str
) -> list[str]:
    """Overrun notes for the fixed prefix (system prompt / tools / skills).

    Returns one ``"<component> <used>/<budget> tokens (over)"`` string
    per overrun, [] when everything fits. Advisory only — the caller
    logs these and notes them on the execution; it never fails a run.
    """
    overruns = []
    for name, text, budget in (
        ("system_prompt", system_prompt, SYSTEM_PROMPT_TOKENS),
        ("tool_definitions", tool_definitions, TOOL_DEFINITIONS_TOKENS),
        ("skills", skills, SKILLS_TOKENS),
    ):
        used = estimate_tokens(text or "")
        if used > budget:
            overruns.append(f"{name} {used}/{budget} tokens (over)")
    return overruns
