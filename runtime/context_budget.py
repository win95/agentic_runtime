"""Context window budget — Phương án A (16,384 tokens).

One source of truth for how the window is split (approved 2026-11-10;
revised 2026-11-15: retrieval and skills up, history down, plus a
long-term-memory block carved out of history; halved 2026-11-27 to
fit the qwen3-8b window — 16,384 input tokens on the LiteLLM proxy).

========================  =========  ======
Component                 Tokens     Share
========================  =========  ======
System prompt             512        3%
Tool definitions          1,024      6%
Skills                    1,024      6%
Retrieved context         4,096      25%
Execution input           2,048      13%
Conversation + tool out   4,096      25%
Output reserve            2,560      16%
KV cache / agent          1,024      6%
------------------------  ---------  ------
TOTAL                     16,384     100%
========================  =========  ======

The history slice (4,096) is shared three ways: the session-history
block (2,048 — prior turns, ``runtime/session_history.py``), the
long-term-memory block (512 — keyword-retrieved trimmed turns,
``runtime/long_term_memory.py``), and the current run's own
conversation + tool output (1,536 nominal). The trim enforces the
slice dynamically: ``HookMiddleware.before_model`` trims turns to
``HISTORY_TOKENS`` minus the history/LTM blocks actually rendered
for the run (``reserved_history_tokens``), so a fresh session with no
history keeps the full 4,096 for its own turns.

Enforcement (best-effort where noted, never crash a run):

- System prompt / tools / skills: measured at build time in
  ``deep_agent_node``; an overrun logs a warning and is noted on the
  execution — the run still proceeds (these are pinned config, failing
  the run would hide the real work).
- Retrieved context: ``retrieval_max_chars()`` derives the hook's char
  cap from RETRIEVAL_TOKENS (4,096 tok x 4 chars). The mission YAML
  may still override ``max_chars`` per mission.
- Execution input: already capped upstream at 8,192 bytes
  (``MAX_INPUT_JSON_BYTES``) = 2,048 tokens — this table just names
  that cap as the input bucket.
- History: ``HookMiddleware.before_model`` trims the oldest complete
  turns (AIMessage + its ToolMessages) until the turn total fits the
  effective history budget. System/human messages are never removed
  and are not counted against the history bucket (they have their own
  buckets). A trimmed turn is NOT lost: it is encoded into
  long-term memory first (``runtime/long_term_memory.py``), so a
  later execution can retrieve it by keyword.
- Output reserve: passed as ``max_tokens`` when the chat model is
  built (``runtime/llm_factory.py``) — one response can never eat more
  than the reserve.
- KV cache: a standalone reserve per agent. The prompt side
  (system + tools + skills + retrieval + input + history) must fit in
  ``PROMPT_SIDE_TOKENS`` = 12,800; the reserve itself is never spent
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.harness.schema import MissionHarness

MAX_CONTEXT_TOKENS = 16_384
SYSTEM_PROMPT_TOKENS = 512
TOOL_DEFINITIONS_TOKENS = 1_024
SKILLS_TOKENS = 1_024
# Flow router: the per-route human message (catalog + instructions). The
# catalog is compiled from the skill map, not the full SKILL body.
# NOT part of the window table above — it is a per-call cap, and the
# flow router is the known weak point, so it stays at 2,048.
ROUTE_INSTRUCTION_TOKENS = 2_048
RETRIEVAL_TOKENS = 4_096
EXECUTION_INPUT_TOKENS = 2_048
HISTORY_TOKENS = 4_096
# The long-term-memory block (keyword-retrieved trimmed turns) is
# carved out of the history slice, not added on top of it:
# 4,096 = 2,048 session history + 512 LTM + 1,536 live turns.
LTM_TOKENS = 512
OUTPUT_RESERVE_TOKENS = 2_560
KV_CACHE_TOKENS_PER_AGENT = 1_024

# Prompt side: everything except the output reserve and the KV cache
# reserve. 512 + 1,024 + 1,024 + 4,096 + 2,048 + 4,096 = 12,800.
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


def effective_history_tokens(
    mission: dict | None,
    *,
    harness: MissionHarness | None = None,
) -> int:
    """History budget for one mission's run (tran 2026-11-19, harness opt-in).

    Default: ``HISTORY_TOKENS`` (4,096) for every mission — same as before
    the agent-analytis history realloc.

    Opt-in: missions set ``harness.context_budget.reclaim_retrieval_for_history:
    true`` to move the unused retrieval slice (4,096 tok) into history
    instead (8,192 total history budget). Missions with
    ``execution.retrieval`` keep the default — their retrieval block consumes
    that slice at run time.

    ``harness`` is the run's already-resolved model when the caller
    has it (the mission graph resolves once per run); without one the
    harness is loaded (cached per mission).
    """
    from runtime.harness.registry import effective_harness

    mission_id = (mission or {}).get("mission_id")
    execution = (mission or {}).get("execution") or {}
    if harness is None:
        harness = effective_harness(mission_id, execution)
    reclaim = harness.context_budget.reclaim_retrieval_for_history
    if reclaim and not execution.get("retrieval"):
        return HISTORY_TOKENS + RETRIEVAL_TOKENS
    return HISTORY_TOKENS


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


def group_turns(messages: list) -> list[list]:
    """Group ai/tool messages into turns (ai + its following tools).

    System/human messages are not part of any turn. A tool message with
    no preceding ai message forms a singleton turn (defensive — should
    not happen in a well-formed history). Shared by ``trim_history``
    and the long-term-memory encoder, so a trimmed turn is stored as
    the same unit it was removed as.
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


def _turns(messages: list) -> list[list]:
    """Backwards-compatible alias for ``group_turns``."""
    return group_turns(messages)


def trim_text_to_token_budget(text: str, max_tokens: int) -> str:
    """Drop whole lines from the start until ``text`` fits ``max_tokens``.

    Used when a fixed block (e.g. the flow catalog) must stay intact and
    a variable block (e.g. session history) yields. Preserves the most
    recent lines at the end of the text.
    """
    if not text or max_tokens < 1:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    marker = "... (earlier session history omitted to fit route budget)"
    marker_tok = estimate_tokens(marker)
    lines = text.split("\n")
    original_count = len(lines)
    while (
        lines
        and estimate_tokens("\n".join(lines)) + marker_tok + (1 if lines else 0)
        > max_tokens
    ):
        lines.pop(0)
    if not lines:
        return (
            marker
            if marker_tok <= max_tokens
            else marker[: max(0, max_tokens * CHARS_PER_TOKEN)]
        )
    body = "\n".join(lines)
    if len(lines) < original_count:
        combined = f"{marker}\n{body}"
        if estimate_tokens(combined) <= max_tokens:
            return combined
        return trim_text_to_token_budget(body, max_tokens)
    return body


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
    turns = group_turns(messages)
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
