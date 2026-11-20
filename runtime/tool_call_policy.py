"""Harness-driven tool-call sequencing and web-result token budgets."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from runtime.context_budget import estimate_tokens

SEQUENTIAL_DEFER_TEMPLATE = (
    "Tool '{tool_name}' was not executed: this mission requires calling "
    "tools {tool_list} one at a time. Finish with the first tool result, "
    "then call the next tool in a separate turn."
)

BUDGET_EXHAUSTED_TEMPLATE = (
    "Tool '{tool_name}' was not executed: the execution web-tool budget "
    "is exhausted ({reason}). Synthesize your answer from the tool "
    "results you already have."
)

TRUNCATION_MARKER = (
    "\n\n[... truncated to fit execution web-tool result budget ...]"
)


def truncate_text_to_token_budget(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to at most ``max_tokens`` (estimate_tokens)."""
    if max_tokens <= 0:
        return ""
    marker_tokens = estimate_tokens(TRUNCATION_MARKER)
    if marker_tokens >= max_tokens:
        return TRUNCATION_MARKER[: max(0, max_tokens * 4)]
    content_budget = max_tokens - marker_tokens
    if estimate_tokens(text) <= content_budget:
        return text
    low, high = 0, len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        chunk = text[:mid]
        if estimate_tokens(chunk) <= content_budget:
            best = chunk
            low = mid + 1
        else:
            high = mid - 1
    return best + TRUNCATION_MARKER


def tool_message_content_as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _messages_from_request(request) -> list:
    state = getattr(request, "state", None)
    if state is None:
        return []
    if isinstance(state, dict):
        return list(state.get("messages") or [])
    return list(getattr(state, "messages", None) or [])


def _last_ai_with_tool_calls(messages: list) -> AIMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and (msg.tool_calls or []):
            return msg
        if getattr(msg, "type", None) == "ai" and (getattr(msg, "tool_calls", None) or []):
            return msg
    return None


def sequential_defer_message(
    request,
    *,
    tool_name: str,
    tool_call_id: str,
    sequential_tools: frozenset[str],
) -> ToolMessage | None:
    """Return a ToolMessage when this call must wait for an earlier sequential tool."""
    if not sequential_tools or tool_name not in sequential_tools:
        return None
    messages = _messages_from_request(request)
    ai_msg = _last_ai_with_tool_calls(messages)
    if ai_msg is None:
        return None
    calls = list(ai_msg.tool_calls or [])
    seq_calls = [tc for tc in calls if str(tc.get("name") or "") in sequential_tools]
    if len(seq_calls) <= 1:
        return None
    first_id = str(seq_calls[0].get("id") or "")
    if tool_call_id == first_id:
        return None
    tool_list = ", ".join(sorted(sequential_tools))
    return ToolMessage(
        content=SEQUENTIAL_DEFER_TEMPLATE.format(
            tool_name=tool_name, tool_list=tool_list
        ),
        tool_call_id=tool_call_id,
        status="error",
    )


def tool_budget_enabled(
    *,
    budget_tools: frozenset[str],
    max_result_tokens: int,
    max_calls: dict[str, int],
) -> bool:
    if not budget_tools:
        return False
    return max_result_tokens > 0 or bool(max_calls)


def budget_token_block_reason(
    shared_counters,
    *,
    max_result_tokens: int,
) -> str | None:
    """Token-budget block reason; caller must hold ``shared_counters._lock``."""
    if max_result_tokens <= 0:
        return None
    if shared_counters.tool_budget_tokens_used >= max_result_tokens:
        return f"total result token budget {max_result_tokens} exhausted"
    return None


def apply_tool_result_budget(
    message: ToolMessage,
    *,
    shared_counters,
    tool_name: str,
    budget_tools: frozenset[str],
    max_result_tokens: int,
    max_calls: dict[str, int],
) -> ToolMessage:
    """Truncate result content and charge the shared execution budget."""
    if tool_name not in budget_tools or not tool_budget_enabled(
        budget_tools=budget_tools,
        max_result_tokens=max_result_tokens,
        max_calls=max_calls,
    ):
        return message
    text = tool_message_content_as_text(message.content)
    original = text
    call_cap = max_calls.get(tool_name)
    with shared_counters._lock:
        if max_result_tokens > 0:
            remaining = max_result_tokens - shared_counters.tool_budget_tokens_used
            if remaining <= 0:
                if call_cap is not None:
                    shared_counters.release_budget_call(tool_name)
                return ToolMessage(
                    content=BUDGET_EXHAUSTED_TEMPLATE.format(
                        tool_name=tool_name, reason="no tokens remaining"
                    ),
                    tool_call_id=message.tool_call_id,
                    status="error",
                )
            if estimate_tokens(text) > remaining:
                text = truncate_text_to_token_budget(text, remaining)
        if call_cap is None:
            shared_counters.tool_budget_calls[tool_name] = (
                shared_counters.tool_budget_calls.get(tool_name, 0) + 1
            )
        if max_result_tokens > 0:
            shared_counters.tool_budget_tokens_used += estimate_tokens(text)
    if text != original:
        return ToolMessage(
            content=text,
            tool_call_id=message.tool_call_id,
            status=message.status,
        )
    return message

