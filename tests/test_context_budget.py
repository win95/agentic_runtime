"""Context window budget (Phương án A) tests — no live LLM needed."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from runtime.context_budget import (
    COMPONENT_BUDGETS,
    HISTORY_TOKENS,
    KV_CACHE_TOKENS_PER_AGENT,
    MAX_CONTEXT_TOKENS,
    OUTPUT_RESERVE_TOKENS,
    PROMPT_SIDE_TOKENS,
    check_prefix_budgets,
    estimate_tokens,
    message_tokens,
    retrieval_max_chars,
    trim_history,
)


def _turn(i: int, chars: int = 6000) -> list:
    """One fake turn: an ai message with a tool call + its tool result."""
    call_id = f"call-{i}"
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": call_id,
                    "name": "search_fault_history",
                    "args": {"asset_id": "O21"},
                }
            ],
        ),
        ToolMessage(content=f"TURN-{i}-MARKER" + "x" * chars, tool_call_id=call_id),
    ]


def _history(n_turns: int, chars: int = 6000) -> list:
    messages: list = [
        SystemMessage(content="retrieved context"),
        HumanMessage(content="analyze O21 faults"),
    ]
    for i in range(n_turns):
        messages.extend(_turn(i, chars))
    return messages


# --- Budget math -------------------------------------------------------


def test_budgets_sum_to_max_context():
    assert sum(COMPONENT_BUDGETS.values()) == MAX_CONTEXT_TOKENS == 32_000
    assert PROMPT_SIDE_TOKENS == 24_832
    assert (
        PROMPT_SIDE_TOKENS + OUTPUT_RESERVE_TOKENS + KV_CACHE_TOKENS_PER_AGENT
        == MAX_CONTEXT_TOKENS
    )
    assert set(COMPONENT_BUDGETS) == {
        "system_prompt",
        "tool_definitions",
        "skills",
        "retrieval",
        "execution_input",
        "history",
        "output_reserve",
        "kv_cache_per_agent",
    }


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("x" * 400) == 100
    assert estimate_tokens("x" * 401) == 101  # ceil
    assert estimate_tokens("x" * 4) == 1


def test_retrieval_max_chars_from_budget():
    assert retrieval_max_chars() == 3_072 * 4 == 12_288
    from runtime.hooks.builtin import DEFAULT_RETRIEVAL_MAX_CHARS

    assert DEFAULT_RETRIEVAL_MAX_CHARS == 12_288


# --- trim_history ------------------------------------------------------


def test_trim_history_noop_under_budget():
    assert trim_history(_history(1)) == []


def test_trim_history_removes_oldest_turn_first():
    messages = _history(12)  # ~12 x 1,500 tok = ~18,000 > 15,616
    removed = trim_history(messages)
    assert removed, "over-budget history must trim something"
    assert messages[2] in removed and messages[3] in removed
    remaining = [m for m in messages if m not in removed]
    assert sum(
        message_tokens(m) for m in remaining if m.type in ("ai", "tool")
    ) <= HISTORY_TOKENS


def test_trim_history_never_removes_system_or_first_human():
    messages = _history(12)
    removed = trim_history(messages)
    assert messages[0] not in removed  # SystemMessage (retrieval)
    assert messages[1] not in removed  # HumanMessage (task input)


def test_trim_history_removes_whole_turn_pairs():
    messages = _history(12)
    removed = trim_history(messages)
    # A removed ai message's tool result goes with it (never orphaned).
    for i, message in enumerate(messages):
        if message.type == "ai" and message in removed:
            tools = [
                m
                for m in messages[i + 1 :]
                if m.type == "tool"
                and m.tool_call_id in {c["id"] for c in message.tool_calls}
            ]
            assert tools and all(m in removed for m in tools)


# --- Prefix check ------------------------------------------------------


def test_check_prefix_budgets():
    assert check_prefix_budgets("x" * 100, "y" * 100, "z" * 100) == []
    over = check_prefix_budgets("x" * 5000, "y" * 100, "z" * 100)
    assert len(over) == 1 and over[0].startswith("system_prompt 1250/1024")
    over = check_prefix_budgets(
        "x" * 100, "y" * 9000, "z" * 5000
    )
    assert {o.split()[0] for o in over} == {"tool_definitions", "skills"}


# --- Output reserve on the model ---------------------------------------


def test_build_chat_model_sets_output_reserve():
    from app.config import Settings
    from runtime.llm_factory import build_chat_model

    cfg = Settings(
        llm_api_base="http://localhost:4000",
        llm_api_key="test-key",
        llm_model="test-model",
    )
    model = build_chat_model(cfg)
    assert model.max_tokens == OUTPUT_RESERVE_TOKENS == 5_120


# --- Middleware through the real agent graph ---------------------------


def test_before_model_trims_long_history_through_real_graph():
    """End-to-end through the real create_deep_agent graph (fake model):
    a 12-turn history (~18k turn tokens) is trimmed before the first
    model call — oldest turns go, the run still completes, and the
    newest turn survives."""
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel,
    )

    from runtime.hooks import HookMiddleware, default_registry

    class _BindableFakeModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ARG002 - tools unused
            return self

    middleware = HookMiddleware(
        default_registry(), execution_id="e1", mission_id="m1", max_continues=0
    )
    answers = [AIMessage(content=f"I am done #{i}.") for i in range(4)]
    model = _BindableFakeModel(messages=iter(answers))
    agent = create_deep_agent(
        model=model, tools=[], system_prompt="test", middleware=[middleware]
    )
    history = _history(12)
    result = agent.invoke({"messages": history})
    assert middleware.trimmed_messages > 0
    assert middleware.trimmed_tokens > 0
    contents = [getattr(m, "content", "") or "" for m in result["messages"]]
    joined = "\n".join(str(c) for c in contents)
    assert "TURN-0-MARKER" not in joined, "oldest turn must be trimmed"
    assert "TURN-11-MARKER" in joined, "newest turn must survive"
    assert contents[0] == "retrieved context"
    assert "analyze O21 faults" in contents[1]
