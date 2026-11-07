"""Autonomous Agent — Deep Agents (Layer 4).

Deep Agents are built on LangGraph and return a compiled graph, so the
agent created here is embedded as a node inside the Mission Graph — there
is no separate agent runtime.

Start with one deep agent; subagents are added only when a task genuinely
needs to be split or run in parallel (the deepagents package manages them).
"""

from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous investigation agent. Plan the investigation, "
    "use the provided tools to gather evidence, and only conclude when the "
    "evidence supports the answer. If a tool is unavailable or returns no "
    "data, say so explicitly instead of inventing facts. Summarize your "
    "findings and the evidence you relied on."
)


def get_chat_model():
    """Model factory from env. Primary: OpenAI (langchain-openai).
    Alternative: LLM_PROVIDER=anthropic (+ the `anthropic` extra)."""
    provider = settings.llm_provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key or None,
            temperature=0,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key or None,
            temperature=0,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'openai' or 'anthropic')")


def build_deep_agent(tools: list, system_prompt: str | None = None):
    """Create a Deep Agent (compiled LangGraph) over the given tools."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
    )


def run_deep_agent(agent, mission: dict, input_data: dict) -> dict:
    """Invoke the deep agent for one mission input.

    Returns {"summary": str, "evidence": [...], "errors": [...]}.
    Evidence here is the agent's own summary; the structured evidence
    still comes from the Capability Graphs and tool outputs.
    """
    objective = mission.get("objective", "")
    prompt = (
        f"Mission: {mission.get('mission_id')}\n"
        f"Objective: {objective}\n"
        f"Input: {input_data}\n\n"
        "Investigate and return a concise summary of your findings, "
        "citing the evidence you gathered."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages") or []
    summary = ""
    for message in reversed(messages):
        content = getattr(message, "content", "") or ""
        if isinstance(content, str) and content.strip():
            summary = content
            break
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            summary = " ".join(t for t in texts if t)
            if summary:
                break
    evidence = []
    if summary:
        evidence.append(
            {"source": "deep_agent", "kind": "summary", "summary": summary, "stub": False}
        )
    return {"summary": summary, "evidence": evidence, "errors": []}
