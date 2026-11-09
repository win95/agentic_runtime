"""Autonomous Agent — Deep Agents (Layer 4).

Deep Agents are built on LangGraph and return a compiled graph, so the
agent created here is embedded as a node inside the Mission Graph — there
is no separate agent runtime.

Start with one deep agent; subagents are added only when a task genuinely
needs to be split or run in parallel (the deepagents package manages them).
"""

from __future__ import annotations

import logging

from runtime.llm_factory import build_chat_model
from runtime.prompts import get_system_prompt

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous investigation agent. Plan the investigation, "
    "use the provided tools to gather evidence, and only conclude when the "
    "evidence supports the answer. If a tool is unavailable or returns no "
    "data, say so explicitly instead of inventing facts. Summarize your "
    "findings and the evidence you relied on."
)

ADMIN_SYSTEM_PROMPT = (
    "You are the admin bot for the agentic-runtime platform. Answer in "
    "Vietnamese. Only use the admin tools you were given; never invent "
    "data — report exactly what the tools return, including their errors. "
    "When a tool returns an error or a refusal, say so plainly and do not "
    "work around it by guessing. Destructive actions (delete_mission, "
    "cancel_execution) require the exact id named in the request: if the "
    "request does not name an exact id, or names it ambiguously, refuse "
    "that action and ask for the exact id. Summarize what you did as "
    "actions_taken and cite the evidence you relied on."
)


def get_chat_model():
    """Chat model via the LiteLLM proxy (OpenAI-compatible).

    See `runtime.llm_factory`: base URL, key and model name come from env
    (LLM_API_BASE / LLM_API_KEY / LLM_MODEL). Missing config raises a
    clear RuntimeError when the agent is built, never at import time.
    """
    return build_chat_model()


def build_deep_agent(tools: list, system_prompt: str | None = None, skills: list | None = None):
    """Create a Deep Agent (compiled LangGraph) over the given tools.

    The system prompt defaults to the ``fault_investigator`` prompt from
    the MLflow Prompt Registry, falling back to DEFAULT_SYSTEM_PROMPT when
    the registry is unavailable (see ``runtime/prompts.py``).

    ``skills`` is a list of skill source paths (e.g. ``["/skills/"]``)
    for the native ``deepagents`` SkillsMiddleware — see
    ``runtime/skills/``. ``None`` means no skills middleware at all.
    """
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=system_prompt or get_system_prompt(DEFAULT_SYSTEM_PROMPT),
        skills=skills,
    )


def run_deep_agent(agent, mission: dict, input_data: dict, callbacks=None, files=None) -> dict:
    """Invoke the deep agent for one mission input.

    Returns {"summary": str, "evidence": [...], "actions_taken": [...],
    "errors": [...]}. Evidence here is the agent's own summary; the
    structured evidence still comes from the Capability Graphs and tool
    outputs. ``actions_taken`` lists the tool names the agent called, in
    order — the Mission Graph uses it to fill the result of autonomous
    missions (see ``deep_agent_node``).

    ``callbacks`` (e.g. the middleware token/audit handler built per run
    by ``deep_agent_node``) are passed through the invoke config so the
    handler sees the agent's LLM and tool calls. ``files`` (e.g. the
    selected skill files from ``runtime/skills/``) are passed through the
    invoke input so the native SkillsMiddleware can disclose them.
    """
    objective = mission.get("objective", "")
    prompt = (
        f"Mission: {mission.get('mission_id')}\n"
        f"Objective: {objective}\n"
        f"Input: {input_data}\n\n"
        "Investigate and return a concise summary of your findings, "
        "citing the evidence you gathered."
    )
    input_payload: dict = {"messages": [{"role": "user", "content": prompt}]}
    if files:
        input_payload["files"] = files
    config = {"callbacks": callbacks} if callbacks else None
    if config is not None:
        result = agent.invoke(input_payload, config=config)
    else:
        # No callbacks: keep the plain call so mock agents with a
        # one-argument invoke() (eval script) keep working.
        result = agent.invoke(input_payload)
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
    actions_taken: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name and name not in actions_taken:
                actions_taken.append(name)
    evidence = []
    if summary:
        evidence.append(
            {"source": "deep_agent", "kind": "summary", "summary": summary, "stub": False}
        )
    return {"summary": summary, "evidence": evidence, "actions_taken": actions_taken, "errors": []}
