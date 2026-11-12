"""Middleware tests — token logging + tool-call audit rows.

No LLM, no live DB: the handler is driven directly with real LangChain
callback objects, and the end-to-end test uses a fake agent that fires
the callbacks it receives through the invoke config (the way the real
Deep Agent would). Tables are the in-memory SQLite DB from conftest.
"""

import json
import uuid

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import runtime.deep_agent as deep_agent_module
import seeds.load_missions as load_missions
from app.services import mission_service
from runtime.middleware import (
    TokenAuditCallbackHandler,
    get_audit_for_execution,
    get_usage_for_execution,
    get_usage_totals,
)
from runtime.mission_graph import run_mission


def _llm_result(usage=None, model_name=None):
    llm_output = {}
    if usage is not None:
        llm_output["token_usage"] = usage
    if model_name is not None:
        llm_output["model_name"] = model_name
    return LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="hi"))]],
        llm_output=llm_output,
    )


def _handler(db_session, **kwargs):
    kwargs.setdefault("execution_id", "exec-1")
    kwargs.setdefault("mission_id", "system_admin")
    kwargs.setdefault("tenant", "default")
    kwargs.setdefault("model", "gpt-4o-mini")
    return TokenAuditCallbackHandler(**kwargs)


def test_llm_end_logs_provider_usage(db_session):
    handler = _handler(db_session)
    handler.on_llm_end(
        _llm_result(
            {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
            model_name="gpt-4o-mini",
        ),
        run_id=uuid.uuid4(),
    )
    rows = get_usage_for_execution(db_session, "exec-1")
    assert len(rows) == 1
    row = rows[0]
    assert row.prompt_tokens == 120
    assert row.completion_tokens == 45
    assert row.total_tokens == 165
    assert row.model == "gpt-4o-mini"
    assert row.mission_id == "system_admin"
    assert row.tenant == "default"
    assert row.usage_source == "provider"


def test_llm_end_missing_usage_logs_nulls(db_session):
    handler = _handler(db_session)
    handler.on_llm_end(_llm_result(), run_id=uuid.uuid4())
    rows = get_usage_for_execution(db_session, "exec-1")
    assert len(rows) == 1
    assert rows[0].prompt_tokens is None
    assert rows[0].completion_tokens is None
    assert rows[0].total_tokens is None
    assert rows[0].usage_source == "missing"
    # Model falls back to the handler's constructor value.
    assert rows[0].model == "gpt-4o-mini"


def test_tool_audit_redacts_truncates_and_times(db_session):
    handler = _handler(db_session)
    run_id = uuid.uuid4()
    long_value = "x" * 3000
    handler.on_tool_start(
        {"name": "get_executions"},
        json.dumps(
            {
                "execution_id": "exec-9",
                "api_key": "sk-secret",
                "nested": {"token": "t", "ok": 1},
                "blob": long_value,
            }
        ),
        run_id=run_id,
    )
    rows = get_audit_for_execution(db_session, "exec-1")
    assert len(rows) == 1
    assert rows[0].tool_name == "get_executions"
    assert rows[0].status == "started"
    assert "sk-secret" not in (rows[0].args_truncated or "")
    assert rows[0].args_truncated.count("[redacted]") == 2
    assert len(rows[0].args_truncated) == 2000  # truncated to the cap
    handler.on_tool_end("ok", run_id=run_id)
    # The handler writes through its own session; expire the fixture
    # session's identity map so the re-query sees the committed update.
    db_session.expire_all()
    rows = get_audit_for_execution(db_session, "exec-1")
    assert len(rows) == 1
    assert rows[0].status == "success"
    assert rows[0].duration_ms is not None and rows[0].duration_ms >= 0


def test_tool_error_row(db_session):
    handler = _handler(db_session)
    run_id = uuid.uuid4()
    handler.on_tool_start({"name": "get_missions"}, "{}", run_id=run_id)
    handler.on_tool_error(ValueError("boom"), run_id=run_id)
    rows = get_audit_for_execution(db_session, "exec-1")
    assert len(rows) == 1
    assert rows[0].status == "error"
    assert "boom" in (rows[0].error or "")
    assert rows[0].duration_ms is not None


def test_tool_end_without_start_still_logs(db_session):
    handler = _handler(db_session)
    handler.on_tool_end("ok", run_id=uuid.uuid4())
    rows = get_audit_for_execution(db_session, "exec-1")
    assert len(rows) == 1
    assert rows[0].status == "success"
    assert rows[0].tool_name == "unknown"
    assert rows[0].duration_ms is None


def test_logging_failure_does_not_raise(db_session):
    def _boom():
        raise RuntimeError("db down")

    handler = _handler(db_session, session_factory=_boom)
    run_id = uuid.uuid4()
    # None of these may propagate, even though every write fails.
    handler.on_llm_end(_llm_result({"prompt_tokens": 1}), run_id=run_id)
    handler.on_tool_start({"name": "get_missions"}, "{}", run_id=run_id)
    handler.on_tool_end("ok", run_id=run_id)
    handler.on_tool_error(ValueError("x"), run_id=uuid.uuid4())


# --- End-to-end: mocked Deep Agent run writes both row kinds ---
class _CallbackFiringFakeAgent:
    """Fake compiled agent: fires the callbacks from the invoke config
    (LLM end with provider usage, one tool start/end), then returns a
    canned summary — the way the real Deep Agent would drive them."""

    def invoke(self, state, config=None):
        callbacks = (config or {}).get("callbacks") or []
        for cb in callbacks:
            cb.on_llm_end(
                _llm_result(
                    {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
                    model_name="gpt-4o-mini",
                ),
                run_id=uuid.uuid4(),
            )
            tool_run_id = uuid.uuid4()
            cb.on_tool_start(
                {"name": "get_executions"}, '{"status": "FAILED"}', run_id=tool_run_id
            )
            cb.on_tool_end('{"executions": []}', run_id=tool_run_id)
        return {"messages": [AIMessage(content="Có 0 execution lỗi hôm nay.")]}


def test_autonomous_run_writes_token_and_audit_rows(db_session, monkeypatch):
    load_missions.main()
    mission = mission_service.get_mission(db_session, "system_admin")
    mission_dict = mission_service.mission_to_dict(mission)

    def _fake_build(tools, system_prompt=None, skills=None, model_config=None,
                    middleware=None, backend=None):
        return _CallbackFiringFakeAgent()

    monkeypatch.setattr(
        deep_agent_module,
        "build_deep_agent",
        _fake_build,
    )

    final = run_mission(mission_dict, {"request": "execution nào lỗi hôm nay?"}, "exec-mw-e2e")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    usage = get_usage_for_execution(db_session, "exec-mw-e2e")
    assert len(usage) == 1
    assert usage[0].total_tokens == 100
    assert usage[0].mission_id == "system_admin"
    assert usage[0].usage_source == "provider"
    audit = get_audit_for_execution(db_session, "exec-mw-e2e")
    assert len(audit) == 1
    assert audit[0].tool_name == "get_executions"
    assert audit[0].status == "success"
    assert audit[0].duration_ms is not None
    assert get_usage_totals(db_session, mission_id="system_admin")["total_tokens"] == 100
