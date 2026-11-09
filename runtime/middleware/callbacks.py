"""LangChain callback handler: token usage + tool-call audit rows.

Attached per Deep Agent run (see ``deep_agent_node`` in
``runtime/mission_graph.py``). Token counts come only from the provider
(``response.llm_output["token_usage"]``); when the provider reports no
usage the counts stay NULL with ``usage_source="missing"`` — counts are
never estimated or invented, and no cost is ever recorded.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app import models

logger = logging.getLogger(__name__)

MAX_ARGS_CHARS = 2000
REDACTED = "[redacted]"
_SECRET_KEY_PARTS = ("key", "token", "password", "secret")


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def redact_value(value: Any) -> Any:
    """Recursively replace values of secret-looking keys with [redacted]."""
    if isinstance(value, dict):
        return {
            k: (REDACTED if _is_secret_key(str(k)) else redact_value(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


def truncate_args(args: Any) -> str | None:
    """Serialize tool input to a redacted string of at most 2000 chars.

    Accepts a dict, a JSON string, or anything else; secrets are redacted
    before truncation. Returns None for a None input.
    """
    if args is None:
        return None
    if isinstance(args, dict):
        text = json.dumps(redact_value(args), ensure_ascii=False, default=str)
    elif isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (ValueError, TypeError):
            text = args
        else:
            if isinstance(parsed, (dict, list)):
                text = json.dumps(redact_value(parsed), ensure_ascii=False, default=str)
            else:
                text = args
    else:
        text = json.dumps(redact_value(args), ensure_ascii=False, default=str)
    return text[:MAX_ARGS_CHARS]


class TokenAuditCallbackHandler(BaseCallbackHandler):
    """Record token usage (on_llm_end) and tool calls (on_tool_*).

    One instance per run, constructed with that run's execution context.
    Every DB write is wrapped so a logging failure is swallowed, never
    raised. The LangChain run id is reused as the AuditLog row id, so a
    tool's start/end/error update the same row.
    """

    raise_error = False

    def __init__(
        self,
        execution_id: str,
        mission_id: str,
        tenant: str = "default",
        model: str | None = None,
        session_factory=None,
    ):
        super().__init__()
        self.execution_id = execution_id
        self.mission_id = mission_id
        self.tenant = tenant
        self.model = model
        # Injectable for tests; defaults to the app SessionLocal.
        self._session_factory = session_factory
        self._tool_runs: dict[UUID, dict] = {}

    def _session(self):
        if self._session_factory is not None:
            return self._session_factory()
        from app.core.db import SessionLocal

        return SessionLocal()

    def _write(self, row) -> None:
        try:
            db = self._session()
            try:
                db.add(row)
                db.commit()
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001 — logging must never raise
            logger.warning("Middleware logging failed (non-fatal): %s", exc)

    def _finish_audit(
        self,
        run_id: UUID,
        status: str,
        duration_ms: float | None,
        tool_name: str | None,
        error: str | None = None,
    ) -> None:
        try:
            db = self._session()
            try:
                row = db.get(models.AuditLog, str(run_id))
                if row is None:
                    row = models.AuditLog(
                        id=str(run_id),
                        execution_id=self.execution_id,
                        mission_id=self.mission_id,
                        tenant=self.tenant,
                        tool_name=tool_name or "unknown",
                        args_truncated=None,
                        status=status,
                    )
                    db.add(row)
                row.status = status
                row.duration_ms = duration_ms
                if error is not None:
                    row.error = error[:MAX_ARGS_CHARS]
                db.commit()
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001 — logging must never raise
            logger.warning("Middleware logging failed (non-fatal): %s", exc)

    # --- LLM calls: token usage ---
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        model = llm_output.get("model_name") or llm_output.get("model") or self.model
        self._write(
            models.LLMUsage(
                id=uuid.uuid4().hex,
                execution_id=self.execution_id,
                mission_id=self.mission_id,
                tenant=self.tenant,
                model=model,
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
                usage_source="provider" if usage else "missing",
            )
        )

    # --- Tool calls: audit trail ---
    def on_tool_start(
        self,
        serialized: dict | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or "unknown"
        self._tool_runs[run_id] = {"started_at": time.monotonic(), "tool_name": name}
        self._write(
            models.AuditLog(
                id=str(run_id),
                execution_id=self.execution_id,
                mission_id=self.mission_id,
                tenant=self.tenant,
                tool_name=name,
                args_truncated=truncate_args(input_str),
                status="started",
            )
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        started = self._tool_runs.pop(run_id, None)
        duration = (
            (time.monotonic() - started["started_at"]) * 1000 if started else None
        )
        self._finish_audit(
            run_id,
            status="success",
            duration_ms=duration,
            tool_name=started["tool_name"] if started else None,
        )

    def on_tool_error(
        self,
        error: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        started = self._tool_runs.pop(run_id, None)
        duration = (
            (time.monotonic() - started["started_at"]) * 1000 if started else None
        )
        self._finish_audit(
            run_id,
            status="error",
            duration_ms=duration,
            tool_name=started["tool_name"] if started else None,
            error=str(error),
        )
