"""Streaming SSE lifecycle events (A.2, docs/agent-definition.md).

Main-feature coverage only: publish is a no-op without Redis, the
stream endpoint emits the terminal event and closes, and the guest
CLI's SSE line parser accepts real event lines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "manual"))

import guest_chat  # noqa: E402

from app import models  # noqa: E402
from app.services import execution_service  # noqa: E402
from runtime import events  # noqa: E402

MISSION_PAYLOAD = {
    "id": "fault_analysis",
    "name": "Fault Analysis",
    "version": 1,
    "objective": "Analyze equipment faults and identify root causes.",
    "input_schema": {
        "type": "object",
        "required": ["asset_id", "start_date", "end_date"],
    },
    "output_schema": {
        "type": "object",
        "properties": {"root_causes": {"type": "array"}},
    },
    "execution_config": {"mode": "hybrid", "agent_id": "fault_investigator"},
    "capabilities": ["fault_history"],
    "agent_id": "fault_investigator",
}

INPUT = {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"}


def _seed_execution(client, db_session, monkeypatch, status="QUEUED"):
    response = client.post("/missions", json=MISSION_PAYLOAD)
    assert response.status_code == 201
    monkeypatch.setattr(
        execution_service, "dispatch_execution", lambda execution_id: None
    )
    response = client.post(
        "/executions", json={"mission_id": "fault_analysis", "input": INPUT}
    )
    assert response.status_code == 202
    execution_id = response.json()["execution_id"]
    if status != "QUEUED":
        execution = db_session.get(models.Execution, execution_id)
        execution.status = status
        if status == "SUCCEEDED":
            db_session.add(
                models.ExecutionResult(
                    id="res_test123456",
                    execution_id=execution_id,
                    output={"answer": "ok"},
                    evidence=[],
                )
            )
        db_session.commit()
    return execution_id


def test_publish_is_noop_without_redis():
    # Tests run with CELERY_BROKER_URL=memory:// — no Redis, so the
    # publisher returns False and never raises.
    assert events.publish_execution_event("exec_x", "status", status="RUNNING") is False


def test_publish_writes_channel_and_payload(monkeypatch):
    published: list[tuple[str, str]] = []

    class FakeRedis:
        def publish(self, channel, message):
            published.append((channel, message))
            return 1

    monkeypatch.setattr(events, "_client", FakeRedis())
    monkeypatch.setattr(events, "_client_tried", True)
    assert events.publish_execution_event("exec_x", "status", status="RUNNING") is True
    assert len(published) == 1
    channel, message = published[0]
    assert channel == "exec_events:exec_x"
    import json

    payload = json.loads(message)
    assert payload["type"] == "status"
    assert payload["status"] == "RUNNING"
    assert payload["execution_id"] == "exec_x"


def test_stream_terminal_execution_closes(client, db_session, monkeypatch):
    execution_id = _seed_execution(client, db_session, monkeypatch, status="SUCCEEDED")
    response = client.get(f"/executions/{execution_id}/stream")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    body = response.text
    assert '"status": "SUCCEEDED"' in body
    assert f'"execution_id": "{execution_id}"' in body


def test_stream_unknown_execution_404(client):
    assert client.get("/executions/nope/stream").status_code == 404


def test_stream_closes_after_timeout(client, db_session, monkeypatch):
    import app.routers.executions as executions_router

    monkeypatch.setattr(executions_router, "MAX_STREAM_SECONDS", 0.5)
    execution_id = _seed_execution(client, db_session, monkeypatch)
    response = client.get(f"/executions/{execution_id}/stream")
    assert response.status_code == 200
    # Still QUEUED, no status change in 0.5s — the stream closes empty
    # (no terminal event) rather than hanging.
    assert '"status": "SUCCEEDED"' not in response.text


def test_parse_sse_event():
    event = guest_chat._parse_sse_event(
        'data: {"type": "status", "execution_id": "e1", "status": "RUNNING"}'
    )
    assert event == {"type": "status", "execution_id": "e1", "status": "RUNNING"}
    assert guest_chat._parse_sse_event(": ping") is None
    assert guest_chat._parse_sse_event("event: status") is None
    assert guest_chat._parse_sse_event("") is None
    assert guest_chat._parse_sse_event("data: not-json") is None
    assert guest_chat._parse_sse_event("data: [1, 2]") is None
