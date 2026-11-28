"""run_execution persists the run's notes into the saved result.

Regression: the worker used to drop final_state["notes"] — they only
survived inside the Markdown report, so `GET /executions/{id}/result`
never showed the sandbox note (`Sandbox backend: <id>` /
`Sandbox unavailable`). Local only, never pushed (tran's rule).
"""

from __future__ import annotations

from app import models
from app.worker.tasks import run_execution


def test_run_execution_persists_notes(db_session, monkeypatch):
    db_session.add(models.Mission(id="m-notes", name="notes mission"))
    db_session.add(
        models.Execution(id="exec-notes", mission_id="m-notes", status="QUEUED")
    )
    db_session.commit()

    canned = {
        "status": "SUCCEEDED",
        "result": {"answer": "ok"},
        "evidence": [],
        "errors": [],
        "notes": ["Sandbox backend: sb-123", "Session history: 0 prior turn(s)"],
    }
    monkeypatch.setattr("runtime.mission_graph.run_mission", lambda *a, **k: canned)

    outcome = run_execution.apply(args=["exec-notes"]).get()

    assert outcome["status"] == "SUCCEEDED"
    db_session.expire_all()
    saved = (
        db_session.query(models.ExecutionResult)
        .filter_by(execution_id="exec-notes")
        .one()
    )
    assert saved.output["notes"] == [
        "Sandbox backend: sb-123",
        "Session history: 0 prior turn(s)",
    ]
    assert saved.output["answer"] == "ok"
