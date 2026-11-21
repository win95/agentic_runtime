"""Threads / Runs aliases (docs/agent-definition.md).

Main-feature coverage only: thread minting, run creation in a thread,
and the read aliases agreeing with the session/execution routes.
"""

from app.services import execution_service

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
        "properties": {
            "root_causes": {"type": "array"},
            "evidence": {"type": "array"},
            "recommendations": {"type": "array"},
        },
    },
    "execution_config": {"mode": "hybrid", "agent_id": "fault_investigator"},
    "capabilities": ["fault_history", "maintenance_analysis", "knowledge_retrieval"],
    "agent_id": "fault_investigator",
}

INPUT = {"asset_id": "O21", "start_date": "2026-10-01", "end_date": "2026-10-31"}


def _seed_mission(client):
    response = client.post("/missions", json=MISSION_PAYLOAD)
    assert response.status_code == 201


def test_create_thread_then_404_until_first_run(client):
    response = client.post("/threads")
    assert response.status_code == 201
    thread_id = response.json()["thread_id"]
    assert thread_id.startswith("sess_")
    assert response.json()["execution_count"] == 0
    assert response.json()["runs"] == []
    # No row until the first run — like sessions, an empty thread is 404.
    assert client.get(f"/threads/{thread_id}").status_code == 404


def test_thread_run_flow(client, monkeypatch):
    _seed_mission(client)
    calls: list[str] = []
    monkeypatch.setattr(
        execution_service,
        "dispatch_execution",
        lambda execution_id: calls.append(execution_id),
    )
    thread_id = client.post("/threads").json()["thread_id"]
    response = client.post(
        f"/threads/{thread_id}/runs",
        json={"mission_id": "fault_analysis", "input": INPUT},
    )
    assert response.status_code == 202
    run = response.json()
    assert run["thread_id"] == thread_id
    assert run["status"] == "QUEUED"
    assert calls == [run["run_id"]]

    response = client.get(f"/threads/{thread_id}")
    assert response.status_code == 200
    thread = response.json()
    assert thread["thread_id"] == thread_id
    assert thread["execution_count"] == 1
    assert [r["run_id"] for r in thread["runs"]] == [run["run_id"]]

    response = client.get(f"/threads/{thread_id}/runs")
    assert response.status_code == 200
    assert [r["run_id"] for r in response.json()] == [run["run_id"]]

    response = client.get(f"/runs/{run['run_id']}")
    assert response.status_code == 200
    assert response.json()["run_id"] == run["run_id"]
    # No result row yet — the worker has not run.
    assert client.get(f"/runs/{run['run_id']}/result").status_code == 404
    # The same run is visible under the old route too.
    response = client.get(f"/executions/{run['run_id']}")
    assert response.status_code == 200
    assert response.json()["execution_id"] == run["run_id"]


def test_thread_run_errors(client):
    _seed_mission(client)
    thread_id = client.post("/threads").json()["thread_id"]
    response = client.post(
        f"/threads/{thread_id}/runs",
        json={"mission_id": "nope", "input": INPUT},
    )
    assert response.status_code == 404
    assert client.get("/threads/bad%20id").status_code == 422
    assert client.get("/threads/nope").status_code == 404
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/nope/result").status_code == 404
