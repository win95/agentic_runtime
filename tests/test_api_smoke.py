"""API smoke tests — health, validation, and the QUEUED dispatch path."""

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


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_execution_validation_error(client):
    # Missing mission_id entirely -> request validation error.
    response = client.post("/executions", json={"input": {}})
    assert response.status_code == 422


def test_create_execution_unknown_mission(client):
    response = client.post(
        "/executions", json={"mission_id": "nope", "input": {"asset_id": "O21"}}
    )
    assert response.status_code == 404


def test_create_execution_missing_required_input(client):
    response = client.post("/missions", json=MISSION_PAYLOAD)
    assert response.status_code == 201
    response = client.post(
        "/executions", json={"mission_id": "fault_analysis", "input": {"asset_id": "O21"}}
    )
    assert response.status_code == 422


def test_create_execution_queued_mocked(client, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        execution_service, "dispatch_execution", lambda execution_id: calls.append(execution_id)
    )
    response = client.post("/missions", json=MISSION_PAYLOAD)
    assert response.status_code == 201

    response = client.post(
        "/executions",
        json={
            "mission_id": "fault_analysis",
            "input": {
                "asset_id": "O21",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
            },
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "QUEUED"
    assert body["execution_id"].startswith("exec_")
    assert body["mission_id"] == "fault_analysis"
    # Only the execution id was dispatched to Celery.
    assert calls == [body["execution_id"]]

    response = client.get(f"/executions/{body['execution_id']}")
    assert response.status_code == 200
    assert response.json()["status"] == "QUEUED"

    # No result yet for a queued execution.
    response = client.get(f"/executions/{body['execution_id']}/result")
    assert response.status_code == 404
