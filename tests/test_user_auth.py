"""JWT user auth on the platform API — verify, own, API-key compat.

The user service signs tokens; this API only verifies them (see
``app/core/security.py``). These tests sign tokens directly with PyJWT
so the platform tests stay independent of the service package.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException

from app import models
from app.config import settings
from app.core.security import decode_bearer_token
from app.services import execution_service

SECRET = "test-secret"

MISSION_PAYLOAD = {
    "id": "fault_analysis",
    "name": "Fault Analysis",
    "version": 1,
    "objective": "Analyze equipment faults and identify root causes.",
    "input_schema": {"type": "object"},
    "output_schema": {"type": "object"},
    "execution_config": {"mode": "hybrid", "agent_id": "fault_investigator"},
    "capabilities": ["fault_history"],
    "agent_id": "fault_investigator",
}


def _token(user_id: str, secret: str = SECRET, expires_seconds: int = 3600) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": user_id,
            "email": f"{user_id}@example.com",
            "role": "user",
            "iat": now,
            "exp": now + timedelta(seconds=expires_seconds),
        },
        secret,
        algorithm="HS256",
    )


@pytest.fixture()
def jwt_secret(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", SECRET)
    return SECRET


@pytest.fixture()
def mission(client):
    response = client.post("/missions", json=MISSION_PAYLOAD)
    assert response.status_code == 201, response.text
    return MISSION_PAYLOAD["id"]


@pytest.fixture()
def no_dispatch(monkeypatch):
    monkeypatch.setattr(
        execution_service, "dispatch_execution", lambda execution_id: None
    )


def _create_execution(client, mission_id, token=None, headers=None):
    hdrs = dict(headers or {})
    if token is not None:
        hdrs["Authorization"] = f"Bearer {token}"
    return client.post(
        "/executions",
        json={"mission_id": mission_id, "input": {}},
        headers=hdrs,
    )


# --- Token verification ---


def test_decode_bearer_token_valid(jwt_secret):
    payload = decode_bearer_token(f"Bearer {_token('user-1')}")
    assert payload["sub"] == "user-1"


def test_decode_bearer_token_no_header_returns_none(jwt_secret):
    assert decode_bearer_token(None) is None
    assert decode_bearer_token("Basic abc") is None


def test_decode_bearer_token_invalid_raises_401(jwt_secret):
    with pytest.raises(HTTPException) as exc:
        decode_bearer_token("Bearer not-a-token")
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException) as exc:
        decode_bearer_token(f"Bearer {_token('user-1', secret='wrong-secret')}")
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException) as exc:
        decode_bearer_token(f"Bearer {_token('user-1', expires_seconds=-10)}")
    assert exc.value.status_code == 401


def test_decode_bearer_token_ignored_without_secret(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", "")
    assert decode_bearer_token(f"Bearer {_token('user-1')}") is None


# --- Executions carry the user ---


def test_create_execution_with_jwt_sets_user_id(
    client, mission, no_dispatch, jwt_secret
):
    response = _create_execution(client, mission, token=_token("user-1"))
    assert response.status_code == 202, response.text
    assert response.json()["user_id"] == "user-1"


def test_create_execution_with_api_key_leaves_user_id_null(
    client, mission, no_dispatch, monkeypatch
):
    monkeypatch.setattr(settings, "api_key", "svc-key")
    response = _create_execution(
        client, mission, headers={"X-API-Key": "svc-key"}
    )
    assert response.status_code == 202, response.text
    assert response.json()["user_id"] is None


def test_no_credential_when_api_key_required(
    client, mission, no_dispatch, monkeypatch
):
    monkeypatch.setattr(settings, "api_key", "svc-key")
    assert _create_execution(client, mission).status_code == 401
    # And a wrong API key does not pass either.
    assert (
        _create_execution(client, mission, headers={"X-API-Key": "wrong"}).status_code
        == 401
    )


def test_list_only_shows_own_executions(
    client, mission, no_dispatch, jwt_secret, monkeypatch
):
    mine = _create_execution(client, mission, token=_token("user-1"))
    theirs = _create_execution(client, mission, token=_token("user-2"))
    assert mine.status_code == 202 and theirs.status_code == 202

    listed = client.get(
        "/executions", headers={"Authorization": f"Bearer {_token('user-1')}"}
    )
    assert listed.status_code == 200
    ids = [e["execution_id"] for e in listed.json()]
    assert ids == [mine.json()["execution_id"]]

    # The service key still sees everything.
    monkeypatch.setattr(settings, "api_key", "svc-key")
    listed = client.get("/executions", headers={"X-API-Key": "svc-key"})
    assert {e["execution_id"] for e in listed.json()} == {
        mine.json()["execution_id"],
        theirs.json()["execution_id"],
    }


def test_get_and_cancel_other_users_execution_404(
    client, mission, no_dispatch, jwt_secret
):
    mine = _create_execution(client, mission, token=_token("user-1"))
    theirs = _create_execution(client, mission, token=_token("user-2"))
    mine_id = mine.json()["execution_id"]
    theirs_id = theirs.json()["execution_id"]

    other = _token("user-1")
    assert (
        client.get(
            f"/executions/{theirs_id}", headers={"Authorization": f"Bearer {other}"}
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/executions/{mine_id}", headers={"Authorization": f"Bearer {other}"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/executions/{theirs_id}/cancel",
            headers={"Authorization": f"Bearer {other}"},
        ).status_code
        == 404
    )
    response = client.post(
        f"/executions/{mine_id}/cancel",
        headers={"Authorization": f"Bearer {other}"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "CANCELLED"


def test_result_of_other_users_execution_404(
    client, mission, no_dispatch, jwt_secret, db_session
):
    created = _create_execution(client, mission, token=_token("user-1"))
    execution_id = created.json()["execution_id"]
    db_session.add(
        models.ExecutionResult(
            id="res_user_1", execution_id=execution_id, output={}, evidence=[]
        )
    )
    db_session.commit()

    assert (
        client.get(
            f"/executions/{execution_id}/result",
            headers={"Authorization": f"Bearer {_token('user-2')}"},
        ).status_code
        == 404
    )
    response = client.get(
        f"/executions/{execution_id}/result",
        headers={"Authorization": f"Bearer {_token('user-1')}"},
    )
    assert response.status_code == 200
    assert response.json()["execution_id"] == execution_id
