"""Lark user sync endpoint + ExecutionCreate.user_id assertion rules.

The user service is stubbed (no HTTP); the tests cover the sync
logic (linked -> return, unlinked -> create with the synthetic email)
and who may assert a user_id on POST /executions (service key yes,
JWT user no — their own sub always wins).
"""

from types import SimpleNamespace


def test_sync_returns_linked_user(client, monkeypatch):
    import app.routers.lark as lark_router

    monkeypatch.setattr(
        lark_router,
        "_user_service_get_by_identity",
        lambda provider, external_id: {"id": "u-1", "email": "a@b.c"},
    )
    resp = client.post(
        "/lark/users/sync",
        json={"provider": "lark", "external_id": "ou_1"},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "user_id": "u-1",
        "email": "a@b.c",
        "created": False,
    }


def test_sync_creates_user_with_synthetic_email(client, monkeypatch):
    import app.routers.lark as lark_router

    monkeypatch.setattr(
        lark_router, "_user_service_get_by_identity", lambda p, e: None
    )
    seen: dict = {}

    def fake_create(email, provider, external_id):
        seen.update(email=email, provider=provider, external_id=external_id)
        return {"user": {"id": "u-2", "email": email}, "created": True}

    monkeypatch.setattr(lark_router, "_user_service_create_user", fake_create)
    resp = client.post(
        "/lark/users/sync",
        json={"provider": "lark", "external_id": "ou_9"},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "user_id": "u-2",
        "email": "lark-ou_9@lark.local",
        "created": True,
    }
    assert seen["provider"] == "lark" and seen["external_id"] == "ou_9"


def test_sync_rejects_other_providers(client):
    resp = client.post(
        "/lark/users/sync",
        json={"provider": "slack", "external_id": "U1"},
    )
    assert resp.status_code == 422


def _fake_execution(**over):
    base = {
        "id": "exec_1",
        "mission_id": "agent_lark_me_mission",
        "mission_version": 1,
        "agent_id": "agent_lark_me",
        "agent_version": 1,
        "agent_version_drift": False,
        "session_id": "lark-abc",
        "user_id": over.get("user_id"),
        "status": "QUEUED",
        "error": None,
        "created_at": None,
        "started_at": None,
        "finished_at": None,
    }
    return SimpleNamespace(**base)


def test_service_key_call_may_assert_user_id(client, monkeypatch):
    """POST /executions with X-API-Key + user_id binds the run to it."""
    import app.services.execution_service as svc

    captured: dict = {}

    def fake_create(db, mission_id, input_data, session_id=None, user_id=None):
        captured["user_id"] = user_id
        return _fake_execution(user_id=user_id), []

    monkeypatch.setattr(svc, "create_execution", fake_create)
    # Auth is disabled in the test env (API_KEY=""), so no key header
    # is needed — the call arrives as a service-key call (user None).
    resp = client.post(
        "/executions",
        json={
            "mission_id": "agent_lark_me_mission",
            "input": {"message": "hi", "chat_id": "oc_1"},
            "user_id": "u-asserted",
        },
    )
    assert resp.status_code == 202, resp.text
    assert captured["user_id"] == "u-asserted"
    assert resp.json()["user_id"] == "u-asserted"


def test_jwt_user_cannot_assert_another_user_id(db_session, monkeypatch):
    """A JWT user's own sub always wins over payload.user_id.

    Builds its own app + client so get_current_user can be overridden
    (the shared ``client`` fixture cannot override dependencies).
    """
    from fastapi.testclient import TestClient

    import app.services.execution_service as svc
    from app.core.db import get_db
    from app.core.security import get_current_user
    from app.main import create_app

    captured: dict = {}

    def fake_create(db, mission_id, input_data, session_id=None, user_id=None):
        captured["user_id"] = user_id
        return _fake_execution(user_id=user_id), []

    monkeypatch.setattr(svc, "create_execution", fake_create)
    app = create_app()

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "u-jwt",
        "role": "admin",
    }
    with TestClient(app) as c:
        resp = c.post(
            "/executions",
            json={
                "mission_id": "agent_lark_me_mission",
                "input": {"message": "hi", "chat_id": "oc_1"},
                "user_id": "u-asserted",
            },
        )
    assert resp.status_code == 202, resp.text
    assert captured["user_id"] == "u-jwt"
    assert resp.json()["user_id"] == "u-jwt"
