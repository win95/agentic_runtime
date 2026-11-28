"""Service-key auth and USER_SERVICE_API_KEY env loading."""

from services.user_service.config import Settings, settings
from services.user_service.db import SessionLocal
from services.user_service.models import UserIdentity
from services.user_service.tests.conftest import make_user


def test_settings_loads_user_service_api_key(monkeypatch):
    monkeypatch.setenv("USER_SERVICE_API_KEY", "sync-secret")
    monkeypatch.delenv("SERVICE_API_KEY", raising=False)
    cfg = Settings()
    assert cfg.service_api_key == "sync-secret"


def test_settings_loads_service_api_key_legacy_alias(monkeypatch):
    monkeypatch.delenv("USER_SERVICE_API_KEY", raising=False)
    monkeypatch.setenv("SERVICE_API_KEY", "legacy-secret")
    cfg = Settings()
    assert cfg.service_api_key == "legacy-secret"


def test_by_identity_fails_closed_when_key_not_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "service_api_key", "")
    resp = client.get(
        "/admin/users/by-identity",
        params={"provider": "lark", "external_id": "ou_x"},
        headers={"X-Service-Key": "anything"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "USER_SERVICE_API_KEY is not configured"


def test_by_identity_rejects_missing_or_wrong_service_key(client, monkeypatch):
    monkeypatch.setattr(settings, "service_api_key", "expected-key")
    missing = client.get(
        "/admin/users/by-identity",
        params={"provider": "lark", "external_id": "ou_x"},
    )
    assert missing.status_code == 401
    wrong = client.get(
        "/admin/users/by-identity",
        params={"provider": "lark", "external_id": "ou_x"},
        headers={"X-Service-Key": "wrong-key"},
    )
    assert wrong.status_code == 401


def test_by_identity_accepts_valid_service_key(client, monkeypatch):
    monkeypatch.setattr(settings, "service_api_key", "expected-key")
    user = make_user("lark-user@example.com")
    db = SessionLocal()
    try:
        db.add(
            UserIdentity(provider="lark", external_id="ou_ok", user_id=user.id)
        )
        db.commit()
    finally:
        db.close()
    resp = client.get(
        "/admin/users/by-identity",
        params={"provider": "lark", "external_id": "ou_ok"},
        headers={"X-Service-Key": "expected-key"},
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == user.id