"""User-service tests — environment first, then service imports.

The service runs against an in-memory SQLite DB (no Postgres needed)
with a fixed test JWT secret. The settings object and the engine are
created at import time, so the env vars must be set before any
``services.user_service`` module is imported.
"""

import os

os.environ["USER_SERVICE_DATABASE_URL"] = "sqlite://"
os.environ["JWT_SECRET"] = "test-secret"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from services.user_service.auth import hash_password  # noqa: E402
from services.user_service.db import Base, SessionLocal, engine  # noqa: E402
from services.user_service.main import app  # noqa: E402
from services.user_service.models import User  # noqa: E402


@pytest.fixture()
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(bind=engine)


def make_user(email: str, password: str = "password123", role: str = "user") -> User:
    db = SessionLocal()
    try:
        user = User(
            email=email.lower(), password_hash=hash_password(password), role=role
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def login(client: TestClient, email: str, password: str = "password123") -> str:
    response = client.post(
        "/auth/login", json={"email": email, "password": password}
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
