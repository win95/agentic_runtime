"""Auth tests — register, login, me, token verification."""

import jwt

from services.user_service.auth import create_access_token
from services.user_service.tests.conftest import auth_headers, login, make_user

SECRET = "test-secret"


def test_register_ok(client):
    response = client.post(
        "/auth/register", json={"email": "An@Example.com", "password": "password123"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "an@example.com"  # lowercased
    assert body["role"] == "user"
    assert body["is_active"] is True
    assert body["id"]
    assert "password" not in body and "password_hash" not in body


def test_register_duplicate_email(client):
    make_user("dup@example.com")
    response = client.post(
        "/auth/register", json={"email": "dup@example.com", "password": "password123"}
    )
    assert response.status_code == 409


def test_register_validation(client):
    assert (
        client.post(
            "/auth/register", json={"email": "not-an-email", "password": "password123"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/auth/register", json={"email": "a@example.com", "password": "short"}
        ).status_code
        == 422
    )


def test_login_ok_returns_jwt(client):
    user = make_user("login@example.com")
    response = client.post(
        "/auth/login", json={"email": "login@example.com", "password": "password123"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 24 * 3600
    payload = jwt.decode(body["access_token"], SECRET, algorithms=["HS256"])
    assert payload["sub"] == user.id
    assert payload["email"] == "login@example.com"
    assert payload["role"] == "user"


def test_login_wrong_password_and_unknown_email_same_401(client):
    make_user("known@example.com")
    wrong = client.post(
        "/auth/login", json={"email": "known@example.com", "password": "nope-nope"}
    )
    unknown = client.post(
        "/auth/login", json={"email": "nobody@example.com", "password": "password123"}
    )
    assert wrong.status_code == 401
    assert unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]


def test_login_disabled_user_refused(client):
    user = make_user("off@example.com")
    from services.user_service.db import SessionLocal

    db = SessionLocal()
    try:
        db_user = db.get(type(user), user.id)
        db_user.is_active = False
        db.commit()
    finally:
        db.close()
    response = client.post(
        "/auth/login", json={"email": "off@example.com", "password": "password123"}
    )
    assert response.status_code == 401


def test_me_requires_and_accepts_token(client):
    assert client.get("/auth/me").status_code == 401
    user = make_user("me@example.com")
    token = login(client, "me@example.com")
    response = client.get("/auth/me", headers=auth_headers(token))
    assert response.status_code == 200
    assert response.json()["id"] == user.id


def test_me_tampered_and_expired_token(client):
    user = make_user("tok@example.com")
    token = login(client, "tok@example.com")
    assert (
        client.get("/auth/me", headers=auth_headers(token + "x")).status_code == 401
    )
    expired = create_access_token(
        user_id=user.id,
        email=user.email,
        role=user.role,
        secret=SECRET,
        expires_hours=-1,
    )
    assert client.get("/auth/me", headers=auth_headers(expired)).status_code == 401
    # A token signed with a different secret is not accepted either.
    other = create_access_token(
        user_id=user.id, email=user.email, role=user.role, secret="other-secret"
    )
    assert client.get("/auth/me", headers=auth_headers(other)).status_code == 401
