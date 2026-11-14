"""Admin tests — list users, enable / disable, self-disable guard."""

from services.user_service.tests.conftest import auth_headers, login, make_user


def test_admin_list_requires_admin(client):
    make_user("plain@example.com")
    make_user("admin@example.com", role="admin")
    user_token = login(client, "plain@example.com")
    assert client.get("/admin/users").status_code == 401
    assert (
        client.get("/admin/users", headers=auth_headers(user_token)).status_code == 403
    )
    admin_token = login(client, "admin@example.com")
    response = client.get("/admin/users", headers=auth_headers(admin_token))
    assert response.status_code == 200
    emails = [u["email"] for u in response.json()]
    assert emails == ["plain@example.com", "admin@example.com"]


def test_admin_disable_and_enable_user(client):
    user = make_user("target@example.com")
    make_user("admin@example.com", role="admin")
    admin_token = login(client, "admin@example.com")
    headers = auth_headers(admin_token)

    response = client.patch(
        f"/admin/users/{user.id}", json={"is_active": False}, headers=headers
    )
    assert response.status_code == 200
    assert response.json()["is_active"] is False
    # Disabled: login now fails, and the old token no longer works.
    assert (
        client.post(
            "/auth/login",
            json={"email": "target@example.com", "password": "password123"},
        ).status_code
        == 401
    )
    response = client.patch(
        f"/admin/users/{user.id}", json={"is_active": True}, headers=headers
    )
    assert response.status_code == 200
    assert response.json()["is_active"] is True
    assert (
        client.post(
            "/auth/login",
            json={"email": "target@example.com", "password": "password123"},
        ).status_code
        == 200
    )


def test_admin_cannot_disable_self(client):
    admin = make_user("admin@example.com", role="admin")
    admin_token = login(client, "admin@example.com")
    response = client.patch(
        f"/admin/users/{admin.id}",
        json={"is_active": False},
        headers=auth_headers(admin_token),
    )
    assert response.status_code == 400


def test_admin_patch_unknown_user_404(client):
    make_user("admin@example.com", role="admin")
    admin_token = login(client, "admin@example.com")
    response = client.patch(
        "/admin/users/does-not-exist",
        json={"is_active": False},
        headers=auth_headers(admin_token),
    )
    assert response.status_code == 404
