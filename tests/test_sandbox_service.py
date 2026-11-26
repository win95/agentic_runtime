"""sandbox_service tests — sqlite (conftest) + fake OpenSandbox client.

No server, no SDK: the client seam takes a fake with the same three
methods (create/connect/destroy), so these tests pin the mapping
lifecycle (create once, re-attach, replace a dead sandbox, reap idle)
without a live OpenSandbox server.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.models import SessionSandbox
from app.services import sandbox_service


def _settings(**overrides):
    cfg = {
        "sandbox_enabled": True,
        "sandbox_api_key": "test-key",
        "sandbox_domain": "localhost:8080",
        "sandbox_image": "python:3.12-slim",
        "sandbox_use_server_proxy": False,
        "sandbox_idle_ttl_seconds": 3600,
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class _FakeCommands:
    def run(self, command, opts=None):  # noqa: ARG002
        return SimpleNamespace(
            logs=SimpleNamespace(stdout=["0"], stderr=[]),
            exit_code=0,
            result=[],
            error=None,
        )


class _FakeFiles:
    def list_directory(self, entry):  # noqa: ARG002
        return []


class _FakeHandle:
    def __init__(self, sandbox_id):
        self.id = sandbox_id
        self.commands = _FakeCommands()
        self.files = _FakeFiles()
        self.renewed = None

    def renew(self, timeout):
        self.renewed = timeout


class _FakeClient:
    def __init__(self):
        self.created = []
        self.connected = []
        self.destroyed = []
        self.fail_connect = set()
        self._n = 0

    def create(self, image, *, timeout_seconds=None):
        self._n += 1
        self.created.append((image, timeout_seconds))
        return _FakeHandle(f"sandbox-{self._n}")

    def renew(self, handle, timeout_seconds):
        handle.renew(timeout_seconds)

    def connect(self, sandbox_id):
        self.connected.append(sandbox_id)
        if sandbox_id in self.fail_connect:
            raise RuntimeError("sandbox gone")
        return _FakeHandle(sandbox_id)

    def destroy(self, handle):
        self.destroyed.append(handle.id)


def _resolve(db_session, client, session_id="sess-1", **settings_overrides):
    return sandbox_service.resolve_session_sandbox(
        db_session,
        session_id=session_id,
        agent_id="fault_investigator",
        mission_id="fault_analysis",
        settings=_settings(**settings_overrides),
        client=client,
    )


def test_disabled_returns_none_without_touching_server(db_session):
    client = _FakeClient()
    assert _resolve(db_session, client, sandbox_enabled=False) is None
    assert client.created == [] and client.connected == []


def test_first_use_creates_sandbox_and_row(db_session):
    client = _FakeClient()
    backend = _resolve(db_session, client)
    assert backend is not None
    assert backend.id == "sandbox-1"
    row = db_session.get(SessionSandbox, "sess-1")
    assert row is not None
    assert row.sandbox_id == "sandbox-1"
    assert row.status == "active"
    assert row.agent_id == "fault_investigator"
    assert row.mission_id == "fault_analysis"
    assert client.created[0][0] == "python:3.12-slim"
    assert client.created[0][1] == 3600


def test_second_use_reconnects_to_same_sandbox(db_session):
    client = _FakeClient()
    first = _resolve(db_session, client)
    second = _resolve(db_session, client)
    assert first.id == second.id == "sandbox-1"
    assert len(client.created) == 1
    assert client.connected == ["sandbox-1"]


def test_dead_sandbox_is_replaced(db_session):
    client = _FakeClient()
    _resolve(db_session, client)
    client.fail_connect.add("sandbox-1")
    backend = _resolve(db_session, client)
    assert backend.id == "sandbox-2"
    row = db_session.get(SessionSandbox, "sess-1")
    assert row.sandbox_id == "sandbox-2"
    assert row.status == "active"


def test_destroy_marks_row_destroyed(db_session):
    client = _FakeClient()
    _resolve(db_session, client)
    assert sandbox_service.destroy_session_sandbox(
        db_session, "sess-1", settings=_settings(), client=client
    ) is True
    assert db_session.get(SessionSandbox, "sess-1").status == "destroyed"
    assert client.destroyed == ["sandbox-1"]
    # Second destroy: nothing to destroy.
    assert sandbox_service.destroy_session_sandbox(
        db_session, "sess-1", settings=_settings(), client=client
    ) is False


def test_reap_idle_destroys_only_idle(db_session):
    client = _FakeClient()
    _resolve(db_session, client, session_id="sess-old")
    _resolve(db_session, client, session_id="sess-fresh")
    old = db_session.get(SessionSandbox, "sess-old")
    old.last_used_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        hours=2
    )
    db_session.commit()
    result = sandbox_service.reap_idle_sandboxes(
        db_session, settings=_settings(), client=client
    )
    assert result == {"checked": 1, "destroyed": 1, "failed": []}
    assert db_session.get(SessionSandbox, "sess-old").status == "destroyed"
    assert db_session.get(SessionSandbox, "sess-fresh").status == "active"
