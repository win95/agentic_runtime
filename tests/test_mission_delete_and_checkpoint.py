"""Regression tests for two bug fixes (2026-11-09).

Bug 1 — DELETE /missions/{id} returned 500 (IntegrityError) when the
mission had executions. It now returns 409 Conflict, and the mission and
its executions are left intact. The service checks the execution count
first, so the 409 is deterministic on every backend: SQLite does not
enforce foreign keys by default, so the IntegrityError catch alone would
not fire in these tests.

Bug 2 — the Postgres checkpointer was silently disabled: in
langgraph-checkpoint-postgres >= 3, `PostgresSaver.from_conn_string()` is
itself a context manager, so calling `.setup()` on it raised
AttributeError. `get_checkpointer()` is now a context manager and
`run_mission()` holds it open for the whole `graph.invoke`.
"""

import sys
import types
from contextlib import contextmanager

import pytest

import infra.repositories as repositories
import runtime.mission_graph as mission_graph
from app import models
from app.config import settings
from tests.test_mission_graph import INPUT, MISSION

MISSION_PAYLOAD = {"id": "fault_analysis", "name": "Fault Analysis"}


def _create_mission(client):
    response = client.post("/missions", json=MISSION_PAYLOAD)
    assert response.status_code == 201
    return response.json()


# --- Bug 1: DELETE /missions/{id} ---------------------------------------


def test_delete_mission_with_executions_returns_409(client, db_session):
    _create_mission(client)
    db_session.add(
        models.Execution(
            id="exec_1",
            mission_id="fault_analysis",
            mission_version=1,
            status="SUCCEEDED",
        )
    )
    db_session.commit()

    response = client.delete("/missions/fault_analysis")
    assert response.status_code == 409
    assert "1 execution(s)" in response.json()["detail"]

    # Nothing was deleted: the mission and its execution are intact.
    assert client.get("/missions/fault_analysis").status_code == 200
    assert db_session.get(models.Execution, "exec_1") is not None


def test_delete_mission_without_executions_returns_204(client):
    _create_mission(client)

    response = client.delete("/missions/fault_analysis")
    assert response.status_code == 204
    assert client.get("/missions/fault_analysis").status_code == 404


def test_delete_unknown_mission_returns_404(client):
    response = client.delete("/missions/nope")
    assert response.status_code == 404


# --- Bug 2: get_checkpointer is a context manager -----------------------


def _fake_postgres_module(*, fail_enter=False, fail_setup=False):
    """Stand-in for `langgraph.checkpoint.postgres` whose PostgresSaver
    mirrors the >= 3 API: `from_conn_string` is a context manager that
    yields the saver and closes the connection on exit."""
    module = types.ModuleType("langgraph.checkpoint.postgres")
    created: list = []

    class FakeSaver:
        def __init__(self):
            self.setup_calls = 0
            self.closed = False

        def setup(self):
            self.setup_calls += 1
            if fail_setup:
                raise RuntimeError("relation does not exist")

        @classmethod
        @contextmanager
        def from_conn_string(cls, conn_string, *, pipeline=False):
            if fail_enter:
                raise RuntimeError("connection refused")
            saver = cls()
            created.append(saver)
            try:
                yield saver
            finally:
                saver.closed = True

    module.PostgresSaver = FakeSaver
    module.created = created
    return module


def test_get_checkpointer_context_manager(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_enabled", True)
    fake = _fake_postgres_module()
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", fake)

    with mission_graph.get_checkpointer() as saver:
        assert saver is not None
        assert saver.setup_calls == 1
        # Still open while the context is held (the old code called
        # .setup() on the context manager itself and failed here).
        assert saver.closed is False
    assert saver.closed is True
    assert len(fake.created) == 1


def test_get_checkpointer_disabled_yields_none():
    # conftest sets CHECKPOINT_ENABLED=false.
    assert settings.checkpoint_enabled is False
    with mission_graph.get_checkpointer() as saver:
        assert saver is None


def test_get_checkpointer_connection_failure_yields_none(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_enabled", True)
    fake = _fake_postgres_module(fail_enter=True)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", fake)

    # Best-effort: a dead database must not crash the run.
    with mission_graph.get_checkpointer() as saver:
        assert saver is None


def test_get_checkpointer_setup_failure_yields_none(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_enabled", True)
    fake = _fake_postgres_module(fail_setup=True)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", fake)

    with mission_graph.get_checkpointer() as saver:
        assert saver is None


# --- Bug 2: run_mission holds the checkpointer open during invoke -------


def _stub_repositories(monkeypatch):
    """Same repository-shaped demo rows as tests/test_mission_graph.py."""
    monkeypatch.setattr(
        repositories,
        "search_fault_history",
        lambda *a, **k: (
            [
                {"id": "fh-1", "code": "F-101", "title": "Door sensor fault"},
                {"id": "fh-2", "code": "F-204", "title": "Motor overheating"},
            ],
            False,
        ),
    )
    monkeypatch.setattr(
        repositories,
        "get_maintenance_records",
        lambda *a, **k: ([{"id": "mr-1", "action": "Sensor cleaned"}], False),
    )
    monkeypatch.setattr(
        repositories,
        "search_documents",
        lambda *a, **k: ([{"doc_id": "d1", "title": "Door sensor doc"}], False),
    )
    monkeypatch.setattr(
        repositories,
        "query_asset_relationships",
        lambda *a, **k: ([{"from": "O21", "type": "HAS_COMPONENT", "to": "motor-M2"}], False),
    )


def test_run_mission_keeps_checkpointer_open(monkeypatch):
    """The checkpointer must be live during graph.invoke, not closed
    before it: a recording MemorySaver must actually receive writes."""
    from langgraph.checkpoint.memory import MemorySaver

    _stub_repositories(monkeypatch)

    class RecordingMemorySaver(MemorySaver):
        def __init__(self):
            super().__init__()
            self.put_calls = 0
            self.put_writes_calls = 0

        def put(self, config, checkpoint, metadata, new_versions):
            self.put_calls += 1
            return super().put(config, checkpoint, metadata, new_versions)

        def put_writes(self, config, writes, task_id, task_path=""):
            self.put_writes_calls += 1
            return super().put_writes(config, writes, task_id, task_path)

    saver = RecordingMemorySaver()

    @contextmanager
    def _recording_checkpointer():
        yield saver

    monkeypatch.setattr(mission_graph, "get_checkpointer", _recording_checkpointer)

    final = mission_graph.run_mission(MISSION, INPUT, "exec_test_checkpoint_recording")
    assert final["status"] == "SUCCEEDED", f"errors: {final.get('errors')}"
    assert saver.put_calls >= 1, "checkpointer received no writes during invoke"
    assert saver.put_writes_calls >= 1


# --- Bug 3 (2026-11-11): SQLAlchemy URL is not a libpq DSN --------------
def test_to_libpq_dsn_strips_sqlalchemy_driver():
    """`postgresql+psycopg://...` is not parseable by libpq: psycopg
    raises `missing "=" after "postgresql+psycopg://..."`. The helper
    strips the driver so from_conn_string receives a plain DSN."""
    converted = mission_graph._to_libpq_dsn(
        "postgresql+psycopg://postgres:postgres@localhost:5432/agentic_runtime"
    )
    assert converted == "postgresql://postgres:postgres@localhost:5432/agentic_runtime"
    # Already-plain DSNs and non-postgres URLs pass through untouched.
    assert mission_graph._to_libpq_dsn(
        "postgresql://postgres:postgres@localhost:5432/agentic_runtime"
    ) == "postgresql://postgres:postgres@localhost:5432/agentic_runtime"
    assert mission_graph._to_libpq_dsn("sqlite:///:memory:") == "sqlite:///:memory:"
    # Unparseable input never raises — the caller's warning path handles it.
    assert mission_graph._to_libpq_dsn("not a url at all") == "not a url at all"


def test_to_libpq_dsn_result_parses_as_libpq():
    """The converted DSN must actually parse with psycopg (the exact
    failure tran reported). Skipped when psycopg is not installed."""
    psycopg = pytest.importorskip("psycopg")
    from psycopg.conninfo import conninfo_to_dict

    converted = mission_graph._to_libpq_dsn(
        "postgresql+psycopg://postgres:postgres@localhost:5432/agentic_runtime"
    )
    parsed = conninfo_to_dict(converted)
    assert parsed["dbname"] == "agentic_runtime"
    assert parsed["host"] == "localhost"
    with pytest.raises(psycopg.ProgrammingError, match='missing "=" after'):
        conninfo_to_dict(
            "postgresql+psycopg://postgres:postgres@localhost:5432/agentic_runtime"
        )
