"""Test setup — environment first, then app imports.

Tests run against an in-memory SQLite DB (no Postgres/Redis needed) and
with API-key auth disabled. Celery dispatch is monkeypatched per test, so
no broker is needed either.
"""

import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["API_KEY"] = ""
os.environ["CELERY_BROKER_URL"] = "memory://"
os.environ["CELERY_RESULT_BACKEND"] = "cache+memory://"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["CHECKPOINT_ENABLED"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.db import SessionLocal, engine, get_db  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _isolate_tool_registry():
    """Restore the global tool registry after every test.

    ``load_mcp_tools()`` registers MCP tools into ``registry._tools``;
    a test that runs a mission while ``MCP_SERVER_URL`` is configured
    (e.g. a dev ``.env`` plus a running mock server) would otherwise
    leak those tools into every later test and break the "exactly the
    built-in set" assertions in alphabetical order.
    """
    from runtime.tools import registry as tool_registry

    snapshot = dict(tool_registry._tools)
    yield
    tool_registry._tools.clear()
    tool_registry._tools.update(snapshot)


@pytest.fixture(autouse=True)
def _disable_mcp(monkeypatch):
    """Keep tests off the MCP server a dev ``.env`` may point at.

    Without this, a mission test that happens to run while the mock (or
    a real) MCP server is up loads its tools, and the skill-map cache is
    then built with MCP tool names that the reception test expects to be
    dropped. Tests that exercise MCP opt in by monkeypatching these two
    settings themselves (``tests/test_mock_hotel_mcp.py``).
    """
    from app.config import settings

    monkeypatch.setattr(settings, "mcp_server_url", "")
    monkeypatch.setattr(settings, "mcp_tool_capabilities", "")


@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        # Clean all rows so tests are independent (same in-memory DB).
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
        session.close()


@pytest.fixture()
def client(db_session):
    app = create_app()

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
