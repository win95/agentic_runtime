"""Agent seed model follows LLM_MODEL from the environment.

The agent YAMLs write ``model: ${LLM_MODEL}``; the loader expands it
at seed time (env first, then settings.llm_model). A model change in
.env + re-seed must create a new AgentVersion (Option C), not silently
keep the old pinned model.
"""

import pathlib

import pytest

from app.config import settings
from app.core.db import SessionLocal
from app.services import agent_version_service as avs
from seeds import load_missions

AGENTS_DIR = pathlib.Path(load_missions.__file__).resolve().parent / "agents"


def test_load_agent_file_reads_llm_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-test")
    assert (
        load_missions.load_agent_file(AGENTS_DIR / "admin_bot.yaml")["model"]
        == "gpt-4o-test"
    )
    assert (
        load_missions.load_agent_file(AGENTS_DIR / "fault_investigator.yaml")["model"]
        == "gpt-4o-test"
    )


def test_load_agent_file_falls_back_to_settings_without_env(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    data = load_missions.load_agent_file(AGENTS_DIR / "admin_bot.yaml")
    assert data["model"] == settings.llm_model  # default gpt-4o-mini


def test_placeholder_inline_default_syntax(monkeypatch, tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "id: t1\nname: T\nmodel: ${LLM_MODEL:-my-fallback}\n", encoding="utf-8"
    )
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert load_missions.load_agent_file(path)["model"] == "my-fallback"
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-env")
    assert load_missions.load_agent_file(path)["model"] == "gpt-4o-env"


def test_unresolvable_placeholder_raises(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("id: t2\nname: T\nmodel: ${NO_SUCH_VAR_XYZ}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NO_SUCH_VAR_XYZ"):
        load_missions.load_agent_file(path)


def test_reseed_with_new_llm_model_creates_new_version(monkeypatch, db_session):
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-model-a")
    load_missions.main()
    db = SessionLocal()
    try:
        v1 = avs.get_active_version(db, "admin_bot")
        assert v1.version == 1 and v1.model == "gpt-4o-model-a"

        monkeypatch.setenv("LLM_MODEL", "gpt-4o-model-b")
        load_missions.main()
        v2 = avs.get_active_version(db, "admin_bot")
        assert v2.version == 2 and v2.model == "gpt-4o-model-b"
        assert avs.get_version(db, "admin_bot", 1).model == "gpt-4o-model-a"

        # Re-seed with the same env value creates nothing new.
        load_missions.main()
        assert avs.get_active_version(db, "admin_bot").version == 2
        assert len(avs.list_versions(db, "admin_bot")) == 2
    finally:
        db.close()
