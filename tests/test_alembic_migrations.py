"""Alembic migration tests — upgrade/downgrade on a scratch SQLite DB.

Runs the real `alembic` CLI in a subprocess (same command `make
migrate` runs) with DATABASE_URL pointed at a tmp file, so the test
proves the migration chain works end-to-end rather than only checking
that the files exist.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def _alembic(tmp_path, *args):
    env = dict(os.environ)
    env["DATABASE_URL"] = f"sqlite:///{tmp_path}/migrate-test.db"
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )


def _tables(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        conn.close()


def _version(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT version_num FROM alembic_version_agentic_runtime"
        ).fetchall()
    finally:
        conn.close()


def test_alembic_upgrade_head_then_downgrade_base(tmp_path):
    db_path = tmp_path / "migrate-test.db"
    result = _alembic(tmp_path, "upgrade", "head")
    assert result.returncode == 0, result.stderr
    assert _version(db_path) == [("0007",)]
    tables = _tables(db_path)
    assert {"missions", "executions", "tools", "agent_versions"} <= tables
    assert "ontology_entities" in tables

    conn = sqlite3.connect(db_path)
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(tools)")]
        assert "read_only" in cols
        exec_cols = [row[1] for row in conn.execute("PRAGMA table_info(executions)")]
        assert "session_id" in exec_cols
        agent_cols = [row[1] for row in conn.execute("PRAGMA table_info(agents)")]
        assert "session_history" in agent_cols
        version_cols = [
            row[1] for row in conn.execute("PRAGMA table_info(agent_versions)")
        ]
        assert "session_history" in version_cols
        exec_indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(executions)")
        }
        assert "ix_executions_session_id" in exec_indexes
        indexes = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE tbl_name='agent_versions' AND type='index'"
            )
        ]
        assert any(
            "uq_agent_versions_agent_version" in name or "autoindex" in name
            for name in indexes
        )
        ontology_indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE tbl_name IN ('ontology_versions', "
                "'ontology_relationships') AND type='index'"
            )
        }
        assert "ix_ontology_versions_subject_version" in ontology_indexes
        assert "ix_ontology_relationships_target" in ontology_indexes
    finally:
        conn.close()

    result = _alembic(tmp_path, "downgrade", "base")
    assert result.returncode == 0, result.stderr
    assert _tables(db_path) == {"alembic_version_agentic_runtime"}


def test_alembic_current_matches_head(tmp_path):
    result = _alembic(tmp_path, "upgrade", "head")
    assert result.returncode == 0, result.stderr
    result = _alembic(tmp_path, "current")
    assert result.returncode == 0, result.stderr
    assert "0007" in result.stdout
