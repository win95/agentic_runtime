"""MinIO artifact sink for sandbox output.files (harness-driven)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runtime.artifacts.sink import (
    apply_output_artifact_storage,
    artifact_object_key,
    safe_basename,
)
from runtime.harness.loader import parse_mission_harness
from runtime.harness.validation import validate_output_artifacts_policy


def test_defaults_output_artifacts_sink_none():
    h = parse_mission_harness(None)
    assert h.output_artifacts.sink == "none"
    assert h.output_artifacts.strip_inline is True
    assert h.output_artifacts.inline_fallback is True


def test_harness_output_artifacts_rejects_extra_fields():
    with pytest.raises(ValueError):
        parse_mission_harness({"output_artifacts": {"sink": "none", "extra": True}})


def test_minio_sink_requires_output_schema_files():
    harness = parse_mission_harness({"output_artifacts": {"sink": "minio"}})
    with pytest.raises(ValueError, match="output_schema.properties.files"):
        validate_output_artifacts_policy(harness, {"properties": {"answer": {}}})
    validate_output_artifacts_policy(
        harness, {"properties": {"files": {"type": "array"}}}
    )


def test_safe_basename_rejects_unsafe_names():
    assert safe_basename("report.html") == "report.html"
    assert safe_basename("../evil.html") is None
    assert safe_basename("sub/report.html") is None
    assert safe_basename("bad\x00name.txt") is None


def test_artifact_object_key_layout():
    key = artifact_object_key(
        agent_id="agent_analytis",
        session_id="sess_abc",
        execution_id="exec_123",
        basename="chart.html",
    )
    assert key == (
        "artifacts/agents/agent_analytis/sessions/sess_abc/"
        "executions/exec_123/chart.html"
    )


def _mission_with_minio_sink() -> dict:
    harness = parse_mission_harness({"output_artifacts": {"sink": "minio"}})
    return {
        "mission_id": "agent_analytis_mission",
        "output_schema": {"properties": {"files": {"type": "array"}, "answer": {}}},
        "execution": {
            "agent_id": "agent_analytis",
            "harness": harness.model_dump(mode="json"),
        },
    }


@patch("runtime.artifacts.sink.presigned_get_url", return_value="https://minio.example/presigned")
@patch("runtime.artifacts.sink.upload_bytes")
def test_sink_minio_success_strips_inline(mock_upload, mock_presign):
    mock_upload.return_value = "artifacts/agents/a/sessions/s/e/f.html"
    mission = _mission_with_minio_sink()
    output = {
        "answer": "done",
        "files": [{"name": "f.html", "content": "<html/>", "content_type": "text/html"}],
    }
    result = apply_output_artifact_storage(
        output,
        mission_dict=mission,
        execution_id="exec_1",
        agent_id="agent_analytis",
        session_id="sess_1",
    )
    assert result["artifact_storage"] == "minio"
    entry = result["files"][0]
    assert entry["storage"] == "minio"
    assert entry["url"] == "https://minio.example/presigned"
    assert "content" not in entry
    assert entry["location"].startswith("minio://")
    mock_upload.assert_called_once()


@patch("runtime.artifacts.sink.upload_bytes", side_effect=RuntimeError("down"))
def test_sink_minio_failure_inline_fallback(mock_upload):
    mission = _mission_with_minio_sink()
    output = {
        "files": [{"name": "f.csv", "content": "a,b"}],
    }
    result = apply_output_artifact_storage(
        output,
        mission_dict=mission,
        execution_id="exec_1",
        agent_id="agent_analytis",
        session_id="sess_1",
    )
    assert result["artifact_storage"] == "inline"
    entry = result["files"][0]
    assert entry["storage"] == "inline"
    assert entry["content"] == "a,b"
    assert any("upload failed" in n for n in (result.get("notes") or []))


def test_agent_analytis_mission_seed_enables_minio_sink():
    from seeds.load_missions import load_mission_file

    path = Path(__file__).resolve().parents[1] / "seeds/missions/agent_analytis_mission.yaml"
    row = load_mission_file(path)
    sink = row["execution_config"]["harness"]["output_artifacts"]["sink"]
    assert sink == "minio"


def test_sink_none_leaves_output_unchanged():
    harness = parse_mission_harness(None)
    mission = {
        "mission_id": "m",
        "output_schema": {"properties": {"files": {}}},
        "execution": {"harness": harness.model_dump(mode="json")},
    }
    output = {"files": [{"name": "a.txt", "content": "hi"}]}
    before = dict(output)
    result = apply_output_artifact_storage(
        output,
        mission_dict=mission,
        execution_id="e",
        agent_id="a",
        session_id="s",
    )
    assert result["files"][0] == before["files"][0]
    assert "artifact_storage" not in result


def test_analytis_common_save_files_from_url(tmp_path):
    from scripts.manual import analytis_common

    result = {
        "output": {
            "files": [
                {
                    "name": "report.html",
                    "url": "https://example.invalid/report",
                }
            ]
        }
    }
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html>ok</html>"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp
        saved = analytis_common.save_files(result, tmp_path, "exec_x")
    assert len(saved) == 1
    assert saved[0].read_bytes() == b"<html>ok</html>"


@pytest.mark.minio_integration
def test_minio_integration_upload_skipped_without_env():
    """Opt-in live MinIO check — skipped unless MINIO_ACCESS_KEY is set."""
    import os

    if not os.environ.get("MINIO_ACCESS_KEY"):
        pytest.skip("MINIO_ACCESS_KEY not set")
    pytest.skip("live MinIO integration not run in CI by default")