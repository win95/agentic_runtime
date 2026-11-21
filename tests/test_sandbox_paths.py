"""Sandbox path resolution for Files API."""

from runtime.sandbox.paths import normalize_output_export_path, resolve_files_api_path


def test_resolve_relative_output_file():
    assert resolve_files_api_path("output/a.html") == "/output/a.html"


def test_resolve_workspace_output_dir():
    assert resolve_files_api_path("/workspace/output") == "/output"


def test_normalize_export_top_level():
    assert normalize_output_export_path("/workspace/output/x.html") == "/output/x.html"