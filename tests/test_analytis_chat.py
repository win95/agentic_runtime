"""analytis_chat CLI — HTML chart check helper (no live API)."""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_chat_module():
    spec = importlib.util.spec_from_file_location(
        "analytis_chat", ROOT / "scripts" / "manual" / "analytis_chat.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_html_has_chart_detects_chartjs():
    chat = _load_chat_module()
    html = '<canvas id="price"></canvas><script>new Chart(ctx, {type: "line"})</script>'
    assert chat._html_has_chart(html) is True


def test_html_has_chart_rejects_chartless_html():
    chat = _load_chat_module()
    assert chat._html_has_chart("<html><body><table></table></body></html>") is False
    # Canvas without the Chart constructor is not a Chart.js chart.
    assert chat._html_has_chart('<canvas id="x"></canvas>') is False
    # A mere mention of the constructor in text is not a chart either.
    assert chat._html_has_chart("<p>new Chart( is a call</p>") is False
    assert chat._html_has_chart("") is False
