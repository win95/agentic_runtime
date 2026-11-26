"""analytis manual helpers — HTML chart + export_checks (no live API)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANUAL = ROOT / "scripts" / "manual"
if str(MANUAL) not in sys.path:
    sys.path.insert(0, str(MANUAL))


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "analytis_common", MANUAL / "analytis_common_1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_html_has_chart_detects_chartjs():
    common = _load_common()
    html = '<canvas id="price"></canvas><script>new Chart(ctx, {type: "line"})</script>'
    assert common.html_has_chart(html) is True


def test_html_has_chart_rejects_chartless_html():
    common = _load_common()
    assert common.html_has_chart("<html><body><table></table></body></html>") is False
    assert common.html_has_chart('<canvas id="x"></canvas>') is False
    assert common.html_has_chart("<p>new Chart( is a call</p>") is False
    assert common.html_has_chart("") is False


def test_export_checks_passes_with_full_payload():
    common = _load_common()
    result = {
        "output": {
            "answer": "Tóm tắt giá cà phê.",
            "files": [
                {
                    "name": "report.html",
                    "content": '<canvas id="c"></canvas><script>new Chart(x)</script>',
                },
                {"name": "data.csv", "content": "a,b\n1,2"},
                {"name": "summary.md", "content": "# Tóm tắt"},
            ],
        }
    }
    checks = dict(common.export_checks(result, succeeded=True))
    assert checks["run SUCCEEDED"] is True
    assert checks["answer không rỗng"] is True
    assert checks["files có .html"] is True
    assert checks["báo cáo HTML có biểu đồ Chart.js"] is True


def test_export_checks_fails_without_chart():
    common = _load_common()
    result = {
        "output": {
            "answer": "ok",
            "files": [
                {"name": "report.html", "content": "<table></table>"},
                {"name": "data.csv", "content": "x"},
                {"name": "summary.md", "content": "y"},
            ],
        }
    }
    checks = dict(common.export_checks(result, succeeded=True))
    assert checks["báo cáo HTML có biểu đồ Chart.js"] is False