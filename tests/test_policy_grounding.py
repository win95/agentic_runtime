"""Policy grounding — no invented policy when documents are missing."""

import importlib.util
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from app.config import settings
from runtime.guest_locale import detect_guest_locale
from runtime.policy_grounding import (
    apply_policy_grounding,
    policy_unavailable_reply,
    search_query_from_guest_message,
)


def _load_mock_server():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "mock_hotel_mcp_server.py"
    spec = importlib.util.spec_from_file_location("mock_hotel_mcp_server", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mock_hotel_mcp_server"] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_search_query_from_guest_message_tokens():
    q = search_query_from_guest_message("Chính sách hủy phòng thế nào?")
    assert "hủy" in q.casefold() or "chính" in q.casefold()
    assert "POL" not in q


def test_apply_policy_grounding_no_docs_unavailable():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "1",
                    "name": "search_policy_documents",
                    "args": {"query": "hủy chính sách"},
                }
            ],
        ),
        ToolMessage(
            content=json.dumps({"documents": [], "count": 0}),
            tool_call_id="1",
            name="search_policy_documents",
        ),
    ]
    guest_message = "Chính sách hủy?"
    locale = detect_guest_locale(guest_message)
    reply, notes = apply_policy_grounding(
        "Khách được hoàn tiền 100% trong 24 giờ.",
        messages,
        guest_message=guest_message,
        tools_by_name=None,
        guest_locale=locale,
    )
    assert reply == policy_unavailable_reply(locale=locale)
    assert reply == settings.policy_unavailable_reply_template_vi.strip()
    assert any("unavailable" in n for n in notes)


def test_apply_policy_grounding_keeps_supported_sentence():
    doc_text = "Hủy trước 3 ngày: phí một đêm."
    messages = [
        ToolMessage(
            content=json.dumps({"doc_id": "POL-CANCEL", "content": doc_text}),
            tool_call_id="1",
            name="read_policy_document",
        ),
    ]
    reply, _ = apply_policy_grounding(
        "Hủy trước 3 ngày sẽ phí một đêm. Hoàn tiền 100% trong mọi trường hợp.",
        messages,
        guest_message="Chính sách hủy?",
        guest_locale="vi",
    )
    assert "3 ngày" in reply
    assert "100%" not in reply


def test_apply_policy_grounding_composes_from_pol_cancel_mock():
    mock_server = _load_mock_server()
    doc = mock_server.read_policy_document("POL-CANCEL")
    content = doc["content"]
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "1",
                    "name": "search_policy_documents",
                    "args": {"query": "hủy"},
                },
                {"id": "2", "name": "read_policy_document", "args": {"doc_id": "POL-CANCEL"}},
            ],
        ),
        ToolMessage(
            content=json.dumps(
                {"documents": [{"doc_id": "POL-CANCEL"}], "count": 1}
            ),
            tool_call_id="1",
        ),
        ToolMessage(
            content=json.dumps(doc),
            tool_call_id="2",
        ),
    ]
    reply, notes = apply_policy_grounding(
        "Cancellation is free anytime with full refund within 24 hours.",
        messages,
        guest_message="Chính sách hủy phòng thế nào?",
        guest_locale="vi",
    )
    assert policy_unavailable_reply(locale="en").casefold() not in reply.casefold()
    assert "3 ngày" in content
    assert "3 ngày" in reply
    assert "1 đêm" in content
    assert "1 đêm" in reply
    assert any("composed" in n for n in notes)