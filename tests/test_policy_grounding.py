"""Policy grounding — no invented policy when documents are missing."""

import json

from langchain_core.messages import AIMessage, ToolMessage

from app.config import settings
from runtime.policy_grounding import (
    apply_policy_grounding,
    policy_unavailable_reply,
    search_query_from_guest_message,
)


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
    reply, notes = apply_policy_grounding(
        "Khách được hoàn tiền 100% trong 24 giờ.",
        messages,
        guest_message="Chính sách hủy?",
        tools_by_name=None,
    )
    assert reply == policy_unavailable_reply()
    assert reply == settings.policy_unavailable_reply_template.strip()
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
    )
    assert "3 ngày" in reply
    assert "100%" not in reply