"""Flow router parsing tests (``runtime/subgraph/flow.py``).

The router is a model call, so the parser is the contract between the
model's reply and the graph edge. These tests pin the two halves of the
v10/v11 fix:

- the instruction must not hand the model a template it can echo back
  (the 3B model appended ``FLOW: <choice>`` to its real declaration,
  which the strict parser then read as two votes);
- the parser still rejects a genuine "no choice" — two real
  declarations or an ``or FLOW:`` hedge — while ignoring a placeholder
  echo of the format template.

No LLM, no graph execution: pure text in, flow id out.
"""

from runtime.subgraph.flow import (
    _is_ambiguous_flow_declaration,
    parse_flow,
)


def test_parse_flow_plain_declaration():
    assert parse_flow("Đã kiểm tra phòng.\nFLOW: booking-dates") == "booking-dates"


def test_parse_flow_tolerates_bold_and_indent_and_trailing_prose():
    reply = "  **FLOW:** check-availability — vì khách hỏi giá"
    assert parse_flow(reply) == "check-availability"


def test_parse_flow_end_and_wait_sentinels():
    assert parse_flow("FLOW: END") == "END"
    assert parse_flow("FLOW: WAIT\n\nCần ngày trả phòng.") == "WAIT"


def test_parse_flow_rejects_two_declarations():
    assert parse_flow("FLOW: booking-dates\nFLOW: policy-lookup") is None
    assert _is_ambiguous_flow_declaration(
        "FLOW: booking-dates\nFLOW: policy-lookup"
    )


def test_parse_flow_rejects_alternative_hedge():
    reply = "FLOW: booking-dates or FLOW: WAIT"
    assert parse_flow(reply) is None
    assert _is_ambiguous_flow_declaration(reply)


def test_parse_flow_ignores_placeholder_template_echo():
    """v11 trace: the model chose booking-lookup, then echoed the
    instruction's own template as a second line. One real choice plus a
    placeholder is not ambiguous."""
    reply = "FLOW: booking-lookup\nFLOW: <choice>"
    assert not _is_ambiguous_flow_declaration(reply)
    assert parse_flow(reply) == "booking-lookup"


def test_placeholder_echo_variants_are_ignored():
    for placeholder in ("FLOW: <flow-id>", "**FLOW:** <id>"):
        reply = f"FLOW: booking-payment\n{placeholder}"
        assert parse_flow(reply) == "booking-payment", placeholder


def test_only_a_placeholder_is_no_choice():
    assert parse_flow("Tôi cần hỏi thêm.\nFLOW: <choice>") is None


def test_no_declaration_is_none():
    assert parse_flow("Khách muốn đặt phòng Deluxe.") is None
    assert parse_flow("") is None
    assert parse_flow(None) is None
