"""Flow subgraph tests — replaces legacy runtime.subgraph.node compiler."""

from langchain_core.messages import AIMessage

from runtime.skills.map_plan import MapBranch, MapNode, SkillMap
from runtime.subgraph.flow import (
    ROUTE_TOOL_NAME,
    eligible_flow_ids,
    parse_route_tool_call,
    use_flow_subgraph,
)


def _mini_map():
    return SkillMap(
        skill_name="test",
        start="route",
        nodes=[
            MapNode(
                id="route",
                goal="route",
                branches=[MapBranch(when="book", goto="booking-dates")],
            ),
            MapNode(id="booking-dates", goal="dates", tools=[]),
        ],
    )


def test_use_flow_subgraph():
    assert use_flow_subgraph("hotel_reception", ["reception"]) is True
    assert use_flow_subgraph("fault_analysis", ["data-access"]) is False
    assert use_flow_subgraph("hotel_reception", []) is False


def test_eligible_flow_ids_skips_visited():
    from runtime.subgraph.flow import Flow

    flows = [Flow(id="a", nodes=[]), Flow(id="b", nodes=[])]
    assert eligible_flow_ids(flows, {"visited": []}) == ["a", "b"]
    assert eligible_flow_ids(flows, {"visited": ["a"]}) == ["b"]


def test_parse_route_tool_call():
    allowed = frozenset({"booking-dates", "END", "WAIT"})
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": ROUTE_TOOL_NAME, "args": {"flow_id": "booking-dates"}, "id": "1"}
        ],
    )
    assert parse_route_tool_call(msg, allowed) == "booking-dates"
    assert parse_route_tool_call(msg, frozenset({"END"})) is None