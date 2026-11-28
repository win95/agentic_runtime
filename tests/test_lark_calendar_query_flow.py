"""agent-lark skill map — calendar-query flow."""

from runtime.skills import load_skills
from runtime.subgraph.flow import derive_flows


def test_agent_lark_skill_map_has_calendar_query_flow():
    skill_map = {s.name: s for s in load_skills()}["agent-lark"].map
    node = skill_map.node_by_id("calendar-query")
    assert node is not None
    assert node.terminal is True
    assert set(node.tools) == {"lark_calendar_freebusy", "lark_reply_message"}

    route = skill_map.node_by_id("route")
    gotos = {b.goto for b in route.branches}
    assert "calendar-query" in gotos
    when = next(b.when for b in route.branches if b.goto == "calendar-query")
    assert len(when) > 40
    assert "rảnh" in when.lower() or "lịch" in when.lower()

    flows = derive_flows(skill_map)
    by_id = {f.id: f for f in flows}
    assert by_id["calendar-query"].nodes[0].id == "calendar-query"