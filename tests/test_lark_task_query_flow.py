"""agent-lark skill map — task-query flow for in-chat task questions."""

from runtime.skills import load_skills
from runtime.subgraph.flow import derive_flows


def test_agent_lark_skill_map_has_task_query_flow():
    skill_map = {s.name: s for s in load_skills()}["agent-lark"].map
    assert skill_map is not None
    node = skill_map.node_by_id("task-query")
    assert node is not None
    assert node.terminal is True
    assert set(node.tools) == {
        "lark_list_tasklists",
        "lark_list_tasklist_tasks",
        "lark_get_task",
        "lark_reply_message",
    }

    route = skill_map.node_by_id("route")
    gotos = {b.goto for b in route.branches}
    assert "task-query" in gotos
    task_query_when = next(b.when for b in route.branches if b.goto == "task-query")
    assert "task" in task_query_when.lower()
    assert "remind_tasks" not in task_query_when

    flows = derive_flows(skill_map)
    by_id = {f.id: f for f in flows}
    assert by_id["task-query"].nodes[0].id == "task-query"
    assert "task-remind" in by_id
    remind_tools = {t for n in by_id["task-remind"].nodes for t in n.tools}
    assert "lark_send_message" in remind_tools
    assert "lark_list_tasklist_tasks" in remind_tools
    assert "lark_reply_message" not in remind_tools

    assert skill_map.reply_tools == ["lark_reply_message", "lark_send_message"]