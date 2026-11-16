"""ToolCallCap denies identical successful repeats (convergence)."""

from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from runtime.subgraph.common import ToolCallCap


def _req(name: str, args: dict, call_id: str = "c1"):
    return SimpleNamespace(
        tool_call={"name": name, "args": args, "id": call_id},
    )


def test_cap_denies_second_identical_success():
    cap = ToolCallCap(max_calls=10)
    seen: list = []

    def handler(_request):
        seen.append(1)
        return ToolMessage(content='{"guest_id": "G-1"}', tool_call_id="c1")

    first = cap.wrap_tool_call(_req("get_booking", {"id": "B1"}), handler)
    assert isinstance(first, ToolMessage)
    second = cap.wrap_tool_call(_req("get_booking", {"id": "B1"}, "c2"), handler)
    assert isinstance(second, ToolMessage)
    assert second.status == "error"
    assert "Prior result" in str(second.content)
    assert len(seen) == 1