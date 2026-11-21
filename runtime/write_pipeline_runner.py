"""Generic write pipelines declared on skill maps."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from runtime.scope import get_bound_guest_id
from runtime.session_write_pending import get_pending, merge_tool_result, save_pending
from runtime.write_idempotency import invoke_idempotent_write, write_idempotency_key

logger = logging.getLogger(__name__)


@dataclass
class WritePipeline:
    """quote -> (confirm) -> automated write, declared under map.pipelines."""

    id: str
    quote_tools: list[str] = field(default_factory=list)
    confirm_node: str = ""
    write_node: str = ""
    write_tool: str = ""
    write_arg_fields: list[str] = field(default_factory=list)
    arg_paths: dict[str, list] = field(default_factory=dict)


def parse_pipelines(raw, skill_name: str) -> list[WritePipeline]:
    if not isinstance(raw, list):
        return []
    out: list[WritePipeline] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("id")
        if not isinstance(pid, str) or not pid.strip():
            logger.warning("Skill %s: pipeline without id; dropping", skill_name)
            continue
        quote_tools = entry.get("quote_tools") or []
        if not isinstance(quote_tools, list):
            quote_tools = []
        quote_tools = [str(t).strip() for t in quote_tools if str(t).strip()]
        write_tool = str(entry.get("write_tool") or "").strip()
        write_node = str(entry.get("write_node") or "").strip()
        confirm_node = str(entry.get("confirm_node") or "").strip()
        arg_fields = entry.get("write_arg_fields") or []
        if not isinstance(arg_fields, list):
            arg_fields = []
        arg_fields = [str(f).strip() for f in arg_fields if str(f).strip()]
        arg_paths_raw = entry.get("arg_paths") or {}
        arg_paths: dict[str, list] = {}
        if isinstance(arg_paths_raw, dict):
            for key, path in arg_paths_raw.items():
                if isinstance(key, str) and isinstance(path, list) and path:
                    arg_paths[key.strip()] = path
        out.append(
            WritePipeline(
                id=pid.strip(),
                quote_tools=quote_tools,
                confirm_node=confirm_node,
                write_node=write_node,
                write_tool=write_tool,
                write_arg_fields=arg_fields,
                arg_paths=arg_paths,
            )
        )
    return out


def pipeline_for_write_node(pipelines: list[WritePipeline], node_id: str) -> WritePipeline | None:
    for pipe in pipelines or []:
        if pipe.write_node == node_id:
            return pipe
    return None


def _tool_results(messages: list) -> list[tuple[str, dict]]:
    results_by_id: dict[str, ToolMessage] = {}
    for message in messages or []:
        call_id = getattr(message, "tool_call_id", None)
        if call_id:
            results_by_id[str(call_id)] = message
    out: list[tuple[str, dict]] = []
    for message in messages or []:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            if not name:
                continue
            tool_msg = results_by_id.get(str(call_id)) if call_id else None
            if tool_msg is None:
                continue
            content = getattr(tool_msg, "content", "")
            data = _content_dict(content)
            if data is not None:
                out.append((str(name), data))
    return out


def _content_dict(content) -> dict | None:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        text = content.strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def capture_quote_tools(
    session_id: str,
    pipeline: WritePipeline,
    messages: list,
) -> None:
    if not session_id:
        return
    quote_set = set(pipeline.quote_tools)
    for name, data in _tool_results(messages):
        if name in quote_set:
            merge_tool_result(session_id, pipeline.id, name, data)


def mark_awaiting_confirm(session_id: str, pipeline: WritePipeline) -> None:
    pending = get_pending(session_id, pipeline.id)
    if pending is None:
        return
    save_pending(
        session_id,
        pipeline.id,
        status="awaiting_confirm",
        payload=pending.get("payload") or {},
        idempotency_key=pending.get("idempotency_key"),
    )


def _resolve_arg_path(tools: dict, path: list) -> object | None:
    if not path:
        return None
    tool_name = str(path[0])
    cur = tools.get(tool_name)
    for step in path[1:]:
        if cur is None:
            return None
        if isinstance(step, int) and isinstance(cur, list):
            cur = cur[step] if 0 <= step < len(cur) else None
        elif isinstance(cur, dict):
            cur = cur.get(step)
        else:
            return None
    return cur


def mark_pipeline_confirmed(session_id: str, pipeline_id: str) -> dict:
    pending = get_pending(session_id, pipeline_id)
    if pending is None:
        return {"error": "no quoted data in session for this pipeline"}
    if pending.get("status") == "written":
        return {"error": "booking already written for this quote"}
    save_pending(
        session_id,
        pipeline_id,
        status="confirmed",
        payload=pending.get("payload") or {},
        idempotency_key=pending.get("idempotency_key"),
    )
    return {"confirmed": True, "pipeline_id": pipeline_id}


def make_confirm_tool(
    session_id: str,
    pipeline_id: str,
    tool_name: str,
    description: str,
) -> StructuredTool:
    """The session-scoped confirm tool for one pipeline.

    The name comes from the agent (pipeline-id-based); this factory
    only builds the tool. Real instances are per flow agent — the
    registry holds only a stub under the same name (see
    :func:`ensure_confirm_tool_registered`), so the L2 call-time gate
    knows the tool.
    """

    def _confirm() -> str:
        payload = mark_pipeline_confirmed(session_id, pipeline_id)
        return json.dumps(payload, ensure_ascii=False)

    return StructuredTool.from_function(
        name=tool_name,
        description=description,
        func=_confirm,
    )


def confirm_tool_stub(tool_name: str, description: str) -> StructuredTool:
    """Registry stub; real instances are session-scoped per flow agent."""

    def _missing() -> str:
        return json.dumps(
            {
                "error": f"{tool_name} requires an active session context",
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        name=tool_name,
        description=description,
        func=_missing,
    )


def ensure_confirm_tool_registered(tool_name: str, description: str) -> None:
    """Register the confirm stub (idempotent) so ``enforce_layer`` knows it.

    The stub is registered under every guest-output capability from the
    mission seeds — the same set the old import-time stub used. The
    per-session tool itself is never registered; only the stub is.
    """
    from runtime.harness.loader import guest_output_capabilities_from_seed_files
    from runtime.tools.registry import registry

    if registry.get(tool_name) is not None:
        return
    stub = confirm_tool_stub(tool_name, description)
    for capability in guest_output_capabilities_from_seed_files():
        registry.register(stub, capability=capability, read_only=True)


def build_write_args(
    pipeline: WritePipeline,
    pending_payload: dict,
    *,
    session_id: str,
    field_resolvers: dict | None = None,
) -> dict | None:
    """Assemble the write tool's args from the quoted data.

    Per field, in order: an agent-supplied ``field_resolvers`` entry
    (a ``tools -> value`` function — e.g. the hotel agent resolving
    ``room_type`` from the rate row the guest asked for), the map's
    ``arg_paths``, then a scan of the quoted tool results for a field
    of the same name. The engine knows no field names of its own.
    """
    tools = (pending_payload or {}).get("tools") or {}
    args: dict = {}
    if "guest_id" in pipeline.write_arg_fields:
        bound = get_bound_guest_id(session_id)
        if isinstance(bound, str) and bound:
            args["guest_id"] = bound
    for field_name in pipeline.write_arg_fields:
        if field_name in args:
            continue
        resolver = (field_resolvers or {}).get(field_name)
        if resolver is not None:
            val = resolver(tools)
            if val is not None:
                args[field_name] = val
                continue
        path = pipeline.arg_paths.get(field_name)
        if path:
            val = _resolve_arg_path(tools, path)
            if val is not None:
                args[field_name] = val
                continue
        for _tool, data in tools.items():
            if isinstance(data, dict) and field_name in data:
                args[field_name] = data[field_name]
                break
    missing = [f for f in pipeline.write_arg_fields if f not in args]
    if missing:
        logger.info(
            "Pipeline %s missing write fields %s (have %s)",
            pipeline.id,
            missing,
            list(args.keys()),
        )
        return None
    return args


def try_automated_write(
    pipeline: WritePipeline,
    *,
    session_id: str,
    execution_id: str,
    tools_by_name: dict,
    session_turns: int,
    agent=None,
    field_resolvers: dict | None = None,
    confirm_tool_name: str | None = None,
    success_template: str | None = None,
) -> tuple[str, list[str], dict | None]:
    """Run the pipeline write tool from pending state. Returns (answer, errors, raw).

    ``agent`` (when the caller has one) is passed through to the
    write preconditions, so the booking-ownership check runs through
    the agent's policy. ``field_resolvers`` / ``confirm_tool_name`` /
    ``success_template`` are the agent's flavour: the engine's
    defaults are neutral (no hotel nouns), the hotel agent supplies
    the room-type resolver, its pipeline-id confirm name and the
    guest-facing booking answer.
    """
    errors: list[str] = []
    pending = get_pending(session_id, pipeline.id)
    if pending is None:
        errors.append(
            f"Automated write for pipeline '{pipeline.id}' refused: "
            "no quoted data in session"
        )
        return "", errors, None
    status = str(pending.get("status") or "")
    if status == "written":
        payload = pending.get("payload") or {}
        prior = payload.get("write_result")
        if isinstance(prior, dict) and not prior.get("error"):
            booking_id = prior.get("booking_id")
            answer = _success_answer(success_template, booking_id)
            return answer, errors, prior
        errors.append(
            f"Automated write for pipeline '{pipeline.id}' refused: "
            "already written"
        )
        return "", errors, None
    if status not in ("confirmed",):
        detail = "guest confirmation required"
        if confirm_tool_name:
            detail += f" ({confirm_tool_name})"
        errors.append(
            f"Automated write for pipeline '{pipeline.id}' refused: {detail}"
        )
        return "", errors, None
    args = build_write_args(
        pipeline, pending.get("payload") or {}, session_id=session_id,
        field_resolvers=field_resolvers,
    )
    if args is None:
        errors.append(
            f"Automated write for pipeline '{pipeline.id}' refused: "
            "incomplete quoted fields"
        )
        return "", errors, None
    tool = tools_by_name.get(pipeline.write_tool)
    if tool is None:
        errors.append(f"Automated write tool '{pipeline.write_tool}' is not available")
        return "", errors, None
    from runtime.tool_preconditions import check_write_preconditions

    pre = check_write_preconditions(
        pipeline.write_tool,
        args,
        session_id=session_id,
        execution_id=execution_id,
        agent=agent,
    )
    if pre:
        errors.append(
            f"Automated write for pipeline '{pipeline.id}' refused: {pre}"
        )
        return "", errors, None
    payload = pending.get("payload") or {}
    write_attempt_id = (
        payload.get("write_attempt_id") or pending.get("write_attempt_id") or ""
    )
    idem = write_idempotency_key(
        write_attempt_id=write_attempt_id,
        execution_id=execution_id,
        write_tool=pipeline.write_tool,
        action="execute",
    )
    result = invoke_idempotent_write(tool, args, idempotency_key=idem)
    if isinstance(result, dict) and result.get("error"):
        errors.append(
            f"Automated {pipeline.write_tool} failed: {result.get('error')}"
        )
        return "", errors, result
    booking_id = result.get("booking_id") if isinstance(result, dict) else None
    answer = _success_answer(success_template, booking_id)
    save_pending(
        session_id,
        pipeline.id,
        status="written",
        payload={**(pending.get("payload") or {}), "write_result": result},
        idempotency_key=idem,
    )
    return answer, errors, result


def _success_answer(success_template: str | None, booking_id: str | None) -> str:
    """The write's success answer: ``{ref}`` is ``" (reference BK-1)"`` or ``""``."""
    template = success_template or "Write completed{ref}."
    ref = f" (reference {booking_id})" if booking_id else ""
    return template.format(ref=ref)