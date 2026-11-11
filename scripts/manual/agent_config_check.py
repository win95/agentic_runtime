"""Assert that a run really uses the agent config pinned in its AgentVersion.

The runtime is config-driven in four layers: the seed YAML (source), the
``agents`` row (live config), the active ``AgentVersion`` (the immutable
snapshot that pins prompt + model/temperature + allowed tools + tool
manifest + session-history limits), and the Deep Agent call (what the
model actually receives). This script checks all four, so a passing run
means "the LLM saw exactly the pinned config", not just "the run
succeeded".

Static checks (no execution, DB + seeds only):

- seed YAML matches the live ``agents`` row — catches "edited the YAML
  but forgot ``make seed``";
- live ``agents`` row matches the active ``AgentVersion`` — catches a
  row edited without creating a version;
- tool-manifest drift: live registry SHA-256 vs the version's pinned one;
- every ``allowed_tools`` entry exists in the registry (the gate can
  only pass tools that exist);
- mission-declared skills exist on disk and ``select_skills`` returns
  exactly them;
- mission-declared hooks resolve to built-in handlers;
- the version's ``session_history`` block parses and resolves to the
  limits the renderer will apply.

Live checks (two turns in one fresh session, so turn 2 has history):

- both executions SUCCEEDED with ``agent_version`` == the active version
  and ``agent_version_drift == false``;
- the DB ``llm_usage.model`` is the pinned model;
- from the run's MLflow trace (``mlflow.sourceRun`` links the trace to
  the ``execution-<id>`` run tagged with ``execution_id``):
  - the system prompt the model received **starts with** the pinned
    ``prompt_text`` (prompt prefix);
  - ``mlflow.llm.model`` / ``temperature`` in the model call equal the
    pinned values;
  - every ``allowed_tools`` entry is offered to the model, and no other
    *registry* tool leaks through the gate (the deepagents built-ins —
    ``ls``, ``read_file``, ``write_file``, ... — are outside the
    registry gate and are reported as a warning, not a failure);
  - every selected skill name appears in the system prompt;
  - turn 2's system prompt carries the session-history header with the
    agent's own limits (``limited to N turn(s) / M tokens for agent X``)
    and exactly one ``<data source="session_history">`` block.

Usage (from the repo root, with API + worker + MLflow running):

    python scripts/manual/agent_config_check.py
    python scripts/manual/agent_config_check.py --cleanup
    python scripts/manual/agent_config_check.py --session-id sess-cfg-1
    python scripts/manual/agent_config_check.py --no-trace   # static + DB only

``--no-trace`` skips the MLflow assertions (for a stack without MLflow);
the static and DB checks still run. By default a missing trace is a
failure: the whole point is to assert the prompt/model/tools the model
actually saw.

Exit code: 0 when every check passes, 1 otherwise. Needs the project
venv (reads Postgres/registry/seed loader and the MLflow client).
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _api import (  # noqa: E402
    require_worker,
    submit,
    wait_for,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MISSION = "system_admin"

# One tool call keeps the small local model deterministic even with the
# ``stop_completion`` hook forcing a tool turn. The second request is the
# same so the only difference in its trace is the session-history block.
DEFAULT_REQUEST = "Hãy gọi system_health một lần và tóm tắt kết quả trong một câu."
DEFAULT_TURN2_REQUEST = "Hãy gọi system_health một lần nữa và xác nhận ngắn gọn."


class Report:
    """Counts PASS/FAIL/SKIP/WARN and prints one line per check."""

    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []
        self.skipped = 0
        self.warnings: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  [PASS] {label}" + (f" — {detail}" if detail else ""))
        else:
            self.failures.append(label)
            print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))
        return ok

    def skip(self, label: str, detail: str = "") -> None:
        self.skipped += 1
        print(f"  [SKIP] {label}" + (f" — {detail}" if detail else ""))

    def warn(self, label: str, detail: str = "") -> None:
        self.warnings.append(label)
        print(f"  [WARN] {label}" + (f" — {detail}" if detail else ""))


# --- Static config -----------------------------------------------------------


def _agent_dict(agent) -> dict:
    return {
        "id": agent.id,
        "model": agent.model,
        "system_prompt": agent.system_prompt,
        "temperature": agent.temperature,
        "allowed_tools": list(agent.allowed_tools or []),
        "session_history": agent.session_history,
    }


def _version_dict(version) -> dict:
    return {
        "version": version.version,
        "prompt_text": version.prompt_text,
        "model": version.model,
        "temperature": version.temperature,
        "allowed_tools": list(version.allowed_tools or []),
        "session_history": version.session_history,
        "manifest_sha256": version.manifest_sha256,
        "is_active": version.is_active,
    }


def load_config(mission_id: str) -> dict | None:
    """Mission + live agent row + active version + live tool digest."""
    from app import models
    from app.core.db import SessionLocal
    from app.services import agent_version_service, mission_service
    from runtime.tools.registry import registry

    db = SessionLocal()
    try:
        mission = db.get(models.Mission, mission_id)
        if mission is None:
            return None
        mission_dict = mission_service.mission_to_dict(mission)
        agent_id = mission_dict.get("agent_id") or ""
        agent = db.get(models.Agent, agent_id) if agent_id else None
        version = (
            agent_version_service.get_active_version(db, agent_id)
            if agent_id
            else None
        )
        # Importing the built-in tool modules registers them; do it
        # before listing so the registry names are complete.
        _, live_digest = agent_version_service.compute_tool_manifest()
        return {
            "mission": mission_dict,
            "agent_id": agent_id,
            "agent": _agent_dict(agent) if agent is not None else None,
            "version": _version_dict(version) if version is not None else None,
            "live_manifest_sha256": live_digest,
            "registry_tools": [entry.name for entry in registry.list()],
        }
    finally:
        db.close()


def load_seed(agent_id: str) -> dict | None:
    """The agent's YAML as the seeder would load it (env placeholders expanded)."""
    path = REPO_ROOT / "seeds" / "agents" / f"{agent_id}.yaml"
    if not path.is_file():
        return None
    from seeds.load_missions import load_agent_file

    return load_agent_file(path)


def _same_float(left, right) -> bool:
    try:
        return abs(float(left) - float(right)) < 1e-9
    except (TypeError, ValueError):
        return left == right


def _field_matches(source: dict, target: dict, field: str,
                   target_field: str | None = None) -> bool:
    """Compare one config field, using float tolerance for temperature."""
    other = target_field or field
    if field == "temperature":
        return _same_float(source[field], target[other])
    return source[field] == target[other]


def check_static(report: Report, config: dict) -> None:
    print("\n--- static config (seeds → agents row → AgentVersion) ---")
    agent = config["agent"]
    version = config["version"]
    agent_id = config["agent_id"]

    if not report.check(agent is not None, f"agent row exists ({agent_id or 'none'})"):
        return
    if not report.check(version is not None, "active AgentVersion exists"):
        return

    print(f"  active v{version['version']}: model={version['model']} "
          f"temperature={version['temperature']} "
          f"session_history={version['session_history']}")

    seed = load_seed(agent_id)
    if report.check(seed is not None, f"seed YAML exists (seeds/agents/{agent_id}.yaml)"):
        mismatches = [
            field for field, seed_field in (
                ("model", "model"),
                ("system_prompt", "system_prompt"),
                ("temperature", "temperature"),
                ("allowed_tools", "allowed_tools"),
                ("session_history", "session_history"),
            )
            if not _field_matches(seed, agent, seed_field, field)
        ]
        report.check(
            not mismatches,
            "seed YAML matches the live agents row",
            "run `make seed` (re-seed the YAML)" if mismatches
            else "prompt/model/temperature/tools/session limits",
        )
        if mismatches:
            print(f"    mismatched field(s): {', '.join(mismatches)}")

        seed_mismatches = [
            key for key, seed_field in (
                ("prompt_text", "system_prompt"),
                ("model", "model"),
                ("temperature", "temperature"),
                ("allowed_tools", "allowed_tools"),
                ("session_history", "session_history"),
            )
            if not _field_matches(seed, version, seed_field, key)
        ]
        report.check(
            not seed_mismatches,
            "seed YAML matches the active version snapshot",
            "drift: " + ", ".join(seed_mismatches) if seed_mismatches
            else f"v{version['version']}",
        )

    row_mismatches = [
        key for key, agent_field in (
            ("prompt_text", "system_prompt"),
            ("model", "model"),
            ("temperature", "temperature"),
            ("allowed_tools", "allowed_tools"),
            ("session_history", "session_history"),
        )
        if not _field_matches(agent, version, agent_field, key)
    ]
    report.check(
        not row_mismatches,
        "live agents row matches the active version snapshot",
        "drift: " + ", ".join(row_mismatches) if row_mismatches
        else f"v{version['version']}",
    )

    report.check(
        config["live_manifest_sha256"] == version["manifest_sha256"],
        "tool manifest has not drifted from the pinned version",
        f"{config['live_manifest_sha256'][:16]}… (drift=False)"
        if config["live_manifest_sha256"] == version["manifest_sha256"]
        else f"live {config['live_manifest_sha256'][:16]}… != "
             f"pinned {version['manifest_sha256'][:16]}…",
    )

    registry_tools = set(config["registry_tools"])
    allowed = version["allowed_tools"]
    missing = [name for name in allowed if name not in registry_tools]
    report.check(
        not missing,
        f"all {len(allowed)} allowed_tools are registered",
        "unknown: " + ", ".join(missing) if missing else "",
    )
    report.check(
        bool(version["model"]),
        "version pins a model",
        version["model"],
    )
    report.check(
        0.0 <= float(version["temperature"]) <= 2.0,
        "version pins a valid temperature",
        str(version["temperature"]),
    )

    # Session-history limits: the renderer must apply the version's caps.
    from runtime.session_history import (
        DEFAULT_LIMITS,
        parse_session_history_config,
        resolve_session_limits,
    )

    raw_limits = version["session_history"]
    try:
        parsed = parse_session_history_config(
            raw_limits, source=f"agent {agent_id} v{version['version']}"
        )
        valid = True
    except ValueError as exc:
        parsed, valid = None, False
        print(f"    invalid session_history: {exc}")
    report.check(valid, "session_history block is valid", json.dumps(raw_limits))
    limits = resolve_session_limits(version_limits=raw_limits)
    expected_turns = (parsed or {}).get("max_turns", DEFAULT_LIMITS.max_turns)
    expected_tokens = (parsed or {}).get("max_tokens", DEFAULT_LIMITS.max_tokens)
    report.check(
        limits.max_turns == expected_turns and limits.max_tokens == expected_tokens,
        "session limits resolve from the pinned version",
        f"{limits.max_turns} turn(s) / {limits.max_tokens:,} tokens",
    )

    # Skills: declared ones must exist and selection must return exactly them.
    from runtime.skills import select_skills

    execution = config["mission"].get("execution") or {}
    declared_skills = [str(name) for name in execution.get("skills") or []]
    missing_files = [
        name for name in declared_skills
        if not (REPO_ROOT / "skills" / name / "SKILL.md").is_file()
    ]
    report.check(
        not missing_files,
        f"all {len(declared_skills)} mission skill files exist",
        "missing: " + ", ".join(missing_files) if missing_files else "",
    )
    selected = [skill.name for skill in select_skills(config["mission"])]
    report.check(
        selected == declared_skills,
        "select_skills returns exactly the mission-declared skills",
        ", ".join(selected) or "(none)",
    )

    # Hooks: declared ones must resolve to built-in handlers.
    from runtime.hooks.registry import select_hooks

    declared_hooks = [str(name) for name in execution.get("hooks") or []]
    resolved_hooks = select_hooks(config["mission"])
    report.check(
        resolved_hooks == declared_hooks,
        f"all {len(declared_hooks)} mission hooks resolve",
        ", ".join(resolved_hooks) or "(none)",
    )


# --- Live run + trace --------------------------------------------------------


def run_turn(session_id: str, request: str, timeout: float) -> tuple[dict, float]:
    """Submit one turn in the session and wait for its terminal state."""
    execution = submit(DEFAULT_MISSION, {"request": request}, session_id=session_id)
    return wait_for(execution["execution_id"], timeout=timeout)


def usage_models(execution_id: str) -> list[str]:
    from app.core.db import SessionLocal
    from runtime.middleware.queries import get_usage_for_execution

    db = SessionLocal()
    try:
        return [row.model for row in get_usage_for_execution(db, execution_id)]
    finally:
        db.close()


class MlxTraceReader:
    """Find one execution's MLflow trace via its tagged execution run."""

    def __init__(self) -> None:
        from app.config import settings

        self.error: str | None = None
        self.client = None
        self.experiment_id: str | None = None
        if not settings.mlflow_tracing_enabled:
            self.error = "MLFLOW_TRACING_ENABLED is off"
            return
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            self.client = MlflowClient()
            experiment = self.client.get_experiment_by_name(
                settings.mlflow_experiment
            )
            if experiment is None:
                self.error = (
                    f"experiment {settings.mlflow_experiment!r} not found at "
                    f"{settings.mlflow_tracking_uri}"
                )
                self.client = None
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as exc:  # noqa: BLE001 — tracing is optional here
            self.error = f"{type(exc).__name__}: {exc}"

    def find(self, execution_id: str, timeout: float
             ) -> tuple[object | None, object | None]:
        """Return (trace, run) for the execution, polling until timeout."""
        if self.client is None:
            return None, None
        deadline = time.monotonic() + timeout
        while True:
            try:
                runs = self.client.search_runs(
                    [self.experiment_id],
                    filter_string=f"tags.execution_id = '{execution_id}'",
                    max_results=5,
                )
                for run in runs:
                    # run_id scopes the trace search to this execution's
                    # run; the autolog trace links to it via
                    # metadata["mlflow.sourceRun"].
                    traces = self.client.search_traces(
                        locations=[run.info.experiment_id],
                        run_id=run.info.run_id,
                        max_results=5,
                    )
                    if traces:
                        return traces[0], run
            except Exception as exc:  # noqa: BLE001 — retry while server catches up
                print(f"    (trace lookup: {type(exc).__name__}: {exc})")
            if time.monotonic() >= deadline:
                return None, None
            time.sleep(3)


def chat_call(trace) -> dict | None:
    """The first CHAT_MODEL span as {system, model, temperature, tools}."""
    for span in trace.data.spans:
        if span.span_type != "CHAT_MODEL":
            continue
        attributes = span.attributes or {}
        messages = (attributes.get("mlflow.spanInputs") or {}).get("messages") or []
        system = ""
        if messages:
            content = messages[0].get("content")
            if isinstance(content, str):
                system = content
            elif isinstance(content, list):
                system = "".join(
                    part.get("text", "")
                    for part in content if isinstance(part, dict)
                )
        invocation = attributes.get("invocation_params") or {}
        tools = []
        for tool in attributes.get("mlflow.chat.tools") or []:
            name = (tool.get("function") or {}).get("name")
            if name:
                tools.append(name)
        return {
            "system": system,
            "model": attributes.get("mlflow.llm.model") or invocation.get("model"),
            "temperature": invocation.get("temperature"),
            "tools": tools,
        }
    return None


def check_turn(
    report: Report,
    execution: dict,
    label: str,
    version: dict,
    config: dict,
    reader: MlxTraceReader | None,
    trace_timeout: float,
) -> dict | None:
    """Assert one execution's identity/config; return its chat call if any."""
    print(f"\n--- live run: {label} ---")
    execution_id = execution["execution_id"]
    report.check(
        execution["status"] == "SUCCEEDED",
        f"{label}: execution SUCCEEDED",
        f"{execution_id} ({execution.get('error') or 'no error'})",
    )
    report.check(
        execution.get("agent_version") == version["version"],
        f"{label}: ran AgentVersion v{version['version']}",
        f"recorded v{execution.get('agent_version')}",
    )
    report.check(
        not execution.get("agent_version_drift"),
        f"{label}: no version drift recorded",
        f"drift={execution.get('agent_version_drift')}",
    )

    models = usage_models(execution_id)
    report.check(
        bool(models) and all(model == version["model"] for model in models),
        f"{label}: llm_usage records the pinned model",
        f"{models} vs {version['model']}",
    )

    if reader is None:
        report.skip(f"{label}: prompt/model/temperature/tools from the trace")
        return None

    trace, run = reader.find(execution_id, trace_timeout)
    if not report.check(
        trace is not None,
        f"{label}: MLflow trace found",
        f"run={getattr(run.info, 'run_id', '?')[:12]}…" if run else
        reader.error or "no trace within timeout (is `MLFLOW_TRACING_ENABLED` on?)",
    ):
        return None

    call_info = chat_call(trace)
    if not report.check(call_info is not None, f"{label}: trace has a CHAT_MODEL span"):
        return None

    report.check(
        call_info["system"].startswith(version["prompt_text"]),
        f"{label}: system prompt starts with the pinned v{version['version']} prompt",
        f"pinned {len(version['prompt_text'])} chars, sent {len(call_info['system'])} chars",
    )
    report.check(
        call_info["model"] == version["model"],
        f"{label}: model call uses the pinned model",
        f"{call_info['model']} vs {version['model']}",
    )
    report.check(
        _same_float(call_info["temperature"], version["temperature"]),
        f"{label}: model call uses the pinned temperature",
        f"{call_info['temperature']} vs {version['temperature']}",
    )

    allowed = version["allowed_tools"]
    missing_tools = [name for name in allowed if name not in call_info["tools"]]
    report.check(
        not missing_tools,
        f"{label}: all {len(allowed)} allowed_tools are offered to the model",
        "missing: " + ", ".join(missing_tools) if missing_tools
        else f"{len(call_info['tools'])} tools at the call",
    )

    # The gate must not leak *registry* tools outside allowed_tools. The
    # deepagents built-ins (ls/read_file/...) are not in the registry and
    # are expected; report them, do not fail on them.
    registry_tools = set(config["registry_tools"])
    leaked = sorted((set(call_info["tools"]) & registry_tools) - set(allowed))
    report.check(
        not leaked,
        f"{label}: no registry tool outside allowed_tools reached the model",
        ", ".join(leaked) if leaked else "",
    )
    builtins = sorted(set(call_info["tools"]) - set(allowed) - registry_tools)
    if builtins:
        report.warn(
            f"{label}: {len(builtins)} deepagents built-in tool(s) outside allowed_tools",
            ", ".join(builtins),
        )

    declared_skills = [
        str(name) for name in
        (config["mission"].get("execution") or {}).get("skills") or []
    ]
    missing_skills = [
        name for name in declared_skills if name not in call_info["system"]
    ]
    report.check(
        not missing_skills,
        f"{label}: all {len(declared_skills)} selected skills appear in the system prompt",
        "missing: " + ", ".join(missing_skills) if missing_skills else "",
    )
    return call_info


def check_session_history(report: Report, call_info: dict | None, session_id: str,
                          agent_id: str, limits, prior_turns: int) -> None:
    print("\n--- session history (turn 2 reads turn 1) ---")
    if call_info is None:
        report.skip("session-history header from the trace")
        return
    system = call_info["system"]
    header = (
        f"Session history: {prior_turns} prior turn(s) in session {session_id}"
        f" (oldest first; limited to {limits.max_turns} turn(s) / "
        f"{limits.max_tokens:,} tokens for agent {agent_id})."
    )
    report.check(
        header in system,
        f"session history header names the agent's limits "
        f"({limits.max_turns} turn(s) / {limits.max_tokens:,} tokens)",
        header,
    )
    blocks = system.count('<data source="session_history"')
    report.check(
        blocks == prior_turns,
        f"{prior_turns} prior turn(s) rendered as data block(s)",
        f"{blocks} block(s)",
    )


# --- Cleanup -----------------------------------------------------------------


def cleanup(session_id: str) -> None:
    """Delete the session's executions and their result/usage/audit rows."""
    from app import models
    from app.core.db import SessionLocal

    db = SessionLocal()
    try:
        execution_ids = [
            row.id for row in
            db.query(models.Execution).filter_by(session_id=session_id)
        ]
        if execution_ids:
            db.query(models.ExecutionResult).filter(
                models.ExecutionResult.execution_id.in_(execution_ids)
            ).delete(synchronize_session=False)
            db.query(models.LLMUsage).filter(
                models.LLMUsage.execution_id.in_(execution_ids)
            ).delete(synchronize_session=False)
            db.query(models.AuditLog).filter(
                models.AuditLog.execution_id.in_(execution_ids)
            ).delete(synchronize_session=False)
            db.query(models.Execution).filter(
                models.Execution.id.in_(execution_ids)
            ).delete(synchronize_session=False)
        db.commit()
        print(f"cleanup: session {session_id}, {len(execution_ids)} execution(s) "
              f"and their result/usage/audit rows")
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--session-id", default=None,
                        help="reuse a session id (default: sess-cfg-<8 hex>)")
    parser.add_argument("--request", default=DEFAULT_REQUEST,
                        help="turn 1 request sent to system_admin")
    parser.add_argument("--turn2-request", default=DEFAULT_TURN2_REQUEST,
                        help="turn 2 request (same session, sees turn 1)")
    parser.add_argument("--no-trace", action="store_true",
                        help="skip the MLflow trace assertions (static + DB only)")
    parser.add_argument("--trace-wait", type=float, default=60,
                        help="seconds to wait for the MLflow trace (default 60)")
    parser.add_argument("--timeout", type=float, default=300,
                        help="seconds to wait per turn (default 300)")
    parser.add_argument("--cleanup", action="store_true",
                        help="delete this session's execution rows afterwards")
    args = parser.parse_args()

    require_worker()
    config = load_config(DEFAULT_MISSION)
    if config is None:
        sys.exit(f"mission {DEFAULT_MISSION!r} not found — run `make seed`")
    if not config["agent_id"] or config["version"] is None:
        sys.exit(
            f"mission {DEFAULT_MISSION!r} has no agent/active version — "
            f"run `make seed`"
        )

    agent_id = config["agent_id"]
    version = config["version"]
    from runtime.session_history import resolve_session_limits

    limits = resolve_session_limits(version_limits=version["session_history"])
    session_id = args.session_id or f"sess-cfg-{secrets.token_hex(4)}"

    print(f"=== agent config check: mission={DEFAULT_MISSION} agent={agent_id}")
    print(f"  session_id={session_id}  active_version=v{version['version']}")

    report = Report()
    check_static(report, config)

    reader = None
    if args.no_trace:
        print("\n--- live run: trace checks disabled (--no-trace) ---")
    else:
        reader = MlxTraceReader()
        if reader.client is None:
            report.check(False, "MLflow trace reader is available",
                         reader.error or "unknown error")
            reader = None

    executions: list[dict] = []
    calls: list[dict | None] = []
    try:
        for label, request in (
            ("turn 1", args.request),
            ("turn 2", args.turn2_request),
        ):
            execution, elapsed = run_turn(session_id, request, args.timeout)
            executions.append(execution)
            print(f"\n[{label}] {execution['execution_id']} "
                  f"{execution['status']} in {elapsed:.1f}s "
                  f"session={execution.get('session_id')}")
            print(f"  request: {request}")
            call_info = check_turn(
                report, execution, label, version, config, reader, args.trace_wait
            )
            calls.append(call_info)

        check_session_history(
            report, calls[1], session_id, agent_id, limits, prior_turns=1
        )
    finally:
        if args.cleanup:
            cleanup(session_id)

    if report.failures:
        print("\n  checks: FAIL")
        for failure in report.failures:
            print(f"    - {failure}")
    else:
        print(f"\n  checks: PASS — {report.passed} passed, "
              f"{report.skipped} skipped, {len(report.warnings)} warning(s)")
    print(f"\n{'PASS' if not report.failures else 'FAIL'}: agent config")
    return 0 if not report.failures else 1


if __name__ == "__main__":
    sys.exit(main())
