"""Admin tools (Layer 5) — administer the agentic-runtime platform itself.

Read tools inspect the platform (executions, missions, agents, dependency
health). Write tools manage it (create/update/delete missions, dispatch
and cancel executions). Every tool here is registered with
``capability="system_admin"`` and ``permission="admin"``: the registry
permission check (``runtime/tools/registry.py``) only hands them to a
mission whose capabilities include ``system_admin`` — the ``fault_analysis``
mission never sees them.

Implementations delegate to ``app.services`` (never raw SQL, never HTTP
calls to this same API). Write tools never raise: failures come back as
structured ``{"error": ...}`` / ``{"refused": ...}`` results. Destructive
tools (``delete_mission``, ``cancel_execution``) act only on the exact id
given in the input.
"""

from __future__ import annotations

import logging
from datetime import datetime

from langchain_core.tools import tool

from app.core.db import SessionLocal
from app.services import execution_service, mission_service
from runtime.tools.registry import registry

logger = logging.getLogger(__name__)


# --- Serializers (rows -> JSON-safe dicts) ---
def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _execution_to_dict(execution) -> dict:
    return {
        "execution_id": execution.id,
        "mission_id": execution.mission_id,
        "mission_version": execution.mission_version,
        "status": execution.status,
        "error": execution.error,
        "created_at": _iso(execution.created_at),
        "started_at": _iso(execution.started_at),
        "finished_at": _iso(execution.finished_at),
    }


def _mission_to_dict(mission) -> dict:
    return {
        "mission_id": mission.id,
        "name": mission.name,
        "version": mission.version,
        "objective": mission.objective,
        "input_schema": mission.input_schema or {},
        "output_schema": mission.output_schema or {},
        "execution": mission.execution_config or {},
        "capabilities": mission.capabilities or [],
        "agent_id": mission.agent_id,
    }


def _agent_to_dict(agent) -> dict:
    return {
        "agent_id": agent.id,
        "name": agent.name,
        "provider": agent.provider,
        "model": agent.model,
        "temperature": agent.temperature,
        "allowed_tools": agent.allowed_tools or [],
    }


# --- Read tools ---
@tool
def list_executions(status: str | None = None, mission_id: str | None = None,
                    limit: int = 20) -> dict:
    """List executions, newest first. Optional filters: status, mission_id."""
    from app import models

    limit = max(1, min(int(limit), 100))
    db = SessionLocal()
    try:
        query = db.query(models.Execution).order_by(models.Execution.created_at.desc())
        if status:
            query = query.filter(models.Execution.status == status)
        if mission_id:
            query = query.filter(models.Execution.mission_id == mission_id)
        rows = query.limit(limit).all()
        return {"executions": [_execution_to_dict(r) for r in rows], "count": len(rows)}
    finally:
        db.close()


@tool
def get_execution(execution_id: str) -> dict:
    """Get one execution with its result and error (if any)."""
    db = SessionLocal()
    try:
        execution = execution_service.get_execution(db, execution_id)
        if execution is None:
            return {"error": "Execution not found", "execution_id": execution_id}
        result = execution_service.get_execution_result(db, execution_id)
        out = _execution_to_dict(execution)
        if result is not None:
            out["result"] = result.output or {}
            out["evidence_count"] = len(result.evidence or [])
            out["evidence"] = result.evidence or []
        else:
            out["result"] = None
            out["evidence_count"] = 0
            out["evidence"] = []
        return out
    finally:
        db.close()


@tool
def list_missions() -> dict:
    """List all missions (id, name, version, objective, agent, capabilities)."""
    db = SessionLocal()
    try:
        rows = mission_service.list_missions(db)
        return {"missions": [_mission_to_dict(r) for r in rows], "count": len(rows)}
    finally:
        db.close()


@tool
def get_mission(mission_id: str) -> dict:
    """Get one mission's full config."""
    db = SessionLocal()
    try:
        mission = mission_service.get_mission(db, mission_id)
        if mission is None:
            return {"error": "Mission not found", "mission_id": mission_id}
        return _mission_to_dict(mission)
    finally:
        db.close()


@tool
def list_agents() -> dict:
    """List all agents (id, name, provider, model, allowed tools)."""
    from app import models

    db = SessionLocal()
    try:
        rows = db.query(models.Agent).order_by(models.Agent.id).all()
        return {"agents": [_agent_to_dict(r) for r in rows], "count": len(rows)}
    finally:
        db.close()


def _check_database() -> dict:
    from sqlalchemy import text

    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "detail": "SELECT 1 succeeded"}
    except Exception as exc:  # noqa: BLE001 — health is best-effort
        return {"status": "fail", "detail": str(exc)[:200]}
    finally:
        db.close()


def _check_redis() -> dict:
    """Ping the Celery broker when it is a real Redis URL."""
    from urllib.parse import urlparse

    from app.config import settings

    url = settings.celery_broker_url or ""
    scheme = urlparse(url).scheme
    if scheme not in ("redis", "rediss"):
        return {"status": "skipped", "detail": f"broker {scheme or 'unset'} (not a live Redis)"}
    try:
        import redis

        client = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
        client.ping()
        return {"status": "ok", "detail": "PING succeeded"}
    except Exception as exc:  # noqa: BLE001 — health is best-effort
        return {"status": "fail", "detail": str(exc)[:200]}


def _check_http(name: str, url: str) -> dict:
    """GET a health endpoint with a short timeout. Any HTTP response means up."""
    import urllib.error
    import urllib.request

    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=2) as response:
            return {"status": "ok", "detail": f"HTTP {response.status}"}
    except urllib.error.HTTPError as exc:
        # The server answered (e.g. 404 on a root path) — it is up.
        return {"status": "ok", "detail": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001 — health is best-effort
        return {"status": "fail", "detail": f"{name} unreachable: {exc}"[:200]}


@tool
def system_health() -> dict:
    """Check the health of each platform dependency (DB, Redis, Qdrant,
    MLflow, MinIO). Best-effort per dependency: each is reported ok/fail/
    skipped with its error; this tool never raises."""
    from app.config import settings

    checks = {"database": _check_database(), "redis": _check_redis()}
    checks["qdrant"] = _check_http("Qdrant", f"{settings.qdrant_url.rstrip('/')}/healthz")
    mlflow_uri = settings.mlflow_tracking_uri or ""
    if mlflow_uri.startswith("http"):
        checks["mlflow"] = _check_http("MLflow", mlflow_uri.rstrip("/"))
    else:
        checks["mlflow"] = {
            "status": "skipped",
            "detail": f"tracking URI {mlflow_uri or 'unset'} (not an HTTP server)",
        }
    scheme = "https" if settings.minio_secure else "http"
    checks["minio"] = _check_http(
        "MinIO", f"{scheme}://{settings.minio_endpoint}/minio/health/live"
    )
    overall = "ok" if all(c["status"] in ("ok", "skipped") for c in checks.values()) else "degraded"
    return {"overall": overall, "checks": checks}


# --- Write tools ---
@tool
def create_mission(mission_id: str, name: str, objective: str = "",
                   input_schema: dict | None = None, output_schema: dict | None = None,
                   execution_mode: str = "hybrid", agent_id: str | None = None,
                   capabilities: list | None = None) -> dict:
    """Create a new mission. Fails (structured error, no raise) when a
    mission with the same id already exists — use update_mission then."""
    from sqlalchemy.exc import IntegrityError

    data = {
        "id": mission_id,
        "name": name,
        "version": 1,
        "objective": objective,
        "input_schema": input_schema or {},
        "output_schema": output_schema or {},
        "execution_config": {"mode": execution_mode, "agent_id": agent_id},
        "capabilities": capabilities or [],
        "agent_id": agent_id,
    }
    db = SessionLocal()
    try:
        try:
            mission = mission_service.create_mission(db, data)
        except IntegrityError:
            db.rollback()
            return {
                "error": f"Mission {mission_id} already exists",
                "mission_id": mission_id,
            }
        return {"created": True, **_mission_to_dict(mission)}
    finally:
        db.close()


@tool
def update_mission(mission_id: str, name: str | None = None, objective: str | None = None,
                   input_schema: dict | None = None, output_schema: dict | None = None,
                   execution_mode: str | None = None, agent_id: str | None = None,
                   capabilities: list | None = None) -> dict:
    """Update a mission (upsert: creates it when the id does not exist yet).
    Only the fields given are changed; the rest keep their current values."""
    db = SessionLocal()
    try:
        existing = mission_service.get_mission(db, mission_id)
        if existing is None:
            data = {
                "id": mission_id,
                "name": name or mission_id,
                "version": 1,
                "objective": objective or "",
                "input_schema": input_schema or {},
                "output_schema": output_schema or {},
                "execution_config": {"mode": execution_mode or "hybrid", "agent_id": agent_id},
                "capabilities": capabilities or [],
                "agent_id": agent_id,
            }
        else:
            current = _mission_to_dict(existing)
            execution_config = dict(current["execution"])
            if execution_mode is not None:
                execution_config["mode"] = execution_mode
            if agent_id is not None:
                execution_config["agent_id"] = agent_id

            def _pick(new, old):
                return old if new is None else new

            data = {
                "id": mission_id,
                "name": _pick(name, current["name"]),
                "version": current["version"],
                "objective": _pick(objective, current["objective"]),
                "input_schema": _pick(input_schema, current["input_schema"]),
                "output_schema": _pick(output_schema, current["output_schema"]),
                "execution_config": execution_config,
                "capabilities": _pick(capabilities, current["capabilities"]),
                "agent_id": _pick(agent_id, current["agent_id"]),
            }
        mission = mission_service.upsert_mission(db, data)
        return {"updated": True, **_mission_to_dict(mission)}
    finally:
        db.close()


@tool
def delete_mission(mission_id: str) -> dict:
    """Delete a mission. Refused (structured result, no cascade) when the
    mission still has executions — delete or archive them first."""
    db = SessionLocal()
    try:
        try:
            deleted = mission_service.delete_mission(db, mission_id)
        except mission_service.MissionHasExecutionsError as exc:
            return {
                "deleted": False,
                "refused": True,
                "reason": f"Mission has {exc.execution_count} execution(s)",
                "execution_count": exc.execution_count,
                "mission_id": mission_id,
            }
        if not deleted:
            return {"deleted": False, "error": "Mission not found", "mission_id": mission_id}
        return {"deleted": True, "mission_id": mission_id}
    finally:
        db.close()


@tool
def dispatch_execution(mission_id: str, input_data: dict | None = None) -> dict:
    """Create and dispatch a run of a mission. Returns the execution id
    (status QUEUED); the worker runs it asynchronously."""
    db = SessionLocal()
    try:
        try:
            execution, missing = execution_service.create_execution(
                db, mission_id, input_data or {}
            )
        except Exception as exc:  # noqa: BLE001 — dispatch failure is a result
            return {"error": f"dispatch failed: {exc}", "mission_id": mission_id}
        if execution is None:
            if missing == ["__mission_not_found__"]:
                return {"error": "Mission not found", "mission_id": mission_id}
            return {
                "error": f"Missing required input fields: {missing}",
                "mission_id": mission_id,
                "missing_fields": missing,
            }
        return {
            "execution_id": execution.id,
            "mission_id": execution.mission_id,
            "status": execution.status,
        }
    finally:
        db.close()


@tool
def cancel_execution(execution_id: str) -> dict:
    """Cancel a QUEUED or RUNNING execution. The worker checks the flag
    before starting and after finishing, so a cancelled run never
    overwrites its result. Already-finished executions keep their status."""
    db = SessionLocal()
    try:
        execution = execution_service.cancel_execution(db, execution_id)
        if execution is None:
            return {"error": "Execution not found", "execution_id": execution_id}
        return {
            "execution_id": execution.id,
            "status": execution.status,
            "cancelled": execution.status == "CANCELLED",
        }
    finally:
        db.close()


# Register all eleven admin tools with their business capability + permission.
# NOTE: create_agent is deliberately NOT here — no agent service exists
# (agents are created directly by the API router), so there is no safe
# service to delegate to.
registry.register(list_executions, capability="system_admin",
                  permission="admin", timeout_seconds=30)
registry.register(get_execution, capability="system_admin",
                  permission="admin", timeout_seconds=30)
registry.register(list_missions, capability="system_admin",
                  permission="admin", timeout_seconds=30)
registry.register(get_mission, capability="system_admin",
                  permission="admin", timeout_seconds=30)
registry.register(list_agents, capability="system_admin",
                  permission="admin", timeout_seconds=30)
registry.register(system_health, capability="system_admin",
                  permission="admin", timeout_seconds=30)
registry.register(create_mission, capability="system_admin",
                  permission="admin", timeout_seconds=30)
registry.register(update_mission, capability="system_admin",
                  permission="admin", timeout_seconds=30)
registry.register(delete_mission, capability="system_admin",
                  permission="admin", timeout_seconds=30)
registry.register(dispatch_execution, capability="system_admin",
                  permission="admin", timeout_seconds=60)
registry.register(cancel_execution, capability="system_admin",
                  permission="admin", timeout_seconds=30)
