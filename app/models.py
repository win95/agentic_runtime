"""SQLAlchemy models — the business state of the platform.

PostgreSQL is the source of truth for Mission / Agent / Tool / Policy /
Execution / Result. LangSmith is observability only and never stores this
state. Column types are deliberately portable (String/JSON/Text) so the
same models run on SQLite in tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    true,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Mission(Base):
    """A Mission is CONFIG (business definition), not a LangGraph."""

    __tablename__ = "missions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    objective: Mapped[str] = mapped_column(Text, nullable=False, default="")
    input_schema: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    output_schema: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    execution_config: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict
    )  # e.g. {"mode": "hybrid", "agent_id": "fault_investigator"}
    capabilities: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    agent_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class Agent(Base):
    """Agent configuration: model, prompt, allowed tools."""

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="openai")
    model: Mapped[str] = mapped_column(String(128), nullable=False, default="gpt-4o-mini")
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    temperature: Mapped[float] = mapped_column(default=0.0)
    allowed_tools: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # Per-agent session-history limits (docs/sessions.md): {"max_turns",
    # "max_tokens"} or NULL (use the global caps). Seeded from the
    # agent YAML's session_history block; pinned into the AgentVersion
    # snapshot, so a limit change creates a new version on re-seed.
    session_history: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class AgentVersion(Base):
    """One immutable snapshot of an agent's full config (Option C).

    A version pins everything that decides the agent's behaviour: the
    exact system prompt text, the MLflow prompt version it was built
    from (best-effort, may be NULL), provider/model/temperature, the
    allowed tool list, and the tool manifest (name, description, schema
    and permission of every tool the agent could call) with its SHA-256.
    ``git_sha`` records the code the snapshot was taken from (NULL when
    taken outside a git checkout). ``session_history`` pins the agent's
    per-agent history-reading limits (NULL = the global caps at the
    time) — activating an older version restores its old limits.

    Only one version per agent is active (``is_active``); the runtime
    uses the active version, and rollback is activating an older one.
    Rows are never edited in place — a change creates a new version.
    """

    __tablename__ = "agent_versions"

    # One row per (agent, version): without this, two snapshots could
    # claim the same version number and rollback/activation would be
    # ambiguous. The service creates versions as max+1, so duplicates
    # should not exist; the constraint is the backstop (migration 0002).
    __table_args__ = (UniqueConstraint("agent_id", "version"),)

    id: Mapped[str] = mapped_column(String(256), primary_key=True)
    agent_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("agents.id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    prompt_registry_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="openai")
    model: Mapped[str] = mapped_column(String(128), nullable=False, default="gpt-4o-mini")
    temperature: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    allowed_tools: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    tool_manifest: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    git_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Pinned per-agent session-history limits (see Agent.session_history).
    session_history: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="seed")
    is_active: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Tool(Base):
    """Registered tool metadata (the runtime registry mirrors this)."""

    __tablename__ = "tools"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    capability: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    input_schema: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    output_schema: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    permission: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    implementation: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # Layer 4: False = the tool writes, deletes or dispatches. Stored per
    # row (mirrored from the registry by seed_tools_demo) so GET /tools/db
    # reports the real value instead of the schema default (True) for
    # every tool — before this column existed, all 17 tools, including
    # the admin write tools, showed read_only=true.
    read_only: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=true()
    )
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Policy(Base):
    """Access policy: which tenant may use which capabilities/tools."""

    __tablename__ = "policies"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    tenant: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    allowed_capabilities: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    allowed_tools: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    max_executions_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Execution(Base):
    """One run of a Mission. Each execution has its own state; executions
    are independent by default (no conversation memory across
    executions) — unless they share a ``session_id`` (see
    docs/sessions.md): executions in one session form a conversation,
    and the Deep Agent of a later run is shown the prior turns."""

    __tablename__ = "executions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    mission_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("missions.id"), nullable=False
    )
    # Option B session (nullable: runs created before sessions, and
    # runs whose client omitted the id — the service auto-generates
    # one — both carry a value; NULL only predates the feature).
    # A session is bound to one agent + one mission: the first
    # execution in it defines both (enforced in execution_service).
    session_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )
    mission_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    agent_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Option C: which AgentVersion ran (NULL when the run predates
    # versioning or no version was active), and whether the live tool
    # manifest had drifted from that version's pinned manifest.
    agent_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    agent_version_drift: Mapped[bool] = mapped_column(default=False)
    input: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # QUEUED | RUNNING | SUCCEEDED | FAILED | CANCELLED
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="QUEUED")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ExecutionResult(Base):
    """Result of one execution (output + evidence)."""

    __tablename__ = "execution_results"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    execution_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("executions.id"), nullable=False, unique=True
    )
    output: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    evidence: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class FaultHistory(Base):
    """One recorded equipment fault. Production rows come from the real
    fault ingestion pipeline."""

    __tablename__ = "fault_history"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="medium")
    occurred_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    technician: Mapped[str | None] = mapped_column(String(128), nullable=True)
    action: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class MaintenanceRecord(Base):
    """One maintenance action on an asset."""

    __tablename__ = "maintenance_records"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    technician: Mapped[str | None] = mapped_column(String(128), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class SensorReading(Base):
    """One sensor reading for an asset (time-series, Postgres for now).

    A plain Postgres table is the pragmatic first store — the same
    pattern as fault_history. If volume grows, this is the table to move
    to TimescaleDB (hypertable on ``at``) without changing the readers:
    they only query (asset_id, metric, at).
    """

    __tablename__ = "sensor_readings"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    metric: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Schedule(Base):
    """A recurring mission run (cron).

    The dispatcher (app/services/schedule_service.py, driven every minute
    by the Celery beat task ``run_due_schedules``) creates one Execution
    per due schedule. Cron times are naive UTC, like every other datetime
    in this codebase. ``next_run_at`` is always set by the service (never
    left NULL by normal writes); a NULL there means the schedule is never
    due. ``last_error`` records the latest dispatch failure so a broken
    schedule is visible instead of silently skipped.
    """

    __tablename__ = "schedules"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    mission_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("missions.id"), nullable=False, index=True
    )
    cron: Mapped[str] = mapped_column(String(64), nullable=False)
    input_data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    enabled: Mapped[bool] = mapped_column(default=True)
    tenant: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class IngestRun(Base):
    """One batch-ingest run for a single source file (Option B).

    A run is per file, not per batch: ``source`` is the file name, so a
    re-run can skip files that already succeeded (watermark). ``status``
    is SUCCEEDED (all rows loaded), PARTIAL (some rows quarantined) or
    FAILED (the file itself could not be read/parsed). ``errors`` holds
    file-level errors; row-level failures live in ``ingest_quarantine``.
    """

    __tablename__ = "ingest_runs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="SUCCEEDED")
    rows_ok: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    errors: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class IngestQuarantine(Base):
    """One row that failed validation or loading during an ingest run.

    The raw row is kept as JSON so it can be fixed and re-run (see
    ``ingest.run --retry``) instead of being silently dropped. ``status``
    is pending, retried, or discarded.
    """

    __tablename__ = "ingest_quarantine"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("ingest_runs.id"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String(512), nullable=False)
    row: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class LLMUsage(Base):
    """One LLM call's token usage, as reported by the provider.

    Written by the middleware callback handler (runtime/middleware).
    Tokens come only from ``response.llm_output["token_usage"]``; when the
    provider reports no usage the counts stay NULL and ``usage_source``
    is "missing" — counts are never estimated or invented. No cost is
    recorded here by design (see README section 13).
    """

    __tablename__ = "llm_usage"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    execution_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    mission_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    tenant: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_source: Mapped[str] = mapped_column(String(32), nullable=False, default="provider")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class AuditLog(Base):
    """One tool call, for the audit trail (runtime/middleware).

    ``args_truncated`` is the tool input with secrets redacted and cut to
    2000 chars; ``status`` is "started" (call began, no end seen yet),
    "success" or "error". The handler writes rows best-effort and never
    raises, so a logging failure cannot fail an execution.
    """

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    execution_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    mission_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    tenant: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    args_truncated: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="started")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# Ontology canonical tables — side-effect import so Alembic sees metadata.
from ontology import models as _ontology_models  # noqa: E402, F401
