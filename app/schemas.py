"""Pydantic schemas for the Platform API (request/response shapes)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Mission ---
class MissionCreate(BaseModel):
    id: str
    name: str
    version: int = 1
    objective: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    execution_config: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    agent_id: str | None = None


class MissionRead(MissionCreate):
    created_at: datetime | None = None
    updated_at: datetime | None = None


# --- Agent ---
class AgentCreate(BaseModel):
    id: str
    name: str
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    system_prompt: str = ""
    temperature: float = 0.0
    allowed_tools: list[str] = Field(default_factory=list)
    # Per-agent session-history limits (docs/sessions.md): None means
    # the global caps. Seeded from the agent YAML, not via the API.
    session_history: dict[str, Any] | None = None


class AgentRead(AgentCreate):
    created_at: datetime | None = None


# --- Agent version (Option C) ---
class AgentVersionRead(BaseModel):
    id: str
    agent_id: str
    version: int
    prompt_text: str = ""
    prompt_registry_version: int | None = None
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    allowed_tools: list[str] = Field(default_factory=list)
    tool_manifest: list[dict[str, Any]] = Field(default_factory=list)
    manifest_sha256: str = ""
    git_sha: str | None = None
    # Pinned per-agent session-history limits (None = global caps).
    session_history: dict[str, Any] | None = None
    source: str = "seed"
    is_active: bool = False
    created_at: datetime | None = None


# --- Tool ---
class ToolRead(BaseModel):
    id: str
    name: str
    description: str = ""
    capability: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    permission: str = ""
    implementation: str = ""
    timeout_seconds: int = 60
    retry_count: int = 1
    # Layer 4: False = the tool writes, deletes or dispatches.
    read_only: bool = True
    is_active: bool = True


# --- Policy ---
class PolicyCreate(BaseModel):
    id: str
    name: str
    tenant: str = "default"
    allowed_capabilities: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    max_executions_per_day: int = 100
    is_active: bool = True


class PolicyRead(PolicyCreate):
    created_at: datetime | None = None


# --- Execution ---
class ExecutionCreate(BaseModel):
    mission_id: str
    input: dict[str, Any] = Field(default_factory=dict)
    # Option B session (docs/sessions.md): optional, client-supplied.
    # Omitted -> the service generates a fresh sess_ id, so the run
    # behaves exactly as before (its own one-turn session).
    session_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$"
    )


class ExecutionRead(BaseModel):
    execution_id: str
    mission_id: str
    mission_version: int
    # Which agent config ran: agent_id from the mission, agent_version /
    # agent_version_drift recorded by the worker (Option C). Without
    # these the API could not say which version produced a result.
    agent_id: str | None = None
    agent_version: int | None = None
    agent_version_drift: bool = False
    session_id: str | None = None
    status: str
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


# --- Permission requests (Option C) ---
class PermissionRequestRead(BaseModel):
    request_id: str
    execution_id: str | None = None
    mission_id: str | None = None
    agent_id: str | None = None
    tool_name: str
    target: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    status: str
    decision_note: str | None = None
    created_at: datetime | None = None
    decided_at: datetime | None = None
    expires_at: datetime | None = None
    # Set on approve when the original execution was re-dispatched.
    redispatched_execution_id: str | None = None


class PermissionDecision(BaseModel):
    note: str | None = None


# --- Session (Option B) ---
class SessionRead(BaseModel):
    session_id: str
    agent_id: str | None = None
    mission_id: str
    execution_count: int
    first_created_at: datetime | None = None
    last_created_at: datetime | None = None
    # The session's executions, oldest first (conversation order).
    executions: list[ExecutionRead] = Field(default_factory=list)


class ExecutionResultRead(BaseModel):
    execution_id: str
    output: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Any] = Field(default_factory=list)
    created_at: datetime | None = None


# --- Schedule ---
class ScheduleCreate(BaseModel):
    mission_id: str
    cron: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    tenant: str = "default"


class ScheduleRead(ScheduleCreate):
    id: str
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ScheduleUpdate(BaseModel):
    cron: str | None = None
    input_data: dict[str, Any] | None = None
    enabled: bool | None = None
    tenant: str | None = None


# --- Misc ---
class HealthResponse(BaseModel):
    status: str = "ok"
    app: str = "agentic-runtime"
    version: str = "0.1.0"
    environment: str = "dev"


class ModelInfo(BaseModel):
    provider: str
    model: str
    default: bool = False
    note: str = ""


# --- Ontology ---
class OntologyEntityRead(BaseModel):
    entity_type: str
    entity_key: str
    properties: dict[str, Any] = Field(default_factory=dict)
    status: str
    version: int
    state_hash: str | None = None


class EntityRelationshipRead(BaseModel):
    from_id: str
    rel_type: str
    to_id: str
    to_label: str | None = None


class KnowledgeDocumentLink(BaseModel):
    document_id: str | None = None
    rel_type: str | None = None


class KnowledgeLinksRead(BaseModel):
    entity_type: str
    entity_key: str
    documents: list[KnowledgeDocumentLink] = Field(default_factory=list)


class OntologySubgraphRead(BaseModel):
    root_type: str
    root_key: str
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)


class ProjectionDrainResult(BaseModel):
    processed: int = 0
    errors: int = 0
    dead: int = 0
    claimed: int | None = None
    detail: str | None = None
