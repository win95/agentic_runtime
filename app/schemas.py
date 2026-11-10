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


class AgentRead(AgentCreate):
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


class ExecutionRead(BaseModel):
    execution_id: str
    mission_id: str
    mission_version: int
    status: str
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


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
