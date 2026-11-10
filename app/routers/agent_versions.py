"""Agent versions — list and activate immutable agent configs (Option C).

A version pins the agent's system prompt, model, temperature, allowed
tools and tool manifest in one row. Rollback is activating an older
version; creating a version is done by the seed loader or by
``scripts/register_agent_version.py``, not by this router.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import models, schemas
from app.core.db import get_db
from app.core.security import require_api_key
from app.services import agent_version_service

router = APIRouter(
    prefix="/agents/{agent_id}/versions",
    tags=["agent_versions"],
    dependencies=[Depends(require_api_key)],
)


def _require_agent(db: Session, agent_id: str) -> models.Agent:
    agent = db.get(models.Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.get("", response_model=list[schemas.AgentVersionRead])
def list_versions(agent_id: str, db: Session = Depends(get_db)):
    _require_agent(db, agent_id)
    return agent_version_service.list_versions(db, agent_id)


@router.get("/{version}", response_model=schemas.AgentVersionRead)
def get_version(agent_id: str, version: int, db: Session = Depends(get_db)):
    _require_agent(db, agent_id)
    row = agent_version_service.get_version(db, agent_id, version)
    if row is None:
        raise HTTPException(status_code=404, detail="Agent version not found")
    return row


@router.post("/{version}/activate", response_model=schemas.AgentVersionRead)
def activate_version(agent_id: str, version: int, db: Session = Depends(get_db)):
    _require_agent(db, agent_id)
    try:
        return agent_version_service.activate_version(db, agent_id, version)
    except agent_version_service.AgentVersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
