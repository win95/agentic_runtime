"""Agent Manager — CRUD over Agent config (model, prompt, allowed tools)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import models, schemas
from app.core.db import get_db
from app.core.security import require_api_key

router = APIRouter(
    prefix="/agents", tags=["agents"], dependencies=[Depends(require_api_key)]
)


@router.post("", response_model=schemas.AgentRead, status_code=201)
def create_agent(payload: schemas.AgentCreate, db: Session = Depends(get_db)):
    if db.get(models.Agent, payload.id) is not None:
        raise HTTPException(
            status_code=409, detail=f"Agent {payload.id} already exists"
        )
    agent = models.Agent(**payload.model_dump())
    db.add(agent)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409, detail=f"Agent {payload.id} already exists"
        ) from None
    db.refresh(agent)
    return agent


@router.get("", response_model=list[schemas.AgentRead])
def list_agents(db: Session = Depends(get_db)):
    return db.query(models.Agent).order_by(models.Agent.id).all()


@router.get("/{agent_id}", response_model=schemas.AgentRead)
def get_agent(agent_id: str, db: Session = Depends(get_db)):
    agent = db.get(models.Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.delete("/{agent_id}", status_code=204)
def delete_agent(agent_id: str, db: Session = Depends(get_db)):
    agent = db.get(models.Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    # Agent versions reference the agent (agent_versions.agent_id FK);
    # deleting the agent would orphan its version history, so refuse
    # with 409 — same semantics as mission delete (no cascade).
    version_count = (
        db.query(models.AgentVersion)
        .filter(models.AgentVersion.agent_id == agent_id)
        .count()
    )
    if version_count > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Agent has {version_count} version(s) "
                "and cannot be deleted"
            ),
        )
    db.delete(agent)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        count = (
            db.query(models.AgentVersion)
            .filter(models.AgentVersion.agent_id == agent_id)
            .count()
        )
        if count > 0:
            raise HTTPException(
                status_code=409,
                detail=f"Agent has {count} version(s) and cannot be deleted",
            ) from None
        raise
    return None
