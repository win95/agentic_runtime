"""Policy Manager — CRUD over access policies (tenant -> capabilities/tools)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import models, schemas
from app.core.db import get_db
from app.core.security import require_api_key

router = APIRouter(
    prefix="/policies", tags=["policies"], dependencies=[Depends(require_api_key)]
)


@router.post("", response_model=schemas.PolicyRead, status_code=201)
def create_policy(payload: schemas.PolicyCreate, db: Session = Depends(get_db)):
    if db.get(models.Policy, payload.id) is not None:
        raise HTTPException(
            status_code=409, detail=f"Policy {payload.id} already exists"
        )
    policy = models.Policy(**payload.model_dump())
    db.add(policy)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409, detail=f"Policy {payload.id} already exists"
        ) from None
    db.refresh(policy)
    return policy


@router.get("", response_model=list[schemas.PolicyRead])
def list_policies(db: Session = Depends(get_db)):
    return db.query(models.Policy).order_by(models.Policy.id).all()


@router.get("/{policy_id}", response_model=schemas.PolicyRead)
def get_policy(policy_id: str, db: Session = Depends(get_db)):
    policy = db.get(models.Policy, policy_id)
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy


@router.delete("/{policy_id}", status_code=204)
def delete_policy(policy_id: str, db: Session = Depends(get_db)):
    policy = db.get(models.Policy, policy_id)
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    db.delete(policy)
    db.commit()
    return None
