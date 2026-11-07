"""Mission Manager — CRUD over Mission config."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import schemas
from app.core.db import get_db
from app.core.security import require_api_key
from app.services import mission_service

router = APIRouter(
    prefix="/missions", tags=["missions"], dependencies=[Depends(require_api_key)]
)


@router.post("", response_model=schemas.MissionRead, status_code=201)
def create_mission(payload: schemas.MissionCreate, db: Session = Depends(get_db)):
    mission = mission_service.create_mission(db, payload.model_dump())
    return mission


@router.get("", response_model=list[schemas.MissionRead])
def list_missions(db: Session = Depends(get_db)):
    return mission_service.list_missions(db)


@router.get("/{mission_id}", response_model=schemas.MissionRead)
def get_mission(mission_id: str, db: Session = Depends(get_db)):
    mission = mission_service.get_mission(db, mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    return mission


@router.delete("/{mission_id}", status_code=204)
def delete_mission(mission_id: str, db: Session = Depends(get_db)):
    if not mission_service.delete_mission(db, mission_id):
        raise HTTPException(status_code=404, detail="Mission not found")
    return None
