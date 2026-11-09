"""Schedule Manager — CRUD over recurring mission runs (cron)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import schemas
from app.core.db import get_db
from app.core.security import require_api_key
from app.services import schedule_service

router = APIRouter(
    prefix="/schedules", tags=["schedules"], dependencies=[Depends(require_api_key)]
)


@router.post("", response_model=schemas.ScheduleRead, status_code=201)
def create_schedule(payload: schemas.ScheduleCreate, db: Session = Depends(get_db)):
    try:
        schedule = schedule_service.create_schedule(db, payload.model_dump())
    except schedule_service.InvalidCronError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except schedule_service.ScheduleMissionNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schedule


@router.get("", response_model=list[schemas.ScheduleRead])
def list_schedules(db: Session = Depends(get_db)):
    return schedule_service.list_schedules(db)


@router.get("/{schedule_id}", response_model=schemas.ScheduleRead)
def get_schedule(schedule_id: str, db: Session = Depends(get_db)):
    schedule = schedule_service.get_schedule(db, schedule_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule


@router.patch("/{schedule_id}", response_model=schemas.ScheduleRead)
def update_schedule(
    schedule_id: str, payload: schemas.ScheduleUpdate, db: Session = Depends(get_db)
):
    try:
        schedule = schedule_service.update_schedule(
            db, schedule_id, payload.model_dump(exclude_unset=True)
        )
    except schedule_service.InvalidCronError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if schedule is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule


@router.post("/{schedule_id}/enable", response_model=schemas.ScheduleRead)
def enable_schedule(schedule_id: str, db: Session = Depends(get_db)):
    schedule = schedule_service.set_enabled(db, schedule_id, True)
    if schedule is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule


@router.post("/{schedule_id}/disable", response_model=schemas.ScheduleRead)
def disable_schedule(schedule_id: str, db: Session = Depends(get_db)):
    schedule = schedule_service.set_enabled(db, schedule_id, False)
    if schedule is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule


@router.delete("/{schedule_id}", status_code=204)
def delete_schedule(schedule_id: str, db: Session = Depends(get_db)):
    if not schedule_service.delete_schedule(db, schedule_id):
        raise HTTPException(status_code=404, detail="Schedule not found")
    return None
