"""Tool Registry API — list the tools the runtime can call.

The registry is the source of the callable tools (see runtime/tools);
the DB rows hold the registered metadata. Runtime permission is enforced
per execution context (mission/tenant), not by this listing.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import models, schemas
from app.core.db import get_db
from app.core.security import require_api_key
from runtime.tools.registry import registry

router = APIRouter(
    prefix="/tools", tags=["tools"], dependencies=[Depends(require_api_key)]
)


@router.get("", response_model=list[schemas.ToolRead])
def list_tools(db: Session = Depends(get_db)):
    """Tools registered in the runtime registry (the callable set)."""
    return [
        schemas.ToolRead(
            id=entry.name,
            name=entry.name,
            description=entry.description,
            capability=entry.capability,
            input_schema=entry.input_schema,
            output_schema={},
            permission=entry.permission,
            implementation=entry.implementation,
            timeout_seconds=entry.timeout_seconds,
            retry_count=entry.retry_count,
            is_active=True,
        )
        for entry in registry.list()
    ]


@router.get("/db", response_model=list[schemas.ToolRead])
def list_db_tools(db: Session = Depends(get_db)):
    """Tool metadata rows stored in PostgreSQL (if any were registered)."""
    return db.query(models.Tool).order_by(models.Tool.name).all()
