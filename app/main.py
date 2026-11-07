"""FastAPI application factory — Platform API entrypoint (Layer 1).

The API manages configuration and execution lifecycle. It never runs an
agent inline; execution runs in the Celery worker (Layer 2).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__, schemas
from app.config import settings
from app.core.db import init_db
from app.routers import agents, executions, missions, models, policies, results, tools
from infra.langsmith_setup import setup_langsmith

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception as exc:  # DB down in dev — API still starts
        logger.warning("init_db failed (DB may be unavailable): %s", exc)
    if setup_langsmith():
        logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    else:
        logger.info("LangSmith tracing disabled.")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic Runtime", version=__version__, lifespan=lifespan)
    @app.get("/health", response_model=schemas.HealthResponse, tags=["health"])
    def health() -> schemas.HealthResponse:
        return schemas.HealthResponse(
            status="ok",
            app=settings.app_name,
            version=__version__,
            environment=settings.environment,
        )

    app.include_router(missions.router)
    app.include_router(agents.router)
    app.include_router(tools.router)
    app.include_router(models.router)
    app.include_router(policies.router)
    app.include_router(executions.router)
    app.include_router(results.router)
    return app


app = create_app()
