"""User service app — FastAPI entrypoint.

Run locally: ``uvicorn services.user_service.main:app --port 8001``
(in the repo root, with the user-service requirements installed).
In Docker: the ``user-service`` container in docker-compose.yml.
"""

from __future__ import annotations

from fastapi import FastAPI

from services.user_service.routers import admin, auth


def create_app() -> FastAPI:
    app = FastAPI(title="user-service", version="0.1.0")
    app.include_router(auth.router)
    app.include_router(admin.router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
