"""Settings for the user service — everything comes from the environment.

The service is self-contained: it reads only its own env vars and never
imports the platform app. The JWT secret is the ONE value shared with
the platform API (``JWT_SECRET``): the service signs tokens with it, the
API verifies them. Keep the two values identical wherever both run.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Own database on the shared Postgres server (see
    # docker/postgres/01-create-user-service-db.sql). Never the platform
    # app's DATABASE_URL — the service owns its tables and migrations.
    user_service_database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/user_service"
    )

    # Shared with the platform API (app/config.py: jwt_secret). Empty =
    # JWT auth is not configured; login refuses to issue tokens.
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24

    # One-off admin seed (services.user_service.seed_admin). Empty =
    # seeding refuses to run; set them in the environment, never in code.
    admin_email: str = ""
    admin_password: str = ""


settings = Settings()
