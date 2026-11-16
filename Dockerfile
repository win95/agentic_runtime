# API + worker image (the worker overrides CMD in docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "langgraph-checkpoint-postgres>=2.0,<4"

COPY pyproject.toml README.md ./
COPY app ./app
COPY runtime ./runtime
COPY infra ./infra
COPY ingest ./ingest
COPY ontology ./ontology
COPY seeds ./seeds
COPY skills ./skills
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini

RUN pip install --no-deps -e . && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]