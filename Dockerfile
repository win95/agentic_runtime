# API + worker image (the worker overrides CMD in docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Pins the /app checkout as the import source. The image also
    # pip-installs the code (pyproject packages.find includes
    # runtime*), and skills discovery derives DEFAULT_SKILLS_DIR from
    # __file__ — the site-packages copy has no skills/ next to it,
    # so without this load_skills() silently returns zero skills
    # (v5 review, point 5).
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

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

RUN pip install --upgrade pip && pip install .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
