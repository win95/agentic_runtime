# Lark frame image — the WS/webhook bridge between Lark and the platform
# (services/lark_frame). Build context is the repo root; run it from
# docker-compose.dev.yml, not from the main compose (the platform stays
# on the host in the dev flow).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/lark_frame/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY services/__init__.py ./services/__init__.py
COPY services/lark_frame ./services/lark_frame

EXPOSE 8002

CMD ["python", "-m", "services.lark_frame"]
