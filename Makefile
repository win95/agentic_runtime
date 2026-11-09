.PHONY: install install-requirements run-api run-worker test lint docker-up docker-down demo eval eval-real eval-genai eval-genai-real langgraph-dev

# Prefer the project venv when it exists (this checkout), else system python.
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

install:
	pip install -e ".[dev]"

# Pinned snapshot for reproducible local installs (see requirements.txt).
install-requirements:
	pip install -r requirements.txt

run-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-worker:
	celery -A app.worker.celery_app worker --loglevel=info

test:
	pytest -q

lint:
	ruff check app runtime infra tests seeds scripts

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

# End-to-end smoke test against a running API (needs the 4 LiteLLM vars).
demo:
	$(PYTHON) scripts/smoke_e2e.py

# MLflow eval, mock LLM (no key/proxy needed).
eval:
	$(PYTHON) scripts/eval_mission.py --mock-llm

# MLflow eval with the real LiteLLM model (needs the 4 LiteLLM vars).
eval-real:
	$(PYTHON) scripts/eval_mission.py --real

# GenAI eval: classic eval + mlflow.genai.evaluate with a judge routed
# through the LiteLLM proxy (needs LLM_API_BASE/LLM_API_KEY; predictions
# still use the mock LLM unless --real is passed).
eval-genai:
	$(PYTHON) scripts/eval_mission.py --mock-llm --genai

# GenAI eval with real predictions (needs the 4 LiteLLM vars).
eval-genai-real:
	$(PYTHON) scripts/eval_mission.py --real --genai

# LangGraph dev server (Studio + traces) — http://localhost:8123
langgraph-dev:
	langgraph dev --allow-blocking
