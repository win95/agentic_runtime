.PHONY: install run-api run-worker test lint docker-up docker-down seed demo eval eval-real langgraph-dev

install:
	pip install -e ".[dev]"

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

seed:
	python -m seeds.seed_demo_data

# End-to-end smoke test against a running API (needs the 4 LiteLLM vars).
demo:
	python scripts/smoke_e2e.py

# MLflow eval, mock LLM (no key/proxy needed).
eval:
	python scripts/eval_mission.py --mock-llm

# MLflow eval with the real LiteLLM model (needs the 4 LiteLLM vars).
eval-real:
	python scripts/eval_mission.py --real

# LangGraph dev server (Studio + traces) — http://localhost:8123
langgraph-dev:
	langgraph dev --allow-blocking
